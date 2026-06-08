"""
Training utilities for Whisper ASR experiments.

This module is designed to be imported into a Colab or Jupyter notebook.
The notebook controls the experiment flow, while this script provides reusable
utilities for loading configuration files, preparing Whisper datasets, defining
the data collator, computing ASR metrics, building training arguments, and
running final hold-out evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import pandas as pd
import torch
import yaml
from datasets import Audio, Dataset
from transformers import EarlyStoppingCallback, Seq2SeqTrainingArguments

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_whisper_batch(
    batch,
    processor,
    audio_column: str,
    text_column: Optional[str] = None,
):
    audio = batch[audio_column]

    processed = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=True,
    )

    batch["input_features"] = processed.input_features[0]

    if "attention_mask" in processed:
        batch["attention_mask"] = processed.attention_mask[0]

    if text_column is not None:
        batch["labels"] = processor.tokenizer(
            batch[text_column],
            truncation=True,
            max_length=448,
        ).input_ids

    return batch
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_asr_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True,
    )

    label_str = processor.tokenizer.batch_decode(
        label_ids,
        skip_special_tokens=True,
    )

    wer = 100 * wer_metric.compute(
        predictions=pred_str,
        references=label_str,
    )

    cer = 100 * cer_metric.compute(
        predictions=pred_str,
        references=label_str,
    )

    return {
        "wer": wer,
        "cer": cer,
    }

def build_training_args(config: dict, output_dir, hub_model_id) -> Seq2SeqTrainingArguments:
    """
    Build Seq2SeqTrainingArguments from a config dict.

    The following settings are enforced regardless of what the config says,
    because they must be correct for early stopping and best-checkpoint
    loading to work reliably on small datasets:

    - ``load_best_model_at_end=True``
    - ``metric_for_best_model="wer"``
    - ``greater_is_better=False``
    - ``predict_with_generate=True``
    - ``save_strategy`` is aligned with ``eval_strategy``
    - ``save_steps`` is aligned with ``eval_steps`` when using step-based eval
    """
    hub_cfg = {k: v for k, v in config["hub"].items() if k != "report_to"}
    merged = {**hub_cfg, **config["training"], **config["evaluation"]}

    # Align save and eval strategies — misalignment causes early stopping to
    # lag and load_best_model_at_end to track the wrong checkpoint.
    eval_strategy = merged.get("eval_strategy", merged.get("evaluation_strategy", "steps"))
    merged["eval_strategy"]  = eval_strategy
    merged["save_strategy"]  = eval_strategy

    if eval_strategy == "steps":
        eval_steps = merged.get("eval_steps", 200)
        merged["eval_steps"] = eval_steps
        merged["save_steps"] = eval_steps

    # These must always be set correctly — override any config value.
    merged["load_best_model_at_end"] = True
    merged["metric_for_best_model"]  = "wer"
    merged["greater_is_better"]      = False
    merged["predict_with_generate"]  = True

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        report_to=config["hub"]["report_to"],
        **merged,
    )

def build_callbacks(config: dict) -> List[EarlyStoppingCallback]:
    """
    Build the callback list for the Trainer.

    Returns an ``EarlyStoppingCallback`` configured from
    ``config["early_stopping"]`` if that section exists, otherwise uses
    safe defaults (patience=5, threshold=0.002 WER points).

    Pass the returned list directly to ``Trainer(callbacks=...)``.
    """
    es_cfg = config.get("early_stopping", {})
    patience  = es_cfg.get("patience", 5)
    threshold = es_cfg.get("threshold", 0.002)
    return [EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
    )]

def run_evaluation(
    model,
    processor,
    dataset: Dataset,
    duration_label: str,
    results_dir,
    batch_size: int = 8,
    model_id: str = None,
    debug: bool = False,
    language: str = "shona",
    task: str = "transcribe",
) -> pd.DataFrame:
    """
    Run inference on the held-out test set, compute WER/CER, and save predictions.

    Returns a DataFrame with columns: model_id, audio_fname, reference, prediction,
    wer_utterance, wer_avg, cer_avg.
    """
    output_csv = Path(results_dir) / f"predictions_{duration_label}.csv"
    predictions = []
    model.eval()

    # Explicit Whisper generation setup
    if language is not None:
        processor.tokenizer.set_prefix_tokens(language=language, task=task)
        model.generation_config.language = language

    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None

    dataset_eval = dataset
    if debug:
        print("[DEBUG] Running evaluation on a small sample of the test set.")
        sample_size = min(16, len(dataset))
        dataset_eval = dataset.select(range(sample_size)) if hasattr(dataset, "select") else dataset[:sample_size]

    for start in range(0, len(dataset_eval), batch_size):
        batch = dataset_eval[start : start + batch_size]

        input_features = torch.tensor(
            batch["input_features"],
            device=model.device,
            dtype=model.dtype if hasattr(model, "dtype") else torch.float32,
        )

        generate_kwargs = {
            "input_features": input_features,
        }

        if "attention_mask" in batch:
            generate_kwargs["attention_mask"] = torch.tensor(
                batch["attention_mask"],
                device=model.device,
            )

        with torch.no_grad():
            predicted_ids = model.generate(**generate_kwargs)

        predictions.extend(
            processor.tokenizer.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )
        )

    results_df = pd.DataFrame({
        "model_id": model_id,
        "audio_fname": dataset_eval["audio_fname"],
        "reference": dataset_eval["sentence"],
        "prediction": predictions,
    })

    results_df["wer_utterance"] = [
        100 * wer_metric.compute(predictions=[p], references=[r])
        for p, r in zip(results_df["prediction"], results_df["reference"])
    ]

    results_df["wer_avg"] = 100 * wer_metric.compute(
        predictions=results_df["prediction"].tolist(),
        references=results_df["reference"].tolist(),
    )

    results_df["cer_avg"] = 100 * cer_metric.compute(
        predictions=results_df["prediction"].tolist(),
        references=results_df["reference"].tolist(),
    )

    results_df.to_csv(output_csv, index=False)

    print(
        f"  WER (corpus): {results_df['wer_avg'].iloc[0]:.2f}%   "
        f"CER (corpus): {results_df['cer_avg'].iloc[0]:.2f}%"
    )
    print(f"  Predictions saved: {output_csv}")

    return results_df