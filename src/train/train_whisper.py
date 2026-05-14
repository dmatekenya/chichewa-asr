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
from transformers import Seq2SeqTrainingArguments

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

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    if text_column is not None:
        batch["labels"] = processor.tokenizer(
            batch[text_column]
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
    hub_cfg = {k: v for k, v in config["hub"].items() if k != "report_to"}
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        report_to=config["hub"]["report_to"],
        **hub_cfg,
        **config["training"],
        **config["evaluation"],
    )

def run_evaluation(
    model,
    processor,
    dataset: Dataset,
    duration_label: str,
    results_dir,
    batch_size: int = 8,
    model_id: str = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Run inference on the held-out test set, compute WER/CER, and save predictions.

    Returns a DataFrame with columns: model_id, audio_fname, reference, prediction,
    wer_utterance, wer_avg, cer_avg.
    """
    output_csv  = Path(results_dir) / f"predictions_{duration_label}.csv"
    predictions = []
    model.eval()

    dataset_eval = dataset
    if debug:
        print("[DEBUG] Running evaluation on a small sample of the test set.")
        sample_size  = min(16, len(dataset))
        dataset_eval = dataset.select(range(sample_size)) if hasattr(dataset, "select") else dataset[:sample_size]

    for start in range(0, len(dataset_eval), batch_size):
        batch          = dataset_eval[start : start + batch_size]
        input_features = torch.tensor(batch["input_features"], device=model.device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        predictions.extend(processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True))

    results_df = pd.DataFrame({
        "model_id":    model_id,
        "audio_fname": dataset_eval["audio_fname"],
        "reference":   dataset_eval["sentence"],
        "prediction":  predictions,
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
    print(f"  WER (corpus): {results_df['wer_avg'].iloc[0]:.2f}%   CER (corpus): {results_df['cer_avg'].iloc[0]:.2f}%")
    print(f"  Predictions saved: {output_csv}")
    return results_df