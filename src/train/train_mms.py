"""
Training utilities for MMS ASR adapter fine-tuning experiments.

This module is designed to be imported into a Colab or Jupyter notebook.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from evaluate import load
from transformers import TrainingArguments, Wav2Vec2Processor

WER_METRIC = load("wer")
CER_METRIC = load("cer")

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def build_training_args(config: dict, output_dir, hub_model_id, **overrides) -> TrainingArguments:
    hub_cfg = {k: v for k, v in config["hub"].items() if k != "report_to"}
    return TrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        report_to=config["hub"]["report_to"],
        **hub_cfg,
        **config["training"],
        **config["evaluation"],
        **overrides,
    )

def prepare_mms_batch(batch, processor, text_column="sentence"):
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch[text_column]).input_ids
    return batch

def compute_mms_corpus_metrics(pred, processor, training_mode: bool = False):
    """
    Compute corpus-level WER/CER for MMS CTC model predictions.

    training_mode=True  → raw WER only (0-1), for Trainer's metric_for_best_model
    training_mode=False → WER and CER as percentages, for standalone evaluation
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = WER_METRIC.compute(predictions=pred_str, references=label_str)

    if training_mode:
        return {"wer": wer}

    cer = CER_METRIC.compute(predictions=pred_str, references=label_str)
    return {"wer": wer * 100, "cer": cer * 100}

def run_evaluation(
    model,
    processor,
    dataset,
    duration_label: str,
    results_dir,
    batch_size: int = 8,
    model_id: Optional[str] = None,
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
        batch = dataset_eval[start : start + batch_size]
        padded = processor.pad(
            [{"input_values": iv} for iv in batch["input_values"]],
            padding=True,
            return_tensors="pt",
        )
        input_values = padded.input_values.to(model.device)
        attention_mask = padded.attention_mask.to(model.device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predictions.extend(processor.batch_decode(predicted_ids))

    results_df = pd.DataFrame({
        "model_id":    model_id,
        "audio_fname": dataset_eval["audio_fname"],
        "sentence":    dataset_eval["sentence"],
        "prediction":  predictions,
    })

    results_df["wer_utterance"] = [
        100 * WER_METRIC.compute(predictions=[p], references=[r])
        for p, r in zip(results_df["prediction"], results_df["sentence"])
    ]
    results_df["wer_avg"] = 100 * WER_METRIC.compute(
        predictions=results_df["prediction"].tolist(),
        references=results_df["sentence"].tolist(),
    )
    results_df["cer_avg"] = 100 * CER_METRIC.compute(
        predictions=results_df["prediction"].tolist(),
        references=results_df["sentence"].tolist(),
    )

    results_df.to_csv(output_csv, index=False)
    print(f"  WER (corpus): {results_df['wer_avg'].iloc[0]:.2f}%   CER (corpus): {results_df['cer_avg'].iloc[0]:.2f}%")
    print(f"  Predictions saved: {output_csv}")
    return results_df
