"""
Training utilities for Whisper ASR experiments.

This module is designed to be imported into a Colab or Jupyter notebook.
The notebook controls the experiment flow, while this script provides reusable
utilities for loading configuration files, preparing Whisper datasets, defining
the data collator, computing ASR metrics, building training arguments, and
running final hold-out evaluation.
"""

from dataclasses import dataclass
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
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        push_to_hub=config["hub"]["push_to_hub"],
        report_to=config["hub"]["report_to"],
        **config["training"],
        **config["evaluation"],
    )

def evaluate_holdout_set(
    model,
    processor,
    dataset: Dataset,
    text_column: str = "sentence",
    fname_column: str = "audio_fname",
    output_csv=None,
    batch_size: int = 8,
):
    """
    Run inference on a prepared hold-out ASR dataset and compute WER/CER.

    Parameters
    ----------
    model : transformers model
        Fine-tuned Whisper model.
    processor : transformers processor
        Whisper processor.
    dataset : datasets.Dataset
        Prepared hold-out test dataset with 'input_features', text_column,
        and fname_column retained from preprocessing.
    text_column : str
        Column containing reference transcripts (default: 'sentence').
    fname_column : str
        Column containing audio filenames for traceability (default: 'audio_fname').
    output_csv : str or Path, optional
        Path to save predictions and references.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict
        Dictionary with WER, CER, and dataframe of predictions.
    """
    predictions = []
    model.eval()

    for start_idx in range(0, len(dataset), batch_size):
        batch = dataset[start_idx : start_idx + batch_size]

        input_features = torch.tensor(
            batch["input_features"],
            device=model.device,
        )

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        pred_text = processor.tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )

        predictions.extend(pred_text)

    results_df = pd.DataFrame({
        fname_column:  dataset[fname_column],
        "reference":   dataset[text_column],
        "prediction":  predictions,
    })

    wer = 100 * wer_metric.compute(
        predictions=results_df["prediction"].tolist(),
        references=results_df["reference"].tolist(),
    )

    cer = 100 * cer_metric.compute(
        predictions=results_df["prediction"].tolist(),
        references=results_df["reference"].tolist(),
    )

    if output_csv is not None:
        results_df.to_csv(output_csv, index=False)

    return {
        "wer": wer,
        "cer": cer,
        "predictions": results_df,
    }