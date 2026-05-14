"""
whisper_duration_experiment.py
-------------------------------
Utilities for Whisper duration-sweep experiments.

Provides everything the notebook needs to import:
  - Config helpers   : build_run_config
  - Model loading    : load_model_and_processor
  - Data preparation : prepare_train_dataset, prepare_test_dataset
  - Training         : run_training
  - Evaluation       : run_evaluation

The experiment loop (including Hub upload) lives in the notebook so
the student can inspect intermediate state between steps.

Typical notebook usage
----------------------
    from src.train.train_whisper import load_config
    from src.train.whisper_duration_experiment import (
        build_run_config,
        load_model_and_processor,
        prepare_train_dataset,
        prepare_test_dataset,
        run_training,
        run_evaluation,
    )
"""

from functools import partial
from pathlib import Path

import torch
from datasets import Audio, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.data_utils.data_utils import load_audio_data
from src.train.train_whisper import (
    DataCollatorSpeechSeq2SeqWithPadding,
    build_training_args,
    compute_asr_metrics,
    prepare_whisper_batch,
    run_evaluation,
)


def load_model_and_processor(config: dict):
    """
    Load a fresh Whisper model and processor from the Hub.
    Always loads from the base checkpoint so every run is independent.

    Returns
    -------
    (model, processor)
    """
    model_id = config["model"]["model_name_or_path"]
    language = config["model"]["language"]
    task     = config["model"]["task"]

    print(f"  Loading Whisper model: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)

    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model.generation_config.language           = language
    model.generation_config.task               = task
    model.generation_config.forced_decoder_ids = None

    return model, processor

def prepare_train_dataset(manifest_path, audio_dir, processor, cache_dir=None, num_proc=4):
    """
    Load a duration manifest and map audio + text to Whisper input features.
    If cache_dir is provided, saves the processed dataset on first run and
    reloads from disk on subsequent runs.

    Returns
    -------
    DatasetDict with 'train' and 'validation' splits, features mapped.
    """
    if cache_dir is not None and Path(cache_dir).exists():
        print(f"  Loading train data from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"  Loading train data: {manifest_path}")
    raw = load_audio_data(manifest_path, audio_dir=audio_dir)
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))

    dataset = raw.map(
        lambda batch: prepare_whisper_batch(
            batch, processor=processor, audio_column="audio", text_column="sentence"
        ),
        remove_columns=raw["train"].column_names,
        num_proc=num_proc,
    )

    if cache_dir is not None:
        print(f"  Saving processed dataset to cache: {cache_dir}")
        dataset.save_to_disk(str(cache_dir))

    return dataset

def prepare_test_dataset(
    manifest_path,
    audio_dir,
    base_config: dict,
    audio_fname_col: str = "audio_filename",
    duration_col: str = "duration_seconds",
    cache_dir=None,
):
    """
    Pre-process the held-out test set once before the sweep loop.
    Whisper log-mel features are model-agnostic so the result is safe
    to reuse across all runs. If cache_dir is provided, saves on first
    run and reloads from disk on subsequent runs.

    Returns
    -------
    Dataset with 'input_features', 'sentence', 'audio_fname'.
    """
    if cache_dir is not None and Path(cache_dir).exists():
        print(f"  Loading test data from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"  Loading test data: {manifest_path}")
    model_id  = base_config["model"]["model_name_or_path"]
    processor = WhisperProcessor.from_pretrained(
        model_id,
        language=base_config["model"]["language"],
        task=base_config["model"]["task"],
    )

    raw = load_audio_data(
        manifest_path,
        audio_dir=audio_dir,
        audio_fname_col=audio_fname_col,
        split_data=False,
        duration_col=duration_col,
    )
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))

    cols_to_remove = [c for c in raw.column_names if c != "audio"]
    dataset = raw.map(
        lambda batch: {
            **prepare_whisper_batch(
                batch, processor=processor, audio_column="audio", text_column="sentence"
            ),
            "audio_fname": batch["audio_fname"],
        },
        remove_columns=cols_to_remove,
        num_proc=1,
    )

    if cache_dir is not None:
        print(f"  Saving processed test dataset to cache: {cache_dir}")
        dataset.save_to_disk(str(cache_dir))

    return dataset

def run_training(model, processor, dataset_train, run_config: dict, hub_model_id, output_dir):
    """
    Configure and run Seq2SeqTrainer.

    Returns
    -------
    Trained Seq2SeqTrainer (model weights updated in-place).
    """
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=build_training_args(run_config, output_dir, hub_model_id),
        train_dataset=dataset_train["train"],
        eval_dataset=dataset_train["validation"],
        data_collator=data_collator,
        compute_metrics=partial(compute_asr_metrics, processor=processor),
        processing_class=processor.feature_extractor,
    )

    print("  Training ...")
    trainer.train()
    return trainer

