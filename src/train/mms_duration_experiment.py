"""
mms_duration_experiment.py
--------------------------
Utilities for MMS adapter fine-tuning duration-sweep experiments.

Mirrors whisper_duration_experiment.py but for MMS (facebook/mms-1b-all).

Provides everything the notebook/script needs:
  - Processor building : build_processor
  - Model loading      : load_model_and_processor
  - Data preparation   : prepare_train_dataset, prepare_test_dataset
  - Training           : run_training
  - Evaluation         : run_evaluation (re-exported from train_mms)

Typical usage
-------------
    from src.train.train_whisper import load_config
    from src.train.mms_duration_experiment import (
        build_processor,
        load_model_and_processor,
        prepare_train_dataset,
        prepare_test_dataset,
        run_training,
        run_evaluation,
    )
"""

import json
from functools import partial
from pathlib import Path

import torch
from datasets import Audio, load_from_disk
from transformers import (
    Trainer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from src.data_utils.data_utils import extract_all_chars, load_audio_data, remove_special_characters
from transformers import EarlyStoppingCallback

from src.train.train_mms import (
    DataCollatorCTCWithPadding,
    build_training_args,
    compute_mms_corpus_metrics,
    prepare_mms_batch,
    run_evaluation,
)

# Punctuation/noise characters to strip before building vocabulary.
# Keeps letters, digits, whitespace, and Chichewa-relevant characters.
_CHARS_TO_REMOVE = r'[,?.!\-;:\"\"\"%\'"„«»—…–]|\n'


def _load_processor_from_local(directory) -> Wav2Vec2Processor:
    """
    Load Wav2Vec2Processor from a local directory without touching the Hub.
    Constructs tokenizer and feature extractor directly from their saved config
    files to avoid transformers' Hub path-validation on local absolute paths.
    """
    directory = Path(directory)

    with open(directory / "tokenizer_config.json") as f:
        tok_cfg = json.load(f)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(directory / "vocab.json"),
        unk_token=tok_cfg.get("unk_token", "[UNK]"),
        pad_token=tok_cfg.get("pad_token", "[PAD]"),
        word_delimiter_token=tok_cfg.get("word_delimiter_token", "|"),
        target_lang=tok_cfg.get("target_lang"),
    )

    with open(directory / "processor_config.json") as f:
        proc_cfg = json.load(f)
    fe_cfg = proc_cfg["feature_extractor"]
    fe_cfg.pop("feature_extractor_type", None)
    feature_extractor = Wav2Vec2FeatureExtractor(**fe_cfg)

    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def build_processor(
    manifest_path,
    audio_dir,
    model_id: str,
    target_lang: str,
    save_dir,
) -> Wav2Vec2Processor:
    """
    Build a Wav2Vec2Processor from the training data vocabulary and save to disk.
    If save_dir already exists, loads and returns the saved processor without rebuilding.

    Build once from the full dataset — all duration-sweep experiments reuse this
    processor so the vocabulary stays consistent across runs.

    Parameters
    ----------
    manifest_path : str or Path
        Path to training manifest CSV with a 'sentence' column.
    audio_dir : str or Path
        Directory containing the audio files.
    model_id : str
        HuggingFace model ID (e.g. 'facebook/mms-1b-all').
    target_lang : str
        ISO 639-3 language code (e.g. 'nya' for Chichewa).
    save_dir : str or Path
        Directory to save/load the processor.

    Returns
    -------
    Wav2Vec2Processor
    """
    save_dir = Path(save_dir)
    if (save_dir / "processor_config.json").exists():
        print(f"  Loading processor from: {save_dir}")
        return _load_processor_from_local(save_dir)

    print(f"  Building processor from: {manifest_path}")

    # Load full dataset without splitting (vocab must cover all characters)
    raw = load_audio_data(manifest_path, audio_dir=audio_dir, split_data=False)

    # Clean transcripts using the same regex as the notebook
    clean = lambda batch: remove_special_characters(
        batch, transcript_column="sentence",
        charecters_to_remove_regex=_CHARS_TO_REMOVE,
    )
    raw = raw.map(clean)

    vocab_extracted = raw.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=raw.column_names,
    )
    vocab_list = sorted(set(vocab_extracted["vocab"][0]))

    vocab_dict = {v: i for i, v in enumerate(vocab_list)}

    # Replace space with pipe (CTC word-delimiter convention)
    vocab_dict["|"] = vocab_dict.pop(" ", len(vocab_dict))
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    # Save as nested {target_lang: vocab_dict} so tokenizer can be loaded with target_lang
    save_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = save_dir / "vocab.json"
    with open(vocab_file, "w") as f:
        json.dump({target_lang: vocab_dict}, f)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=target_lang,
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(str(save_dir))
    print(f"  Processor saved to: {save_dir}  (vocab size: {len(vocab_dict)})")
    return processor


def load_model_and_processor(config: dict, processor_dir) -> tuple:
    """
    Load a fresh MMS model and processor.
    Freezes the base model and initialises language-specific adapter layers.

    Returns
    -------
    (model, processor)
    """
    model_id    = config["model"]["model_name_or_path"]
    target_lang = config["model"]["target_lang"]
    processor_dir = Path(processor_dir)

    print(f"  Loading processor from: {processor_dir}")
    processor = _load_processor_from_local(processor_dir)

    print(f"  Loading MMS model: {model_id}")
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )

    model.init_adapter_layers()
    model.freeze_base_model()

    adapter_weights = model.load_adapter(target_lang)
    print(f"  Adapter loaded for lang: {target_lang}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Model device: {device}")

    return model, processor


def prepare_train_dataset(manifest_path, audio_dir, processor, cache_dir=None, num_proc=1):
    """
    Load training manifest and map audio + text to MMS input features.
    Preserves 'input_length' column needed for group_by_length sampling.
    Saves/loads from cache_dir if provided.

    Returns
    -------
    DatasetDict with 'train' and 'validation' splits.
    """
    if cache_dir is not None and Path(cache_dir).exists():
        print(f"  Loading train data from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"  Loading train data: {manifest_path}")
    raw = load_audio_data(manifest_path, audio_dir=audio_dir)
    raw = raw.map(lambda b: remove_special_characters(
        b, transcript_column="sentence", charecters_to_remove_regex=_CHARS_TO_REMOVE,
    ))
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))

    dataset = raw.map(
        lambda batch: prepare_mms_batch(batch, processor=processor, text_column="sentence"),
        remove_columns=raw["train"].column_names,
        num_proc=num_proc,
    )

    if cache_dir is not None:
        print(f"  Saving train dataset to cache: {cache_dir}")
        dataset.save_to_disk(str(cache_dir))

    return dataset


def prepare_test_dataset(
    manifest_path,
    audio_dir,
    processor,
    audio_fname_col: str = "audio_filename",
    duration_col: str = "duration_seconds",
    cache_dir=None,
):
    """
    Pre-process the held-out test set.
    Preserves 'audio_fname' and 'sentence' columns for evaluation.
    Saves/loads from cache_dir if provided.

    Returns
    -------
    Dataset with 'input_values', 'input_length', 'sentence', 'audio_fname'.
    """
    if cache_dir is not None and Path(cache_dir).exists():
        print(f"  Loading test data from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"  Loading test data: {manifest_path}")
    raw = load_audio_data(
        manifest_path,
        audio_dir=audio_dir,
        audio_fname_col=audio_fname_col,
        split_data=False,
        duration_col=duration_col,
    )
    raw = raw.map(lambda b: remove_special_characters(
        b, transcript_column="sentence", charecters_to_remove_regex=_CHARS_TO_REMOVE,
    ))
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))

    dataset = raw.map(
        lambda batch: {
            **prepare_mms_batch(batch, processor=processor, text_column="sentence"),
            "audio_fname": batch["audio_fname"],
            "sentence":    batch["sentence"],
        },
        remove_columns=raw.column_names,  # remove all; lambda defines output exactly
        num_proc=1,
    )

    if cache_dir is not None:
        print(f"  Saving test dataset to cache: {cache_dir}")
        dataset.save_to_disk(str(cache_dir))

    return dataset


def run_training(
    model,
    processor,
    dataset_train,
    run_config: dict,
    hub_model_id,
    output_dir,
    debug: bool = False,
    patience: int = 3,
):
    """
    Configure and run Trainer for MMS CTC adapter fine-tuning.

    Parameters
    ----------
    debug : bool
        Forces CPU training (required on MPS — CTC loss not supported on Apple GPU).
    patience : int
        Early-stopping patience in evaluation steps. Set to 0 to disable.

    Returns
    -------
    Trained Trainer (model weights updated in-place).
    """
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    overrides = {}
    if debug:
        overrides["use_cpu"] = True
        model.to("cpu")
        print("  DEBUG: forcing CPU (MPS does not support CTC loss)")

    training_args = build_training_args(run_config, output_dir, hub_model_id, **overrides)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if patience > 0 else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train["train"],
        eval_dataset=dataset_train["validation"],
        data_collator=data_collator,
        compute_metrics=partial(compute_mms_corpus_metrics, processor=processor, training_mode=True),
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    print("  Training ...")
    trainer.train()
    return trainer
