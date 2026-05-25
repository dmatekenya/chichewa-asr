"""
Multi-GPU training script for MMS dataset duration sweep experiments.

Mirrors scripts/train_dataset_experiment.py but for MMS adapter fine-tuning
(facebook/mms-1b-all with frozen base + language adapter layers).

Usage (single GPU):
    python scripts/train_mms_dataset_experiment.py --duration_label 14h

Usage (multi-GPU):
    accelerate launch --num_processes 2 scripts/train_mms_dataset_experiment.py --duration_label 14h

Usage (debug):
    python scripts/train_mms_dataset_experiment.py --duration_label 14h --debug

Usage (torchrun alternative):
    torchrun --nproc_per_node 2 scripts/train_mms_dataset_experiment.py --duration_label 14h

Notes
-----
- Run with PYTHONPATH=<project_root> if installed in editable mode is unavailable:
    PYTHONPATH=/home/user/chichewa-asr accelerate launch ...
- Build the processor once from the full dataset; all duration runs reuse it.
  The --processor_dir arg points to that saved processor directory.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import DatasetDict

# ==========================================
# PATH SETUP
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.train.train_whisper import load_config
from src.train.mms_duration_experiment import (
    build_processor,
    load_model_and_processor,
    prepare_train_dataset,
    prepare_test_dataset,
    run_training,
    run_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MMS dataset duration sweep experiment")
    parser.add_argument("--duration_label", type=str, required=True,
                        help="Label for this experiment run, e.g. '14h', '7h'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (defaults to mms_hparams_baseline.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: uses debug config and small data subset")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Number of processes for dataset preprocessing (default: 1)")
    parser.add_argument("--processor_dir", type=str, default=None,
                        help="Path to saved Wav2Vec2Processor (defaults to models/mms_processor). "
                             "Built from the full training manifest if it does not exist.")
    parser.add_argument("--train_cache", type=str, default=None,
                        help="Path to cached preprocessed training dataset")
    parser.add_argument("--test_cache", type=str, default=None,
                        help="Path to cached preprocessed test dataset")
    parser.add_argument("--no_push", action="store_true",
                        help="Skip pushing model to Hub after training")
    return parser.parse_args()


def main():
    args = parse_args()

    # ==========================================
    # PATHS
    # ==========================================
    DIR_DATA              = PROJECT_ROOT / "data"
    DIR_DEV               = DIR_DATA / "dev"
    DIR_TEST              = DIR_DATA / "test"
    FILE_MANIFEST_DEV     = DIR_DEV / "metadata.csv"
    FILE_MANIFEST_TEST    = DIR_TEST / "metadata.csv"
    DIR_OUTPUTS           = PROJECT_ROOT / "outputs"
    DIR_RESULTS           = DIR_OUTPUTS / "datasets_experiments_mms"
    DIR_MODELS            = PROJECT_ROOT / "models"
    DIR_MODEL_CHECKPOINTS = DIR_MODELS / "checkpoints"
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    DIR_MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    processor_dir = Path(args.processor_dir) if args.processor_dir else DIR_MODELS / "mms_processor"

    # ==========================================
    # CONFIG
    # ==========================================
    if args.debug:
        FILE_CONFIG = PROJECT_ROOT / "configs" / "mms_hparams_debug.yaml"
        print("DEBUG MODE: ON — using debug config with minimal steps")
    elif args.config:
        FILE_CONFIG = Path(args.config)
    else:
        FILE_CONFIG = PROJECT_ROOT / "configs" / "mms_hparams_baseline.yaml"

    base_config = load_config(FILE_CONFIG)
    print(f"Config loaded: {FILE_CONFIG}")

    MODEL_ID          = base_config["model"]["model_name_or_path"]
    MODEL_NAME        = MODEL_ID.split("/")[-1]
    BASE_HUB_MODEL_ID = f"dmatekenya/{MODEL_NAME}-chichewa"

    # ==========================================
    # LOGIN
    # ==========================================
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    # ==========================================
    # BUILD / LOAD PROCESSOR (once, reused across all runs)
    # ==========================================
    print(f"\nStep 1/5 — Processor")
    processor = build_processor(
        manifest_path=FILE_MANIFEST_DEV,
        audio_dir=DIR_DEV,
        model_id=MODEL_ID,
        target_lang=base_config["model"]["target_lang"],
        save_dir=processor_dir,
    )
    print(f"  Processor vocab size: {len(processor.tokenizer)}")

    # ==========================================
    # PREPARE HELD-OUT TEST SET (once)
    # ==========================================
    print(f"\nStep 2/5 — Test dataset")
    test_cache = Path(args.test_cache) if args.test_cache else None
    dataset_test = prepare_test_dataset(
        FILE_MANIFEST_TEST,
        audio_dir=DIR_TEST,
        processor=processor,
        audio_fname_col="audio_filename",
        duration_col="duration_seconds",
        cache_dir=test_cache,
    )
    print(f"  Held-out test set ready: {len(dataset_test):,} utterances")

    # ==========================================
    # LOAD MODEL
    # ==========================================
    print(f"\nStep 3/5 — Model")
    model, processor = load_model_and_processor(base_config, processor_dir)

    # ==========================================
    # PREPARE TRAINING DATASET
    # ==========================================
    print(f"\nStep 4/5 — Train dataset")
    train_cache = Path(args.train_cache) if args.train_cache else None
    dataset_train = prepare_train_dataset(
        FILE_MANIFEST_DEV,
        DIR_DEV,
        processor,
        cache_dir=train_cache,
        num_proc=args.num_proc,
    )

    if args.debug:
        print("  DEBUG MODE: using small data subset")
        dataset_train = DatasetDict({
            "train":      dataset_train["train"].select(range(100)),
            "validation": dataset_train["validation"].select(range(20)),
        })

    # ==========================================
    # TRAIN
    # ==========================================
    print(f"\nStep 5/5 — Training")
    hub_model_id = f"{BASE_HUB_MODEL_ID}-{args.duration_label}"
    output_dir   = DIR_MODEL_CHECKPOINTS / f"{MODEL_NAME}-chichewa-{args.duration_label}"

    print(f"\n{'='*60}\n  EXPERIMENT: {args.duration_label}\n{'='*60}")
    train_start = time.time()
    trainer = run_training(model, processor, dataset_train, base_config, hub_model_id, output_dir)
    train_minutes = (time.time() - train_start) / 60
    print(f"  Training complete in {train_minutes:.1f} minutes")

    if not args.no_push and base_config.get("hub", {}).get("push_to_hub", False):
        print(f"  Pushing to Hub: {hub_model_id}")
        trainer.push_to_hub()

    # ==========================================
    # EVALUATE
    # ==========================================
    df_results = run_evaluation(
        model, processor, dataset_test,
        args.duration_label,
        DIR_RESULTS,
        model_id=hub_model_id,
        debug=args.debug,
    )
    print(df_results.head())


if __name__ == "__main__":
    main()
