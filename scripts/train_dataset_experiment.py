"""
Multi-GPU training script for Whisper dataset experiments.

Equivalent to notebooks/whisper/whisper_finetune_datasets_experiments.ipynb
but runnable with accelerate/torchrun for multi-GPU support.

Usage (single GPU):
    python scripts/train_dataset_experiment.py --duration_label 14h

Usage (multi-GPU):
    accelerate launch --num_processes 2 scripts/train_dataset_experiment.py --duration_label 14h

Usage (debug):
    python scripts/train_dataset_experiment.py --duration_label 14h --debug

Usage (torchrun alternative):
    torchrun --nproc_per_node 2 scripts/train_dataset_experiment.py --duration_label 14h
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
import torch
import wandb
from datasets import DatasetDict

# ==========================================
# PATH SETUP
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.train.train_whisper import load_config
from src.train.whisper_duration_experiment import (
    load_model_and_processor,
    prepare_train_dataset,
    prepare_test_dataset,
    run_training,
    run_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper dataset fine-tuning experiment")
    parser.add_argument("--duration_label", type=str, required=True,
                        help="Label for this experiment run, e.g. '14h'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (defaults to whisper_hparams_baseline.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: uses debug config and small data subset")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Number of processes for dataset preprocessing (default: 1)")
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
    DIR_RESULTS           = DIR_OUTPUTS / "datasets_experiments"
    DIR_MODELS            = PROJECT_ROOT / "models"
    DIR_WANDB_LOGS        = DIR_MODELS / "wandb"
    DIR_MODEL_CHECKPOINTS = DIR_MODELS / "checkpoints"
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    DIR_MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # CONFIG
    # ==========================================
    if args.debug:
        FILE_CONFIG = PROJECT_ROOT / "configs" / "whisper_hparams_debug.yaml"
        print("DEBUG MODE: ON — using debug config with minimal steps")
    elif args.config:
        FILE_CONFIG = Path(args.config)
    else:
        FILE_CONFIG = PROJECT_ROOT / "configs" / "whisper_hparams_baseline.yaml"

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
    os.environ["WANDB_DIR"] = str(DIR_WANDB_LOGS)
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # ==========================================
    # PREPARE HELD-OUT TEST SET (once)
    # ==========================================
    test_cache = Path(args.test_cache) if args.test_cache else None
    dataset_test = prepare_test_dataset(
        FILE_MANIFEST_TEST,
        audio_dir=DIR_TEST,
        base_config=base_config,
        audio_fname_col="audio_filename",
        duration_col="duration_seconds",
        cache_dir=test_cache,
    )
    print(f"Held-out test set ready: {len(dataset_test):,} utterances")

    # ==========================================
    # EXPERIMENT
    # ==========================================
    hub_model_id = f"{BASE_HUB_MODEL_ID}-{args.duration_label}"
    output_dir   = DIR_MODEL_CHECKPOINTS / f"{MODEL_NAME}-chichewa-{args.duration_label}"

    model, processor = load_model_and_processor(base_config)

    train_cache = Path(args.train_cache) if args.train_cache else None
    dataset_train = prepare_train_dataset(
        FILE_MANIFEST_DEV,
        DIR_DEV,
        processor,
        cache_dir=train_cache,
        num_proc=args.num_proc,
    )

    if args.debug:
        print("DEBUG MODE: using small data subset")
        dataset_train = DatasetDict({
            "train":      dataset_train["train"].select(range(100)),
            "validation": dataset_train["validation"].select(range(20)),
        })

    print(f"\n{'='*60}\n  EXPERIMENT: {args.duration_label}\n{'='*60}")
    train_start = time.time()
    trainer = run_training(model, processor, dataset_train, base_config, hub_model_id, output_dir)
    train_minutes = (time.time() - train_start) / 60
    print(f"  Training complete in {train_minutes:.1f} minutes")

    if not args.no_push:
        print(f"  Pushing to Hub: {hub_model_id}")
        trainer.push_to_hub()

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
