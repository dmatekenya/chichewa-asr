"""
Multi-GPU duration sweep script for MMS adapter fine-tuning.

Mirrors notebooks/mms/mms_finetune_duration_sweep.ipynb but runs from the
command line with accelerate/torchrun for proper multi-GPU support.

Usage (single GPU):
    PYTHONPATH=. python scripts/train_mms_dataset_experiment.py

Usage (multi-GPU, recommended):
    PYTHONPATH=. accelerate launch --num_processes 2 scripts/train_mms_dataset_experiment.py

Usage (debug — CPU, 20 steps):
    PYTHONPATH=. python scripts/train_mms_dataset_experiment.py --debug

Notes
-----
- The processor is built once from the full manifest and reused across all runs.
- Early stopping is enabled by default (patience=3 eval periods).
- To run a single duration instead of the full sweep, use --duration_label.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

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
    parser = argparse.ArgumentParser(description="MMS duration sweep experiment")
    parser.add_argument("--duration_label", type=str, default=None,
                        help="Run a single duration (e.g. '1h'). Omit to run full sweep.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (defaults to mms_hparams_baseline.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: uses debug config and small data subset")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Processes for dataset preprocessing (default: 1)")
    parser.add_argument("--processor_dir", type=str, default=None,
                        help="Path to saved processor. Defaults to models/artifacts/<model>/processor")
    parser.add_argument("--train_cache", type=str, default=None,
                        help="Path to cached training dataset")
    parser.add_argument("--test_cache", type=str, default=None,
                        help="Path to cached test dataset")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience in eval periods (0 = disabled)")
    parser.add_argument("--no_push", action="store_true",
                        help="Skip pushing model to Hub after training")
    return parser.parse_args()


def main():
    args = parse_args()

    # ==========================================
    # PATHS
    # ==========================================
    DIR_DATA                = PROJECT_ROOT / "data"
    DIR_DEV                 = DIR_DATA / "dev"
    DIR_TEST                = DIR_DATA / "test"
    FILE_MANIFEST_DEV       = DIR_DEV / "metadata.csv"
    FILE_MANIFEST_TEST      = DIR_TEST / "metadata.csv"
    DIR_DEV_NESTED_DURATION = DIR_DATA / "dev_nested_duration"
    DIR_OUTPUTS             = PROJECT_ROOT / "outputs"
    DIR_MODELS              = PROJECT_ROOT / "models"
    DIR_MODEL_CHECKPOINTS   = DIR_MODELS / "checkpoints"
    DIR_RESULTS             = DIR_OUTPUTS / "duration-exp-mms-1b-all"
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    DIR_MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # CONFIG
    # ==========================================
    if args.debug:
        FILE_CONFIG = PROJECT_ROOT / "configs" / "mms_hparams_debug.yaml"
        print("DEBUG MODE: ON — using debug config")
    elif args.config:
        FILE_CONFIG = Path(args.config)
    else:
        FILE_CONFIG = PROJECT_ROOT / "configs" / "mms_hparams_baseline.yaml"

    config = load_config(FILE_CONFIG)
    print(f"Config: {FILE_CONFIG.name}")

    MODEL_ID   = config["model"]["model_name_or_path"]
    MODEL_NAME = MODEL_ID.split("/")[-1]
    HUB_MODEL_ID = f"dmatekenya/{MODEL_NAME}-chichewa"

    processor_dir = (
        Path(args.processor_dir)
        if args.processor_dir
        else DIR_MODELS / "artifacts" / MODEL_NAME / "processor"
    )

    # ==========================================
    # LOGIN
    # ==========================================
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    # ==========================================
    # DURATION DATASETS
    # ==========================================
    if args.duration_label:
        manifest = DIR_DEV_NESTED_DURATION / f"train_{args.duration_label}.csv"
        duration_datasets = {args.duration_label: manifest}
    else:
        duration_datasets = {
            f"{h}h": DIR_DEV_NESTED_DURATION / f"train_{h}h.csv"
            for h in [1, 2, 5, 10, 14]
        }
        duration_datasets["14h"] = FILE_MANIFEST_DEV

    missing = [str(p) for p in duration_datasets.values() if not p.exists()]
    if missing:
        print("WARNING — manifest files not found, will skip:")
        for m in missing:
            print(f"  {m}")

    # ==========================================
    # BUILD / LOAD PROCESSOR  (once)
    # ==========================================
    print(f"\nStep 1 — Processor")
    processor = build_processor(
        manifest_path=FILE_MANIFEST_DEV,
        audio_dir=DIR_DEV,
        model_id=MODEL_ID,
        target_lang=config["model"]["target_lang"],
        save_dir=processor_dir,
    )
    print(f"  Vocab size: {len(processor.tokenizer)}")

    # ==========================================
    # PREPARE HELD-OUT TEST SET  (once)
    # ==========================================
    print(f"\nStep 2 — Test dataset")
    test_cache = Path(args.test_cache) if args.test_cache else None
    dataset_test = prepare_test_dataset(
        FILE_MANIFEST_TEST,
        audio_dir=DIR_TEST,
        processor=processor,
        audio_fname_col="audio_filename",
        duration_col="duration_seconds",
        cache_dir=test_cache,
    )
    print(f"  Test set: {len(dataset_test):,} utterances")

    # ==========================================
    # SWEEP
    # ==========================================
    summary = []

    for duration_label, manifest_path in duration_datasets.items():
        if not manifest_path.exists():
            print(f"\nSkipping {duration_label} — manifest not found.")
            continue

        print(f"\n{'='*60}\n  EXPERIMENT: {duration_label}\n{'='*60}")

        hub_model_id = f"{HUB_MODEL_ID}-{duration_label}"
        output_dir   = DIR_MODEL_CHECKPOINTS / f"{MODEL_NAME}-chichewa-{duration_label}"

        # Load fresh model for each run
        model, processor = load_model_and_processor(config, processor_dir)

        train_cache = Path(args.train_cache) if args.train_cache else None
        dataset_train = prepare_train_dataset(
            manifest_path, DIR_DEV, processor,
            cache_dir=train_cache,
            num_proc=args.num_proc,
        )

        train_start = time.time()
        trainer = run_training(
            model, processor, dataset_train, config, hub_model_id, output_dir,
            debug=args.debug,
            patience=args.patience,
        )
        train_minutes = (time.time() - train_start) / 60
        print(f"  Training complete in {train_minutes:.1f} min")

        push_to_hub = (
            not args.no_push
            and not args.debug
            and config.get("hub", {}).get("push_to_hub", False)
        )
        if push_to_hub:
            print(f"  Pushing to Hub: {hub_model_id}")
            trainer.push_to_hub()

        df_results = run_evaluation(
            model, processor, dataset_test,
            duration_label, DIR_RESULTS,
            model_id=hub_model_id,
            debug=args.debug,
        )
        summary.append({
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration":      duration_label,
            "wer":           df_results["wer_avg"].iloc[0],
            "cer":           df_results["cer_avg"].iloc[0],
            "hub_model_id":  hub_model_id,
            "train_minutes": round(train_minutes, 2),
        })

        pd.DataFrame(summary).to_csv(DIR_RESULTS / "duration_sweep_summary.csv", index=False)

    print("\nSweep complete.")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
