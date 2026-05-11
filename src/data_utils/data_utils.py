from pathlib import Path
import pandas as pd
import numpy as np

from datasets import Dataset,DatasetDict, Audio, Features, Value



def _to_hf_dataset(df: pd.DataFrame, duration_col: str, sampling_rate: int) -> Dataset:
    """Build a HF Dataset from a DataFrame using from_dict to avoid
    PyArrow large_string → Audio struct cast errors."""
    data = {
        "audio":      df["audio"].tolist(),       # plain Python str, not large_string
        "sentence":   df["sentence"].tolist(),
        duration_col: df[duration_col].tolist(),
    }

    # Preserve original filename for traceability (e.g. hold-out evaluation CSVs).
    # Only included when the column exists — test sets carry it; train/val splits do not.
    if "audio_fname" in df.columns:
        data["audio_fname"] = df["audio_fname"].tolist()

    ds = Dataset.from_dict(data)
    return ds.cast_column("audio", Audio(sampling_rate=sampling_rate))


def duration_split_train_val(df, duration_col, valid_frac, seed):
    """
    Shuffles utterances, greedily fills the validation bucket up to its
    duration target. Everything remaining goes to train.
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    val_cap = df[duration_col].sum() * valid_frac
    val_idx, train_idx = [], []
    val_dur = 0.0

    for i, row in df.iterrows():
        d = row[duration_col]
        if val_dur < val_cap:
            val_idx.append(i);   val_dur += d
        else:
            train_idx.append(i)

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
    )

def load_audio_data(
    manifest_path: str,
    audio_dir: str,
    valid_frac: float = 0.10,
    duration_col: str = "duration",
    audio_fname_col: str = "audio_fname",
    transcript_col: str = "transcript",
    sampling_rate: int = 16_000,
    seed: int = 42,
    split_data: bool = True,
) -> "DatasetDict | Dataset":
    """
    Load an ASR dataset from a CSV manifest and a folder of audio files using Hugging Face Datasets.

    Args:
        manifest_path:    Path to the CSV manifest file.
        audio_dir:        Root directory containing the audio files.
        valid_frac:       Fraction of total duration to reserve for validation (only used if split_data=True).
        duration_col:     Manifest column containing clip duration in seconds.
        audio_fname_col:  Manifest column containing audio file names/paths.
        transcript_col:   Manifest column containing transcriptions.
        sampling_rate:    Target sampling rate for all audio (default: 16 kHz).
        seed:             Random seed for reproducible shuffling.
        split_data:       If True, split into train/validation sets. If False, return all data as a single Dataset (for hold-out/test sets).

    Returns:
        DatasetDict with 'train' and 'validation' splits if split_data=True, otherwise a single Dataset with all data.
    """
    # ── 1. Load & validate manifest ───────────────────────────────────────────
    df = pd.read_csv(manifest_path)

    missing = [c for c in [duration_col, audio_fname_col, transcript_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in manifest: {missing}")

    df["audio_fname"] = df[audio_fname_col]
    df["audio"] = df[audio_fname_col].apply(
        lambda f: str(Path(audio_dir) / f) if not Path(f).is_absolute() else f
    )
    df = df.rename(columns={transcript_col: "sentence"})
    df = df[["audio", "audio_fname", "sentence", duration_col]].reset_index(drop=True)

    total_duration = df[duration_col].sum()
    print(f"Total duration : {total_duration / 3600:.2f} hrs  ({len(df):,} utterances)")

    if split_data:
        # ── 2. Split ──────────────────────────────────────────────────────────────
        df_train, df_valid = duration_split_train_val(df, duration_col, valid_frac, seed)

        for name, split in [("train", df_train), ("validation", df_valid)]:
            hrs = split[duration_col].sum() / 3600
            pct = 100 * split[duration_col].sum() / total_duration
            print(f"  {name:12s}: {len(split):>5,} utterances  |  {hrs:.2f} hrs  ({pct:.1f}%)")

        # ── 3. Build DatasetDict ──────────────────────────────────────────────────
        dataset = DatasetDict({
            "train":      _to_hf_dataset(df_train, duration_col, sampling_rate),
            "validation": _to_hf_dataset(df_valid, duration_col, sampling_rate),
        })
        return dataset
    else:
        # No split, return all data as a single Dataset (for hold-out/test set)
        print(f"  all_data   : {len(df):>5,} utterances  |  {total_duration / 3600:.2f} hrs (100.0%)")
        return _to_hf_dataset(df, duration_col, sampling_rate)
    
def sample_by_duration(
    df: pd.DataFrame,
    duration_column: str = "duration",
    target_hours: float = 10,
    duration_bins: list[float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample audio records up to a target number of hours while preserving the
    distribution of audio durations.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata dataframe containing audio durations.
    duration_column : str
        Name of the column containing duration in seconds.
    target_hours : float
        Target number of hours to sample.
    duration_bins : list[float] | None
        Duration bin edges in seconds. If None, default ASR-friendly bins are used.
    random_state : int
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Duration-stratified sampled dataframe.
    """

    if duration_bins is None:
        duration_bins = [0, 5, 10, 15, 20, 30, 45]

    df = df.copy()

    target_seconds = target_hours * 3600

    df["duration_bin"] = pd.cut(
        df[duration_column],
        bins=duration_bins,
        include_lowest=True,
    )

    bin_summary = (
        df.groupby("duration_bin", observed=True)
        .agg(total_seconds=(duration_column, "sum"))
        .reset_index()
    )

    bin_summary["duration_fraction"] = (
        bin_summary["total_seconds"] / bin_summary["total_seconds"].sum()
    )

    sampled_parts = []

    for _, row in bin_summary.iterrows():
        bin_label = row["duration_bin"]
        target_bin_seconds = row["duration_fraction"] * target_seconds

        subset = df[df["duration_bin"] == bin_label].copy()

        subset = subset.sample(
            frac=1,
            random_state=random_state,
        )

        subset["cum_duration"] = subset[duration_column].cumsum()

        sampled_subset = subset[
            subset["cum_duration"] <= target_bin_seconds
        ]

        sampled_parts.append(sampled_subset)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    sampled_df = sampled_df.drop(
        columns=["cum_duration"],
        errors="ignore",
    )

    return sampled_df