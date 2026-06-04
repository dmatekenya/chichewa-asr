import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import librosa
from prompt_toolkit import HTML
from IPython.display import display, HTML
from tqdm import tqdm

from datasets import Dataset, DatasetDict, Audio
from data_utils.audio_utils import compute_audio_attributes
from data_utils.text_utils import (
    normalize_transcript,
    count_words,
    count_chars,
    words_per_second as _wps,
    chars_per_second as _cps,
)



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

def compute_total_audio_duration(
    audio_dir: str | Path,
    extensions: tuple = (".wav", ".mp3", ".flac", ".m4a"),
) -> pd.DataFrame:
    """
    Load audio files in a folder and compute durations.

    Parameters
    ----------
    audio_dir : str | Path
        Directory containing audio files.
    extensions : tuple
        Audio file extensions to include.

    Returns
    -------
    pd.DataFrame
        DataFrame containing file paths and durations.
    """

    audio_dir = Path(audio_dir)

    audio_files = []

    for ext in extensions:
        audio_files.extend(audio_dir.rglob(f"*{ext}"))

    records = []

    for audio_path in tqdm(audio_files):

        try:
            duration = librosa.get_duration(
                path=str(audio_path)
            )

            records.append(
                {
                    "audio_path": str(audio_path),
                    "duration": duration,
                }
            )

        except Exception as e:

            print(f"Failed: {audio_path}")
            print(e)

    df = pd.DataFrame(records)

    total_hours = df["duration"].sum() / 3600

    print(f"\nTotal files: {len(df):,}")
    print(f"Total duration: {total_hours:.2f} hours")

    return total_hours

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

def remove_special_characters(batch, transcript_column="sentence", 
                              charecters_to_remove_regex=r"[^a-zA-Z0-9\s]"):
    batch[transcript_column] = re.sub(charecters_to_remove_regex, '', batch[transcript_column]).lower()
    return batch

def extract_all_chars(batch, transcript_column="sentence"):
    all_text = " ".join(batch[transcript_column])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def compute_quality_metrics_for_manifest(
    manifest_csv: str | Path,
    audio_dir: str | Path,
    audio_col: str = "audio_filename",
    transcript_col: Optional[str] = None,
    use_vad: bool = False,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compute audio quality metrics for every file listed in a manifest CSV.

    Calls :func:`~data_utils.audio_utils.compute_audio_attributes` on each
    file and merges the results back into the original manifest as additional
    columns. Optionally computes transcript-level speaking-rate metrics and
    Silero VAD metrics.

    Parameters
    ----------
    manifest_csv : str or Path
        Path to the CSV manifest file.
    audio_dir : str or Path
        Directory containing the audio files.
    audio_col : str, optional
        Column in the manifest containing audio filenames,
        by default ``'audio_filename'``.
    transcript_col : str or None, optional
        Column in the manifest containing transcript text. When provided,
        ``num_words``, ``num_chars``, ``words_per_second``, and
        ``chars_per_second`` are computed from the normalised transcript.
    use_vad : bool, optional
        If True, run Silero VAD on every clip to compute
        ``speech_duration_sec``, ``speech_ratio``, ``silence_ratio``,
        ``num_speech_segments``, ``mean_segment_duration``, and
        ``snr_proxy_db``. Slower than the default heuristic path.
        Falls back gracefully if silero-vad is not installed.
    output_csv : str or Path, optional
        If provided, saves the enriched DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        Original manifest with quality metric columns appended.
    """
    from data_utils.audio_utils import load_audio_mono

    audio_dir = Path(audio_dir)
    df = pd.read_csv(manifest_csv)

    _audio_nan: Dict[str, Any] = {k: float("nan") for k in [
        "sample_rate", "num_samples", "duration_sec",
        "rms", "peak_abs", "rms_dbfs", "peak_dbfs", "crest_factor_db",
        "silence_ratio", "vad_speech_ratio",
        "snr_db_est", "clipping_ratio",
        "spectral_flatness", "zero_crossing_rate", "effective_bandwidth_hz",
        "rt60_est_sec",
        "num_channels", "channel_strategy", "channel_rms_diff", "dominant_channel",
    ]}
    _vad_nan: Dict[str, Any] = {k: float("nan") for k in [
        "speech_duration_sec", "speech_ratio", "silence_ratio",
        "num_speech_segments", "mean_segment_duration", "snr_proxy_db",
    ]}
    _tx_nan: Dict[str, Any] = {
        "num_words": float("nan"), "num_chars": float("nan"),
        "words_per_second": float("nan"), "chars_per_second": float("nan"),
    }

    audio_records: List[Dict[str, Any]] = []
    vad_records:   List[Dict[str, Any]] = []
    tx_records:    List[Dict[str, Any]] = []

    has_transcripts = transcript_col is not None and transcript_col in df.columns

    for i, fname in enumerate(tqdm(df[audio_col], desc="Computing audio quality metrics")):
        audio_path = audio_dir / fname

        # ── audio attributes ──────────────────────────────────────────────
        try:
            attrs = compute_audio_attributes(str(audio_path))
        except Exception as e:
            attrs = dict(_audio_nan)
            print(f"  Warning: could not process {fname}: {e}")
        audio_records.append(attrs)

        # ── VAD metrics (optional) ────────────────────────────────────────
        if use_vad:
            try:
                from data_utils.vad_utils import compute_vad_metrics
                waveform, sr, _ = load_audio_mono(str(audio_path))
                vad_attrs = compute_vad_metrics(waveform, sr)
            except Exception as e:
                vad_attrs = dict(_vad_nan)
                print(f"  Warning: VAD failed for {fname}: {e}")
            vad_records.append(vad_attrs)

        # ── transcript metrics (optional) ─────────────────────────────────
        if has_transcripts:
            try:
                raw_text = str(df[transcript_col].iloc[i]) if pd.notna(df[transcript_col].iloc[i]) else ""
                norm_text = normalize_transcript(raw_text)
                dur = attrs.get("duration_sec", float("nan"))
                tx_records.append({
                    "num_words":        count_words(norm_text),
                    "num_chars":        count_chars(norm_text),
                    "words_per_second": _wps(norm_text, dur),
                    "chars_per_second": _cps(norm_text, dur),
                })
            except Exception as e:
                tx_records.append(dict(_tx_nan))
                print(f"  Warning: transcript metrics failed for row {i}: {e}")

    # ── assemble result ───────────────────────────────────────────────────
    parts = [df.reset_index(drop=True), pd.DataFrame(audio_records)]
    if use_vad and vad_records:
        parts.append(pd.DataFrame(vad_records))
    if has_transcripts and tx_records:
        parts.append(pd.DataFrame(tx_records))

    result = pd.concat(parts, axis=1)

    if output_csv is not None:
        result.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")

    return result


def compute_bad_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quality flag columns and an additive bad-quality score to a
    profiled manifest DataFrame.

    Expects the DataFrame to have been produced by
    :func:`compute_quality_metrics_for_manifest`. Works with or without
    VAD and transcript columns: missing columns are silently skipped.

    Each of the seven criteria below contributes 1 to
    ``bad_quality_score``:

    - ``flag_duration``    — ``duration_sec`` outside [0.7, 25] seconds
    - ``flag_speech_ratio``— best available speech ratio < 0.35
    - ``flag_snr``         — best available SNR < 5 dB
    - ``flag_rms``         — ``rms_dbfs`` < −35 dBFS
    - ``flag_clipping``    — ``clipping_ratio`` > 0.005
    - ``flag_wps``         — ``words_per_second`` outside [0.5, 4.5]
    - ``flag_fragmented``  — ``num_speech_segments`` above clip-length
                             expectation *and* ``mean_segment_duration`` < 0.7 s

    Parameters
    ----------
    df : pd.DataFrame
        Profiled manifest (output of ``compute_quality_metrics_for_manifest``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with flag columns and ``bad_quality_score`` added
        in-place (the original is not copied).
    """
    df = df.copy()

    # duration
    if "duration_sec" in df.columns:
        df["flag_duration"] = ~df["duration_sec"].between(0.7, 25.0)
    else:
        df["flag_duration"] = False

    # speech ratio — prefer VAD-based column, fall back to heuristic
    if "speech_ratio" in df.columns:
        df["flag_speech_ratio"] = df["speech_ratio"] < 0.35
    elif "vad_speech_ratio" in df.columns:
        df["flag_speech_ratio"] = df["vad_speech_ratio"] < 0.35
    else:
        df["flag_speech_ratio"] = False

    # SNR — prefer VAD-aware proxy, fall back to heuristic
    if "snr_proxy_db" in df.columns:
        df["flag_snr"] = df["snr_proxy_db"] < 5.0
    elif "snr_db_est" in df.columns:
        df["flag_snr"] = df["snr_db_est"] < 5.0
    else:
        df["flag_snr"] = False

    # RMS dBFS
    if "rms_dbfs" in df.columns:
        df["flag_rms"] = df["rms_dbfs"] < -35.0
    else:
        df["flag_rms"] = False

    # clipping
    if "clipping_ratio" in df.columns:
        df["flag_clipping"] = df["clipping_ratio"] > 0.005
    else:
        df["flag_clipping"] = False

    # words per second (skipped if column absent)
    if "words_per_second" in df.columns:
        df["flag_wps"] = ~df["words_per_second"].between(0.5, 4.5)
    else:
        df["flag_wps"] = False

    # fragmentation: many short VAD spans
    if "num_speech_segments" in df.columns and "mean_segment_duration" in df.columns:
        # expect roughly 1 segment per 3 seconds; flag when much higher
        expected_segments = (df["duration_sec"] / 3.0).clip(lower=1.0)
        df["flag_fragmented"] = (
            (df["num_speech_segments"] > expected_segments * 2)
            & (df["mean_segment_duration"] < 0.7)
        )
    else:
        df["flag_fragmented"] = False

    flag_cols = [
        "flag_duration", "flag_speech_ratio", "flag_snr",
        "flag_rms", "flag_clipping", "flag_wps", "flag_fragmented",
    ]
    df["bad_quality_score"] = df[flag_cols].sum(axis=1).astype(int)

    return df


def create_data_variants(
    df: pd.DataFrame,
    transcript_col: str = "transcript",
    duration_col: str = "duration_sec",
) -> Dict[str, pd.DataFrame]:
    """
    Create the four data variants recommended for Whisper ablation experiments.

    Requires ``bad_quality_score`` to be present; call
    :func:`compute_bad_quality_score` first.

    Parameters
    ----------
    df : pd.DataFrame
        Profiled and scored manifest DataFrame.
    transcript_col : str, optional
        Column containing raw transcript text, by default ``'transcript'``.
    duration_col : str, optional
        Column containing clip duration in seconds, by default
        ``'duration_sec'``.

    Returns
    -------
    dict
        Keys ``'A'``, ``'B'``, ``'C'``, ``'D'``, each a filtered
        pd.DataFrame. Also prints a retention summary table.

    Variants
    --------
    A  Unfiltered baseline.
    B  A + ``normalize_transcript()`` applied to the transcript column.
    C  B + strong-reject filter (extreme duration / speech / RMS / clipping).
    D  C + drop worst 10–15 % by ``bad_quality_score >= 2``.
    """
    if "bad_quality_score" not in df.columns:
        raise ValueError(
            "DataFrame must have a 'bad_quality_score' column. "
            "Call compute_bad_quality_score(df) first."
        )

    has_tx = transcript_col in df.columns

    # ── Variant A — unfiltered ────────────────────────────────────────────
    variant_a = df.copy()

    # ── Variant B — normalised transcripts ───────────────────────────────
    variant_b = variant_a.copy()
    if has_tx:
        variant_b[transcript_col] = variant_b[transcript_col].fillna("").apply(
            normalize_transcript
        )

    # ── Variant C — strong-reject filter ─────────────────────────────────
    strong_reject = pd.Series(False, index=variant_b.index)

    if "duration_sec" in variant_b.columns:
        strong_reject |= ~variant_b["duration_sec"].between(0.5, 30.0)

    # prefer VAD speech_ratio, fall back to heuristic
    sr_col = "speech_ratio" if "speech_ratio" in variant_b.columns else "vad_speech_ratio"
    if sr_col in variant_b.columns:
        strong_reject |= variant_b[sr_col] < 0.20

    if "rms_dbfs" in variant_b.columns:
        strong_reject |= variant_b["rms_dbfs"] < -40.0

    if "clipping_ratio" in variant_b.columns:
        strong_reject |= variant_b["clipping_ratio"] > 0.02

    variant_c = variant_b[~strong_reject].reset_index(drop=True)

    # ── Variant D — moderate cleanup (worst ~10-15 % by score) ───────────
    variant_d = variant_c[variant_c["bad_quality_score"] < 2].reset_index(drop=True)

    variants: Dict[str, pd.DataFrame] = {
        "A": variant_a,
        "B": variant_b,
        "C": variant_c,
        "D": variant_d,
    }

    # ── Print retention summary ───────────────────────────────────────────
    print(f"\n{'Variant':<10} {'Clips':>8} {'Hours':>8}  Description")
    print("-" * 55)
    descriptions = {
        "A": "Unfiltered baseline",
        "B": "A + normalised transcripts",
        "C": "B + strong-reject filter",
        "D": "C + moderate quality filter (score < 2)",
    }
    for key, vdf in variants.items():
        clips = len(vdf)
        if duration_col in vdf.columns:
            hours = vdf[duration_col].sum() / 3600
        elif "duration" in vdf.columns:
            hours = vdf["duration"].sum() / 3600
        else:
            hours = float("nan")
        print(f"  {key:<8} {clips:>8,} {hours:>8.2f}h  {descriptions[key]}")
    print()

    return variants
