"""
Utility script for generating standardized acoustic attributes for ASR datasets.

This script computes common audio-level attributes such as duration, RMS,
peak amplitude, silence ratio, and a heuristic SNR estimate. It is designed
to be used on fixed train/dev/test metadata files so that attribute generation
is standardized across experiments.

Example
-------
python generate_audio_attributes.py \
    --input_csv data/test_metadata.csv \
    --audio_col audio_path \
    --output_csv data/test_metadata_with_attrs.csv

Expected input CSV
------------------
A CSV file with at least one column containing audio file paths.

Notes
-----
- Audio is converted to mono before analysis.
- Silence ratio is computed using frame-level RMS values.
- SNR is only a heuristic estimate when no clean reference signal is available.
"""

from __future__ import annotations

# Standard library imports
import argparse
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import librosa
import numpy as np
import pandas as pd
import requests
import soundfile as sf
from tqdm import tqdm

# Optional/ASR-related imports
import torch
from jiwer import cer, wer
from openai import OpenAI
from transformers import pipeline
import torch
from jiwer import cer, wer
from openai import OpenAI
from tqdm import tqdm
from transformers import pipeline

def get_total_audio_duration(folder: str | Path, exts=None) -> float:

    """
    Calculate the total duration of all audio files in a folder.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing audio files.
    exts : list or None
        List of file extensions to include (e.g., ['.wav', '.mp3']). If None, defaults to common audio types.
    units : str, optional
        Units for return value: 'seconds', 'minutes', or 'hours'. Default is 'seconds'.

    Returns
    -------
    float
        Total duration in the requested units.
    """
    if exts is None:
        exts = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    folder = Path(folder)
    total_duration = 0.0
    for ext in exts:
        for audio_file in folder.glob(f'*{ext}'):
            try:
                duration = librosa.get_duration(filename=str(audio_file))
                total_duration += duration
            except Exception as e:
                print(f"Could not process {audio_file}: {e}")
    return total_duration


def get_total_audio_duration_with_units(folder: str | Path, exts=None, units: str = 'seconds') -> float:
    """
    Calculate the total duration of all audio files in a folder, with selectable units.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing audio files.
    exts : list or None
        List of file extensions to include (e.g., ['.wav', '.mp3']). If None, defaults to common audio types.
    units : str, optional
        Units for return value: 'seconds', 'minutes', or 'hours'. Default is 'seconds'.

    Returns
    -------
    float
        Total duration in the requested units.
    """
    total_seconds = get_total_audio_duration(folder, exts)
    units = units.lower()
    if units == 'seconds':
        return total_seconds
    elif units == 'minutes':
        return total_seconds / 60
    elif units == 'hours':
        return total_seconds / 3600
    else:
        raise ValueError("units must be one of 'seconds', 'minutes', or 'hours'")


def compute_rms(waveform: np.ndarray) -> float:
    """
    Compute RMS energy of a waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Input waveform.

    Returns
    -------
    float
        RMS value.
    """
    if waveform.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(waveform))))


def load_audio_mono(audio_path: str) -> tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Load an audio file, record channel metadata, and convert to mono float32.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.

    Returns
    -------
    tuple[np.ndarray, int, Dict[str, Any]]
        A tuple containing:
        - mono waveform as a 1D numpy array of type float32
        - sample rate as an integer
        - dictionary with channel metadata
    """
    waveform, sample_rate = sf.read(audio_path, always_2d=False)

    channel_info: Dict[str, Any] = {
        "num_channels": 1,
        "channel_strategy": "mono_original",
        "channel_rms_diff": 0.0,
        "dominant_channel": "mono",
    }

    if waveform.ndim == 1:
        mono_waveform = waveform.astype(np.float32)
        return mono_waveform, sample_rate, channel_info

    if waveform.ndim == 2:
        num_channels = waveform.shape[1]
        channel_info["num_channels"] = int(num_channels)

        if num_channels == 2:
            left = waveform[:, 0].astype(np.float32)
            right = waveform[:, 1].astype(np.float32)

            left_rms = compute_rms(left)
            right_rms = compute_rms(right)

            channel_info["channel_rms_diff"] = float(abs(left_rms - right_rms))

            if np.isclose(left_rms, right_rms, atol=1e-6):
                channel_info["dominant_channel"] = "balanced"
            elif left_rms > right_rms:
                channel_info["dominant_channel"] = "left"
            else:
                channel_info["dominant_channel"] = "right"

            channel_info["channel_strategy"] = "averaged_to_mono"

        else:
            per_channel_rms = [compute_rms(waveform[:, i].astype(np.float32)) for i in range(num_channels)]
            channel_info["channel_rms_diff"] = float(np.max(per_channel_rms) - np.min(per_channel_rms))
            channel_info["dominant_channel"] = str(int(np.argmax(per_channel_rms)))
            channel_info["channel_strategy"] = "averaged_to_mono"

        mono_waveform = waveform.mean(axis=1).astype(np.float32)
        return mono_waveform, sample_rate, channel_info

    raise ValueError(f"Unsupported waveform shape: {waveform.shape}")


def safe_log10(x: float, eps: float = 1e-12) -> float:
    """
    Compute log10 safely by flooring the input at eps.

    Parameters
    ----------
    x : float
        Input value.
    eps : float, optional
        Minimum allowed value, by default 1e-12.

    Returns
    -------
    float
        The base-10 logarithm of max(x, eps).
    """
    return math.log10(max(x, eps))


def compute_frame_rms(
    waveform: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Compute frame-level RMS values.

    Parameters
    ----------
    waveform : np.ndarray
        Input mono waveform.
    frame_length : int, optional
        Frame size in samples, by default 400.
    hop_length : int, optional
        Hop size in samples, by default 160.

    Returns
    -------
    np.ndarray
        1D array of frame-level RMS values.
    """
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    return rms.squeeze(0)


def compute_silence_ratio(
    frame_rms: np.ndarray,
    silence_threshold_db: float = -40.0,
) -> float:
    """
    Compute the proportion of frames considered silent.

    Silence is defined relative to the maximum frame RMS in the file.
    Frames whose RMS is below `max_rms + silence_threshold_db` are treated
    as silent. For example, with -40 dB, frames more than 40 dB below the
    file's loudest frame are considered silent.

    Parameters
    ----------
    frame_rms : np.ndarray
        Frame-level RMS values.
    silence_threshold_db : float, optional
        Relative silence threshold in dB, by default -40.0.

    Returns
    -------
    float
        Fraction of silent frames in the range [0, 1].
    """
    if frame_rms.size == 0:
        return float("nan")

    max_rms = float(np.max(frame_rms))
    if max_rms <= 0:
        return 1.0

    threshold = max_rms * (10 ** (silence_threshold_db / 20.0))
    silent = frame_rms < threshold

    return float(np.mean(silent))


def estimate_snr_db(
    frame_rms: np.ndarray,
    noise_percentile: float = 20.0,
) -> float:
    """
    Estimate SNR in dB using a heuristic frame-energy approach.

    The noise floor is approximated as a low percentile of frame RMS values,
    while the signal level is approximated from the higher-energy frames.
    This is a rough estimate and should not be treated as true SNR.

    Parameters
    ----------
    frame_rms : np.ndarray
        Frame-level RMS values.
    noise_percentile : float, optional
        Percentile used to approximate the noise floor, by default 20.0.

    Returns
    -------
    float
        Estimated SNR in dB.
    """
    if frame_rms.size == 0:
        return float("nan")

    frame_power = np.square(frame_rms)

    noise_power = float(np.percentile(frame_power, noise_percentile))
    total_power = float(np.mean(frame_power))

    signal_power = max(total_power - noise_power, 1e-12)
    noise_power = max(noise_power, 1e-12)

    snr_db = 10.0 * safe_log10(signal_power / noise_power)
    return float(snr_db)


def compute_audio_attributes(
    audio_path: str,
    frame_length: int = 400,
    hop_length: int = 160,
    silence_threshold_db: float = -40.0,
    noise_percentile: float = 20.0,
) -> Dict[str, Any]:
    """
    Compute standardized acoustic attributes for one audio file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    frame_length : int, optional
        Frame size in samples, by default 400.
    hop_length : int, optional
        Hop size in samples, by default 160.
    silence_threshold_db : float, optional
        Relative silence threshold in dB, by default -40.0.
    noise_percentile : float, optional
        Percentile used to estimate the noise floor for SNR, by default 20.0.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing computed attributes.
    """
    waveform, sample_rate, channel_info = load_audio_mono(audio_path)

    duration_sec = len(waveform) / sample_rate if sample_rate > 0 else float("nan")
    rms = compute_rms(waveform)
    peak_abs = float(np.max(np.abs(waveform))) if waveform.size > 0 else float("nan")

    frame_rms = compute_frame_rms(
        waveform=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    silence_ratio = compute_silence_ratio(
        frame_rms=frame_rms,
        silence_threshold_db=silence_threshold_db,
    )

    snr_db_est = estimate_snr_db(
        frame_rms=frame_rms,
        noise_percentile=noise_percentile,
    )

    return {
        "sample_rate": sample_rate,
        "num_samples": int(len(waveform)),
        "duration_sec": float(duration_sec),
        "rms": rms,
        "peak_abs": peak_abs,
        "silence_ratio": silence_ratio,
        "snr_db_est": snr_db_est,
        "num_channels": channel_info["num_channels"],
        "channel_strategy": channel_info["channel_strategy"],
        "channel_rms_diff": channel_info["channel_rms_diff"],
        "dominant_channel": channel_info["dominant_channel"],
    }


def load_audio_16k_mono(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio, convert to mono, resample to 16 kHz.
    """
    waveform, sr = sf.read(audio_path, always_2d=False)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32)

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    return waveform, sr


def write_temp_wav_16k(audio_path: str) -> str:
    """
    Convert audio to temporary 16k mono WAV and return temp path.
    Useful for API calls.
    """
    waveform, sr = load_audio_16k_mono(audio_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, waveform, sr)
    return tmp_path


class HFASRBackend:
    """
    Generic Hugging Face ASR backend using pipeline().
    Works for Whisper, wav2vec2, MMS, and many ASR-capable checkpoints.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: Optional[str] = None,
        chunk_length_s: Optional[int] = None,
        batch_size: int = 1,
        language: Optional[str] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self.language = language
        self.generate_kwargs = generate_kwargs or {}

        dtype = None
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=self.device,
            torch_dtype=dtype,
        )
