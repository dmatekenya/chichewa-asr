"""
Silero VAD wrapper for speech activity detection.

Provides segment-level and clip-level VAD metrics. The Silero model is
loaded once per process via torch.hub and cached with lru_cache so
repeated calls within a batch do not re-download or re-initialise it.

If the silero-vad package is unavailable the functions fall back to the
heuristic energy-based VAD already implemented in audio_utils.py, returning
nan for segment-count and mean-segment-duration fields.

Example
-------
>>> import soundfile as sf
>>> waveform, sr = sf.read("audio/sample.wav")
>>> from data_utils.vad_utils import compute_vad_metrics
>>> metrics = compute_vad_metrics(waveform, sr)
>>> print(metrics["speech_ratio"], metrics["num_speech_segments"])
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List

import librosa
import numpy as np
import torch

from .audio_utils import compute_vad_speech_ratio


@functools.lru_cache(maxsize=1)
def _load_silero_model():
    """
    Load the Silero VAD model via torch.hub (cached after first call).

    Returns
    -------
    tuple
        (model, get_speech_timestamps) from the silero-vad package,
        or (None, None) if the package is not available.
    """
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
            trust_repo=True,
        )
        return model, utils[0]
    except Exception:
        return None, None


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _safe_log10(x: float, eps: float = 1e-12) -> float:
    return math.log10(max(x, eps))


def _resample_to_16k(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample waveform to 16 kHz mono float32."""
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    return waveform


def compute_vad_segments(
    waveform: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> List[Dict[str, float]]:
    """
    Detect speech segments using Silero VAD.

    Parameters
    ----------
    waveform : np.ndarray
        Input audio waveform (any sample rate, mono or stereo).
    sample_rate : int
        Sample rate of ``waveform`` in Hz.
    threshold : float, optional
        Silero speech probability threshold, by default 0.5.
    min_speech_duration_ms : int, optional
        Minimum speech segment length to keep in ms, by default 250.
    min_silence_duration_ms : int, optional
        Minimum silence gap to split segments in ms, by default 100.

    Returns
    -------
    list of dict
        Each dict has keys ``start_sec``, ``end_sec``, ``duration_sec``.
        Returns an empty list if the model is unavailable.
    """
    model, get_speech_timestamps = _load_silero_model()
    if model is None:
        return []

    wav16k = _resample_to_16k(waveform, sample_rate)
    wav_tensor = torch.from_numpy(wav16k)

    timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        threshold=threshold,
        sampling_rate=16000,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
    )

    return [
        {
            "start_sec":    float(ts["start"]),
            "end_sec":      float(ts["end"]),
            "duration_sec": float(ts["end"] - ts["start"]),
        }
        for ts in timestamps
    ]


def compute_vad_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> Dict[str, Any]:
    """
    Compute clip-level VAD quality metrics using Silero VAD.

    Falls back to the heuristic energy-based ``compute_vad_speech_ratio``
    from ``audio_utils`` when Silero is unavailable, setting
    ``num_speech_segments`` and ``mean_segment_duration`` to ``nan``.

    Parameters
    ----------
    waveform : np.ndarray
        Input audio waveform (any sample rate, mono or stereo).
    sample_rate : int
        Sample rate of ``waveform`` in Hz.
    threshold : float, optional
        Silero speech probability threshold, by default 0.5.
    min_speech_duration_ms : int, optional
        Minimum speech segment length in ms, by default 250.
    min_silence_duration_ms : int, optional
        Minimum silence gap to split segments in ms, by default 100.

    Returns
    -------
    dict
        Keys:

        ``speech_duration_sec``
            Total duration of detected speech spans in seconds.
        ``speech_ratio``
            Fraction of the clip that is speech (0–1).
        ``silence_ratio``
            Fraction of the clip that is non-speech (1 - speech_ratio).
        ``num_speech_segments``
            Number of distinct VAD speech spans.
        ``mean_segment_duration``
            Mean duration of speech spans in seconds.
        ``snr_proxy_db``
            RMS of speech frames minus RMS of non-speech frames in dB.
            Positive values indicate speech stands out above the background.
    """
    model, _ = _load_silero_model()
    silero_available = model is not None

    mono = waveform.mean(axis=1).astype(np.float32) if waveform.ndim == 2 else waveform.astype(np.float32)
    total_samples = len(mono)
    total_duration = total_samples / sample_rate if sample_rate > 0 else float("nan")

    if silero_available:
        segments = compute_vad_segments(
            waveform=waveform,
            sample_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        speech_duration = sum(s["duration_sec"] for s in segments)
        num_segments = len(segments)
        mean_seg_dur = speech_duration / num_segments if num_segments > 0 else float("nan")
        speech_ratio = min(
            speech_duration / total_duration if total_duration and total_duration > 0 else float("nan"),
            1.0,
        )

        speech_mask = np.zeros(total_samples, dtype=bool)
        for seg in segments:
            start_idx = max(0, min(int(seg["start_sec"] * sample_rate), total_samples))
            end_idx   = max(0, min(int(seg["end_sec"]   * sample_rate), total_samples))
            speech_mask[start_idx:end_idx] = True

        speech_samples     = mono[speech_mask]
        non_speech_samples = mono[~speech_mask]
        snr_proxy = (
            20.0 * _safe_log10(_rms(speech_samples)) - 20.0 * _safe_log10(_rms(non_speech_samples))
            if speech_samples.size > 0 and non_speech_samples.size > 0
            else float("nan")
        )

    else:
        speech_ratio    = compute_vad_speech_ratio(waveform=mono, sample_rate=sample_rate)
        speech_duration = speech_ratio * total_duration if not math.isnan(speech_ratio) else float("nan")
        num_segments    = float("nan")
        mean_seg_dur    = float("nan")
        snr_proxy       = float("nan")

    silence_ratio = 1.0 - speech_ratio if not math.isnan(speech_ratio) else float("nan")

    return {
        "speech_duration_sec":   float(speech_duration),
        "speech_ratio":          float(speech_ratio),
        "silence_ratio":         float(silence_ratio),
        "num_speech_segments":   float(num_segments),
        "mean_segment_duration": float(mean_seg_dur),
        "snr_proxy_db":          float(snr_proxy),
    }
