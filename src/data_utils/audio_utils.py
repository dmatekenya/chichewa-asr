"""
Utility module for generating standardized acoustic attributes for ASR datasets.

This module computes common audio-level attributes such as duration, RMS,
peak amplitude, silence ratio, a heuristic SNR estimate, voice-activity ratio,
effective bandwidth, crest factor, zero-crossing rate, and a rough RT60
estimate. It is designed to be used on fixed train/dev/test metadata files so
that attribute generation is standardized across experiments.

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
- VAD speech ratio uses a dual-criterion energy + spectral-flatness approach.
- RT60 is estimated via Schroeder backward integration; treat as approximate.
- ASR inference backends live in asr_backends.py, not here.
"""

from __future__ import annotations

# Standard library imports
import math
import tempfile
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import librosa
import numpy as np
import soundfile as sf


def get_total_audio_duration(folder: str | Path, exts=None) -> float:
    """
    Calculate the total duration of all audio files in a folder.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing audio files.
    exts : list or None
        List of file extensions to include (e.g., ['.wav', '.mp3']).
        If None, defaults to common audio types.

    Returns
    -------
    float
        Total duration in seconds.
    """
    if exts is None:
        exts = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    folder = Path(folder)
    total_duration = 0.0
    for ext in exts:
        for audio_file in folder.glob(f'*{ext}'):
            try:
                duration = librosa.get_duration(path=str(audio_file))
                total_duration += duration
            except Exception as e:
                print(f"Could not process {audio_file}: {e}")
    return total_duration


def get_total_audio_duration_with_units(
    folder: str | Path,
    exts=None,
    units: str = 'seconds',
) -> float:
    """
    Calculate the total duration of all audio files in a folder, with
    selectable output units.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing audio files.
    exts : list or None
        List of file extensions to include (e.g., ['.wav', '.mp3']).
        If None, defaults to common audio types.
    units : str, optional
        Output units: ``'seconds'``, ``'minutes'``, or ``'hours'``.
        Default is ``'seconds'``.

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
            per_channel_rms = [
                compute_rms(waveform[:, i].astype(np.float32))
                for i in range(num_channels)
            ]
            channel_info["channel_rms_diff"] = float(
                np.max(per_channel_rms) - np.min(per_channel_rms)
            )
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
    Frames whose RMS is below ``max_rms + silence_threshold_db`` are treated
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


def compute_clipping_ratio(waveform: np.ndarray, threshold: float = 0.99) -> float:
    """
    Compute the fraction of samples at or near maximum amplitude (clipping).

    Parameters
    ----------
    waveform : np.ndarray
        Input mono waveform (float32, range [-1, 1]).
    threshold : float, optional
        Fraction of max amplitude above which a sample is considered clipped,
        by default 0.99.

    Returns
    -------
    float
        Fraction of clipped samples in [0, 1].
        Values above ~0.01 indicate recording issues.
    """
    if waveform.size == 0:
        return float("nan")
    return float(np.mean(np.abs(waveform) >= threshold))


def compute_spectral_flatness(waveform: np.ndarray) -> float:
    """
    Compute mean spectral flatness of a waveform.

    Spectral flatness is the ratio of geometric mean to arithmetic mean of the
    power spectrum. Values near 1.0 indicate noise-like content; values near
    0.0 indicate tonal/speech content.

    Parameters
    ----------
    waveform : np.ndarray
        Input mono waveform.

    Returns
    -------
    float
        Mean spectral flatness in [0, 1].
    """
    if waveform.size == 0:
        return float("nan")
    flatness = librosa.feature.spectral_flatness(y=waveform)
    return float(np.mean(flatness))


def compute_zero_crossing_rate(waveform: np.ndarray) -> float:
    """
    Compute the mean zero-crossing rate (ZCR) of a waveform.

    A high ZCR in nominally silent frames can indicate electrical hiss,
    buzzing, or codec noise. For speech frames a typical ZCR is 0.02–0.15.

    Parameters
    ----------
    waveform : np.ndarray
        Input mono waveform.

    Returns
    -------
    float
        Mean ZCR across all frames (crossings per sample).
    """
    if waveform.size == 0:
        return float("nan")
    zcr = librosa.feature.zero_crossing_rate(y=waveform)
    return float(np.mean(zcr))


def compute_crest_factor_db(rms: float, peak_abs: float) -> float:
    """
    Compute the crest factor in dB (peak-to-RMS ratio).

    A very low crest factor indicates a compressed or over-normalised
    recording; a very high value indicates spiky transients.
    Healthy speech typically falls in the 10–20 dB range.

    Parameters
    ----------
    rms : float
        RMS amplitude of the waveform.
    peak_abs : float
        Peak absolute amplitude of the waveform.

    Returns
    -------
    float
        Crest factor in dB.
    """
    if rms <= 0 or peak_abs <= 0:
        return float("nan")
    return float(20.0 * safe_log10(peak_abs / (rms + 1e-12)))


def compute_effective_bandwidth_hz(
    waveform: np.ndarray,
    sample_rate: int,
    energy_threshold: float = 0.01,
) -> float:
    """
    Estimate the effective bandwidth of an audio file.

    The effective bandwidth is the highest frequency whose FFT magnitude
    exceeds ``energy_threshold`` times the spectral peak. Narrowband or
    telephone-quality audio will show values well below the Nyquist limit,
    typically below 4 kHz even when the file is stored at 16 kHz.

    Parameters
    ----------
    waveform : np.ndarray
        Input mono waveform.
    sample_rate : int
        Sample rate in Hz.
    energy_threshold : float, optional
        Fraction of the spectral peak below which a bin is ignored,
        by default 0.01 (i.e., 1 %).

    Returns
    -------
    float
        Estimated highest significant frequency in Hz.
    """
    if waveform.size == 0 or sample_rate <= 0:
        return float("nan")

    fft_mag = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate)
    threshold = energy_threshold * fft_mag.max()
    significant = fft_mag > threshold

    if not significant.any():
        return 0.0

    return float(freqs[significant].max())


def compute_vad_speech_ratio(
    waveform: np.ndarray,
    sample_rate: int,
    frame_length: int = 400,
    hop_length: int = 160,
    energy_threshold_db: float = -40.0,
    flatness_threshold: float = 0.3,
) -> float:
    """
    Estimate the fraction of frames that contain speech.

    Uses a dual-criterion energy-and-spectral-flatness VAD:
    a frame is labelled as **speech** when it is both above the energy
    threshold *and* has spectral flatness below ``flatness_threshold``
    (speech is more tonal than broadband noise).

    Parameters
    ----------
    waveform : np.ndarray
        Input mono waveform.
    sample_rate : int
        Sample rate in Hz (currently unused, reserved for future use).
    frame_length : int, optional
        Frame size in samples, by default 400.
    hop_length : int, optional
        Hop size in samples, by default 160.
    energy_threshold_db : float, optional
        Relative energy threshold in dB below the file maximum,
        by default -40.0.
    flatness_threshold : float, optional
        Maximum spectral flatness for a frame to be considered speech,
        by default 0.3.

    Returns
    -------
    float
        Fraction of speech frames in [0, 1].
    """
    if waveform.size == 0:
        return float("nan")

    frame_rms = compute_frame_rms(waveform, frame_length, hop_length)

    flatness = librosa.feature.spectral_flatness(
        y=waveform,
        n_fft=frame_length,
        hop_length=hop_length,
    ).squeeze(0)

    # Align lengths — librosa framing may differ by ±1 frame
    min_len = min(len(frame_rms), len(flatness))
    frame_rms = frame_rms[:min_len]
    flatness = flatness[:min_len]

    if frame_rms.size == 0:
        return float("nan")

    max_rms = float(np.max(frame_rms))
    if max_rms <= 0:
        return 0.0

    energy_threshold = max_rms * (10 ** (energy_threshold_db / 20.0))
    is_speech = (frame_rms > energy_threshold) & (flatness < flatness_threshold)

    return float(np.mean(is_speech))


def estimate_rt60(
    waveform: np.ndarray,
    sample_rate: int,
    decay_db: float = 20.0,
) -> float:
    """
    Estimate RT60 using Schroeder backward integration.

    The energy decay curve is computed by backward-integrating the squared
    waveform (Schroeder 1965). The time for the signal to decay ``decay_db``
    decibels is measured and then scaled to a full 60 dB decay.

    This estimate is most reliable on short, dry utterances recorded in a
    single room. Treat the result as approximate — a rough indicator of
    reverberation level rather than a calibrated acoustic measurement.

    Parameters
    ----------
    waveform : np.ndarray
        Input mono waveform.
    sample_rate : int
        Sample rate in Hz.
    decay_db : float, optional
        The dB drop used to measure decay before scaling to RT60,
        by default 20.0.

    Returns
    -------
    float
        Estimated RT60 in seconds, or ``nan`` if the decay cannot be
        measured reliably.
    """
    if waveform.size == 0 or sample_rate <= 0:
        return float("nan")

    # Schroeder backward integration
    energy = waveform[::-1] ** 2
    schroeder = np.cumsum(energy)[::-1]
    schroeder = schroeder / (schroeder[0] + 1e-12)
    schroeder_db = 10.0 * np.log10(schroeder + 1e-12)

    # Locate where the curve crosses 0 dB and -decay_db dB
    try:
        idx_start = int(np.where(schroeder_db <= 0.0)[0][0])
        idx_end = int(np.where(schroeder_db <= -decay_db)[0][0])
    except IndexError:
        return float("nan")

    if idx_end <= idx_start:
        return float("nan")

    t_decay = (idx_end - idx_start) / sample_rate
    return float(t_decay * (60.0 / decay_db))


def compute_audio_attributes(
    audio_path: str,
    frame_length: int = 400,
    hop_length: int = 160,
    silence_threshold_db: float = -40.0,
    noise_percentile: float = 20.0,
    bw_energy_threshold: float = 0.01,
    vad_flatness_threshold: float = 0.3,
    rt60_decay_db: float = 20.0,
) -> Dict[str, Any]:
    """
    Compute standardized acoustic quality attributes for one audio file.

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
    bw_energy_threshold : float, optional
        Spectral energy fraction used for bandwidth estimation, by default 0.01.
    vad_flatness_threshold : float, optional
        Spectral-flatness ceiling for VAD speech detection, by default 0.3.
    rt60_decay_db : float, optional
        dB decay used in the RT60 Schroeder estimate, by default 20.0.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all computed attributes.

    Attributes returned
    -------------------
    sample_rate, num_samples, duration_sec
        Basic file properties.
    rms, peak_abs
        Overall signal level (linear amplitude).
    rms_dbfs, peak_dbfs
        RMS and peak level in dBFS (20·log10 of the linear value).
    crest_factor_db
        Peak-to-RMS ratio in dB (10–20 dB is healthy speech).
    silence_ratio
        Fraction of energy-silent frames.
    vad_speech_ratio
        Fraction of frames containing speech (energy + spectral flatness).
    snr_db_est
        Heuristic SNR estimate in dB.
    clipping_ratio
        Fraction of samples at or near full-scale.
    spectral_flatness
        Mean spectral flatness (0 = tonal, 1 = noise-like).
    zero_crossing_rate
        Mean zero-crossing rate (crossings per sample).
    effective_bandwidth_hz
        Highest frequency with significant spectral energy.
    rt60_est_sec
        Rough reverberation time estimate in seconds.
    num_channels, channel_strategy, channel_rms_diff, dominant_channel
        Channel diagnostics.
    """
    waveform, sample_rate, channel_info = load_audio_mono(audio_path)

    duration_sec = len(waveform) / sample_rate if sample_rate > 0 else float("nan")
    rms = compute_rms(waveform)
    peak_abs = float(np.max(np.abs(waveform))) if waveform.size > 0 else float("nan")
    rms_dbfs = float(20.0 * safe_log10(rms))
    peak_dbfs = float(20.0 * safe_log10(peak_abs))

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
    clipping_ratio = compute_clipping_ratio(waveform)
    spectral_flatness = compute_spectral_flatness(waveform)
    zero_crossing_rate = compute_zero_crossing_rate(waveform)
    crest_factor_db = compute_crest_factor_db(rms, peak_abs)
    effective_bandwidth_hz = compute_effective_bandwidth_hz(
        waveform=waveform,
        sample_rate=sample_rate,
        energy_threshold=bw_energy_threshold,
    )
    vad_speech_ratio = compute_vad_speech_ratio(
        waveform=waveform,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        energy_threshold_db=silence_threshold_db,
        flatness_threshold=vad_flatness_threshold,
    )
    rt60_est_sec = estimate_rt60(
        waveform=waveform,
        sample_rate=sample_rate,
        decay_db=rt60_decay_db,
    )

    return {
        # Basic properties
        "sample_rate":            sample_rate,
        "num_samples":            int(len(waveform)),
        "duration_sec":           float(duration_sec),
        # Signal level
        "rms":                    rms,
        "peak_abs":               peak_abs,
        "rms_dbfs":               rms_dbfs,
        "peak_dbfs":              peak_dbfs,
        "crest_factor_db":        crest_factor_db,
        # Activity
        "silence_ratio":          silence_ratio,
        "vad_speech_ratio":       vad_speech_ratio,
        # Noise / quality
        "snr_db_est":             snr_db_est,
        "clipping_ratio":         clipping_ratio,
        # Spectral character
        "spectral_flatness":      spectral_flatness,
        "zero_crossing_rate":     zero_crossing_rate,
        "effective_bandwidth_hz": effective_bandwidth_hz,
        # Reverberation
        "rt60_est_sec":           rt60_est_sec,
        # Channel info
        "num_channels":           channel_info["num_channels"],
        "channel_strategy":       channel_info["channel_strategy"],
        "channel_rms_diff":       channel_info["channel_rms_diff"],
        "dominant_channel":       channel_info["dominant_channel"],
    }


def load_audio_16k_mono(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio, convert to mono, and resample to 16 kHz.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.

    Returns
    -------
    tuple[np.ndarray, int]
        Mono float32 waveform and sample rate (always 16000).
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
    Convert audio to a temporary 16 kHz mono WAV file and return the path.

    The caller is responsible for deleting the temporary file when done.

    Parameters
    ----------
    audio_path : str
        Path to the source audio file.

    Returns
    -------
    str
        Path to the temporary WAV file.
    """
    waveform, sr = load_audio_16k_mono(audio_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, waveform, sr)
    return tmp_path
