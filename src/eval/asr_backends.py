"""
ASR inference backends for the Chichewa ASR project.

This module provides backend classes for running automatic speech recognition
using different model providers (Hugging Face Transformers, OpenAI Whisper API,
etc.). Audio loading and quality utilities live in data_utils/audio_utils.py.

Example
-------
>>> backend = HFASRBackend(model_id="facebook/mms-300m", device="cpu")
>>> text = backend.transcribe("audio/sample.wav")
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from jiwer import cer, wer
from openai import OpenAI
from transformers import pipeline

from data_utils.audio_utils import write_temp_wav_16k


class HFASRBackend:
    """
    Generic Hugging Face ASR backend using ``pipeline()``.

    Works for Whisper, wav2vec2, MMS, and any other ASR-capable checkpoint
    available on the Hugging Face Hub.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier (e.g. ``"facebook/mms-300m"``).
    device : str, optional
        ``"cpu"`` or ``"cuda"``, by default ``"cpu"``.
    torch_dtype : str or None, optional
        Optional dtype override: ``"float16"`` or ``"bfloat16"``.
    chunk_length_s : int or None, optional
        Chunk length for long-form transcription (Whisper-style).
    batch_size : int, optional
        Batch size for the pipeline, by default 1.
    language : str or None, optional
        Force a specific language (model-dependent).
    generate_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``model.generate()``.
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

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a single audio file.

        The file is resampled to 16 kHz mono before being passed to the model.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.

        Returns
        -------
        str
            Transcribed text.
        """
        tmp_path = write_temp_wav_16k(audio_path)
        try:
            kwargs: Dict[str, Any] = {}
            if self.chunk_length_s is not None:
                kwargs["chunk_length_s"] = self.chunk_length_s
            if self.language is not None:
                kwargs["generate_kwargs"] = {
                    **self.generate_kwargs,
                    "language": self.language,
                }
            else:
                kwargs["generate_kwargs"] = self.generate_kwargs

            result = self.pipe(tmp_path, batch_size=self.batch_size, **kwargs)
            return result["text"].strip()
        finally:
            os.unlink(tmp_path)

    def evaluate(
        self,
        audio_paths: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Transcribe a list of files and compute WER and CER against references.

        Parameters
        ----------
        audio_paths : list of str
            Paths to audio files.
        references : list of str
            Ground-truth transcripts aligned with ``audio_paths``.

        Returns
        -------
        dict
            Dictionary with keys ``"wer"`` and ``"cer"``.
        """
        hypotheses = [self.transcribe(p) for p in audio_paths]
        return {
            "wer": float(wer(references, hypotheses)),
            "cer": float(cer(references, hypotheses)),
        }
