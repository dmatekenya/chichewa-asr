"""
Transcript utility functions for ASR data preparation.

This module provides text normalisation and basic transcript statistics.
All normalisation steps are independently togglable so the same function
can be applied consistently across train, dev, holdout, and scoring.

Example
-------
>>> from data_utils.text_utils import normalize_transcript, count_words
>>> normalize_transcript("Ku Malawi kuno [noise] mpaka!")
'ku malawi kuno mpaka'
>>> count_words("ku malawi kuno mpaka")
4
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Unicode punctuation category covers Pc, Pd, Pe, Pf, Pi, Po, Ps.
# We keep the apostrophe (U+0027 and U+2019) because it appears in
# Chichewa contractions and possessives.
_KEEP_PUNCT = frozenset(["'", "’"])


def _strip_unicode_punctuation(text: str) -> str:
    """Remove Unicode punctuation characters, preserving apostrophes."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "P" or ch in _KEEP_PUNCT
    )


def normalize_transcript(
    text: str,
    unicode_nfc: bool = True,
    collapse_whitespace: bool = True,
    remove_brackets: bool = True,
    strip_punctuation: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Apply canonical normalisation to a transcript string.

    Each step is independently togglable so the same function can be used
    for training targets, evaluation references, and scoring without
    introducing inconsistencies between splits.

    Parameters
    ----------
    text : str
        Raw transcript string.
    unicode_nfc : bool, optional
        Normalise Unicode to NFC form (unifies composed and decomposed
        diacritics), by default True.
    collapse_whitespace : bool, optional
        Collapse tabs, newlines, and repeated spaces to a single space
        and strip leading/trailing whitespace, by default True.
    remove_brackets : bool, optional
        Remove bracketed and parenthetical annotations such as
        ``[noise]``, ``[laughter]``, ``(music)``, by default True.
    strip_punctuation : bool, optional
        Remove Unicode punctuation, preserving apostrophes, by default True.
    lowercase : bool, optional
        Convert text to lowercase, by default True.

    Returns
    -------
    str
        Normalised transcript string.

    Examples
    --------
    >>> normalize_transcript("Ku Malawi kuno [noise] mpaka!")
    'ku malawi kuno mpaka'
    >>> normalize_transcript("Iye anati: 'Zikomo.'")
    "iye anati 'zikomo'"
    >>> normalize_transcript("[laughter] Eee, ndili bwino.", remove_brackets=True)
    'eee ndili bwino'
    """
    if not text or not isinstance(text, str):
        return ""

    if unicode_nfc:
        text = unicodedata.normalize("NFC", text)

    if remove_brackets:
        text = re.sub(r"\[.*?\]", " ", text)   # [noise], [laughter], etc.
        text = re.sub(r"\(.*?\)", " ", text)   # (music), (unintelligible), etc.

    if strip_punctuation:
        text = _strip_unicode_punctuation(text)

    if lowercase:
        text = text.lower()

    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Transcript statistics
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """
    Count whitespace-delimited words in a string.

    Parameters
    ----------
    text : str
        Transcript string (normalise first for consistent counts).

    Returns
    -------
    int
        Number of words.
    """
    return len(text.split()) if text and text.strip() else 0


def count_chars(text: str, exclude_whitespace: bool = True) -> int:
    """
    Count characters in a string.

    Parameters
    ----------
    text : str
        Transcript string.
    exclude_whitespace : bool, optional
        If True (default), whitespace characters are not counted.

    Returns
    -------
    int
        Character count.
    """
    if not text:
        return 0
    if exclude_whitespace:
        return sum(1 for ch in text if not ch.isspace())
    return len(text)


def words_per_second(text: str, duration_sec: float) -> float:
    """
    Compute speaking rate in words per second.

    Parameters
    ----------
    text : str
        Transcript string (normalise first for consistent counts).
    duration_sec : float
        Audio clip duration in seconds.

    Returns
    -------
    float
        Words per second, or ``nan`` if duration is zero or invalid.
    """
    if not duration_sec or duration_sec <= 0:
        return float("nan")
    return count_words(text) / duration_sec


def chars_per_second(text: str, duration_sec: float) -> float:
    """
    Compute character rate (excluding whitespace) per second.

    More stable than words per second when word segmentation is
    language-sensitive or highly agglutinative.

    Parameters
    ----------
    text : str
        Transcript string.
    duration_sec : float
        Audio clip duration in seconds.

    Returns
    -------
    float
        Characters per second, or ``nan`` if duration is zero or invalid.
    """
    if not duration_sec or duration_sec <= 0:
        return float("nan")
    return count_chars(text) / duration_sec
