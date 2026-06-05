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



# Unicode punctuation category covers Pc, Pd, Pe, Pf, Pi, Po, Ps.
# We keep U+0027 (straight ASCII apostrophe) because it appears in
# Chichewa contractions and possessives (e.g. ng’ono, ng’ona).
_STRAIGHT_APOS = chr(0x27)  # U+0027 ASCII apostrophe — editor-proof, not subject to autocorrect
_KEEP_PUNCT = frozenset([_STRAIGHT_APOS])

# All Unicode characters that should be treated as an apostrophe in Chichewa
# text, plus the Unicode replacement character U+FFFD which appears when an
# apostrophe survives a codec mismatch (e.g. UTF-8 read as Latin-1).
_APOSTROPHE_VARIANTS: dict[str, str] = {
    "’": _STRAIGHT_APOS,  # ‘ RIGHT SINGLE QUOTATION MARK (most common curly apostrophe)
    "‘": _STRAIGHT_APOS,  # ‘ LEFT SINGLE QUOTATION MARK (sometimes used as apostrophe)
    "‛": _STRAIGHT_APOS,  # ‛ SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "ʼ": _STRAIGHT_APOS,  # ʼ MODIFIER LETTER APOSTROPHE
    "ʹ": _STRAIGHT_APOS,  # ʹ MODIFIER LETTER PRIME
    "ʻ": _STRAIGHT_APOS,  # ʻ MODIFIER LETTER TURNED COMMA
    "`": _STRAIGHT_APOS,  # ` GRAVE ACCENT (used as apostrophe in some transcription tools)
    "´": _STRAIGHT_APOS,  # ´ ACUTE ACCENT
    "�": _STRAIGHT_APOS,  # REPLACEMENT CHARACTER — codec-corrupted apostrophe in ng’ clusters
}


_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _strip_unicode_punctuation(text: str) -> str:
    """Remove Unicode punctuation characters, preserving apostrophes."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "P" or ch in _KEEP_PUNCT
    )


def normalize_apostrophes(text: str) -> str:
    """
    Normalise all apostrophe-like characters to a plain ASCII apostrophe
    (U+0027) and recover apostrophes that were corrupted by codec mismatches.

    Chichewa uses the apostrophe phonetically in clusters such as ``ng'ono``
    and ``ng'ona``. When source files are read with the wrong encoding the
    apostrophe can appear as the Unicode replacement character ``?``
    (U+FFFD) or as various curly/modifier-letter variants. This function
    maps all of those to ``'`` so downstream steps treat them consistently.

    Parameters
    ----------
    text : str
        Raw or partially normalised transcript string.

    Returns
    -------
    str
        Text with all apostrophe variants replaced by ``'``.

    Examples
    --------
    >>> normalize_apostrophes("kakang�ono")   # U+FFFD replacement char
    "kakang'ono"
    >>> normalize_apostrophes("ng’ono")       # curly right single quote
    "ng'ono"
    >>> normalize_apostrophes("ngʼono")       # modifier letter apostrophe
    "ng'ono"
    """
    if not text:
        return text
    return text.translate(str.maketrans(_APOSTROPHE_VARIANTS))


def remove_newlines(text: str, replacement: str = " ") -> str:
    """
    Replace all newline sequences with ``replacement`` and collapse the
    result to a single space between words.

    Handles ``\\n``, ``\\r\\n``, and ``\\r`` — including cases where
    newlines appear mid-sentence, at the start, or at the end of the
    string. The function is deliberately narrow: it only fixes newlines
    and trailing/leading whitespace, leaving everything else untouched.
    Use :func:`normalize_transcript` when broader cleaning is also needed.

    Parameters
    ----------
    text : str
        Raw transcript string, potentially containing newline characters.
    replacement : str, optional
        String to substitute for each newline sequence, by default a
        single space ``" "``. Pass ``""`` to delete newlines entirely.

    Returns
    -------
    str
        Transcript with newlines replaced and surrounding whitespace
        stripped.

    Examples
    --------
    >>> remove_newlines("Ndili bwino\\nKaya inu?")
    'Ndili bwino Kaya inu?'
    >>> remove_newlines("Zikomo.\\r\\nTionana.")
    'Zikomo. Tionana.'
    >>> remove_newlines("\\nKu Malawi kuno\\n")
    'Ku Malawi kuno'
    >>> remove_newlines("word1\\n\\nword2", replacement=" ")
    'word1 word2'
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"\r\n|\r|\n", replacement, text)
    if replacement == " ":
        text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _int_to_words_en(n: int) -> str:
    """Recursively convert a non-negative integer to English words."""
    if n < 0:
        return "minus " + _int_to_words_en(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        rest = (" " + _ONES[n % 10]) if n % 10 else ""
        return _TENS[n // 10] + rest
    if n < 1_000:
        rest = (" " + _int_to_words_en(n % 100)) if n % 100 else ""
        return _ONES[n // 100] + " hundred" + rest
    if n < 1_000_000:
        rest = (" " + _int_to_words_en(n % 1_000)) if n % 1_000 else ""
        return _int_to_words_en(n // 1_000) + " thousand" + rest
    if n < 1_000_000_000:
        rest = (" " + _int_to_words_en(n % 1_000_000)) if n % 1_000_000 else ""
        return _int_to_words_en(n // 1_000_000) + " million" + rest
    return str(n)  # fallback for numbers >= 1 billion


def verbalize_digits(text: str) -> str:
    """
    Replace every standalone digit sequence in *text* with its English
    spoken-word equivalent.

    Digits are converted using a self-contained English number engine so
    no extra dependency is required. The function handles integers from 0
    up to 999,999,999; anything larger is left as-is.

    Only pure digit runs are replaced — digits that are already embedded
    inside a word (e.g. ``mp3``, ``covid19``) are left untouched because
    the regex requires a word boundary on both sides.

    Parameters
    ----------
    text : str
        Transcript string, possibly containing digit sequences.

    Returns
    -------
    str
        Text with digit sequences replaced by English words.

    Examples
    --------
    >>> verbalize_digits("analowa 5 pa bwalo")
    'analowa five pa bwalo'
    >>> verbalize_digits("chaka cha 2024")
    'chaka cha two thousand twenty four'
    >>> verbalize_digits("ndi 13 mtengo")
    'ndi thirteen mtengo'
    >>> verbalize_digits("mp3 player")   # embedded digit — unchanged
    'mp3 player'
    """
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        n = int(match.group())
        if n < 1_000_000_000:
            return _int_to_words_en(n)
        return match.group()  # leave very large numbers alone

    return re.sub(r"\b\d+\b", _replace, text)


def normalize_transcript(
    text: str,
    unicode_nfc: bool = True,
    fix_apostrophes: bool = True,
    strip_newlines: bool = True,
    verbalize_numbers: bool = False,
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
    fix_apostrophes : bool, optional
        Map all apostrophe-like characters and the Unicode replacement
        character ``?`` to a plain ASCII apostrophe, recovering
        Chichewa ``ng'`` clusters corrupted by codec mismatches,
        by default True.
    strip_newlines : bool, optional
        Replace ``\\n``, ``\\r\\n``, and ``\\r`` with a space before any
        other processing, by default True.
    collapse_whitespace : bool, optional
        Collapse repeated spaces to a single space and strip
        leading/trailing whitespace, by default True.
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
    >>> normalize_transcript("Ndili bwino\\nKaya inu?")
    'ndili bwino kaya inu'
    >>> normalize_transcript("Iye anati: 'Zikomo.'")
    "iye anati 'zikomo'"
    >>> normalize_transcript("kakang�ono")   # codec-corrupted apostrophe
    "kakang'ono"
    """
    if not text or not isinstance(text, str):
        return ""

    if unicode_nfc:
        text = unicodedata.normalize("NFC", text)

    if fix_apostrophes:
        text = normalize_apostrophes(text)

    if strip_newlines:
        text = remove_newlines(text)

    if verbalize_numbers:
        text = verbalize_digits(text)

    if remove_brackets:
        text = re.sub(r"\[.*?\]", " ", text)
        text = re.sub(r"\(.*?\)", " ", text)

    if strip_punctuation:
        text = _strip_unicode_punctuation(text)

    if lowercase:
        text = text.lower()

    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


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
