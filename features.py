"""
features.py
===========
Step 2 — Feature Extraction

From a normalised name string, extract structured features used for
blocking and similarity scoring:

  - tokens       : list of word tokens
  - surname      : last token (heuristic; works for Indian names)
  - initials     : single-character tokens (abbreviated first/middle names)
  - full_tokens  : tokens that are more than one character
  - phonetic_surname : Soundex code of the surname
  - token_count  : number of tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Soundex implementation (no external dependency) ────────────────────────────

_SOUNDEX_TABLE = str.maketrans(
    "bfpvcgjkqsxzdtlmnr",
    "111122222222334556",
)


def soundex(word: str) -> str:
    """
    Return the American Soundex code for *word*.

    Returns "0000" for empty/non-alphabetic input.

    Bug fixed: str.translate() returns the original character unchanged for
    letters not in the mapping (vowels, h, w, y).  The old code checked
    `if code and code != "0"` which let vowels like 'a','i' pass through,
    producing codes like "Si1a" instead of "S140".  Now we check
    `digit.isdigit()` — only actual digit codes are appended.
    Vowels also reset the prev_digit guard so adjacent same-coded
    consonants separated by a vowel are each counted (standard Soundex).
    """
    if not word:
        return "0000"

    word = word.lower()
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return "0000"

    first = letters[0].upper()
    prev_digit = letters[0].translate(_SOUNDEX_TABLE)
    if not prev_digit.isdigit():
        prev_digit = ""   # first letter is a vowel — no suppression seed

    coded = first

    for char in letters[1:]:
        digit = char.translate(_SOUNDEX_TABLE)
        if not digit.isdigit():
            # vowel / h / w / y — separator; reset duplicate-suppression
            prev_digit = ""
            continue
        if digit != prev_digit:
            coded += digit
        prev_digit = digit

    return coded[:4].ljust(4, "0")


# ── Feature dataclass ─────────────────────────────────────────────────────────

@dataclass
class NameFeatures:
    """Structured features for one normalised name."""

    raw_name: str                       # original raw string
    norm_name: str                      # after normalize_name()
    tokens: list[str] = field(default_factory=list)
    surname: str = ""
    initials: list[str] = field(default_factory=list)
    full_tokens: list[str] = field(default_factory=list)   # len > 1
    phonetic_surname: str = "0000"
    token_count: int = 0
    initials_key: str = ""              # e.g. "pvs" for P V Shetty


def extract_features(raw_name: str, norm_name: str) -> NameFeatures:
    """
    Build a NameFeatures object from the normalised name.

    Parameters
    ----------
    raw_name  : original string (stored for tracing)
    norm_name : output of normalize_name()
    """
    feat = NameFeatures(raw_name=raw_name, norm_name=norm_name)

    if not norm_name:
        return feat

    feat.tokens = norm_name.split()
    feat.token_count = len(feat.tokens)

    # Single-char tokens are initials; multi-char are full name tokens
    feat.initials = [t for t in feat.tokens if len(t) == 1]
    feat.full_tokens = [t for t in feat.tokens if len(t) > 1]

    # Surname = last full token; fall back to last token if all are initials
    if feat.full_tokens:
        feat.surname = feat.full_tokens[-1]
    elif feat.tokens:
        feat.surname = feat.tokens[-1]

    # Initials key: first character of each token, in order
    feat.initials_key = "".join(t[0] for t in feat.tokens if t)

    feat.phonetic_surname = soundex(feat.surname)

    return feat