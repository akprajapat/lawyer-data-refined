"""
normalize.py
============
Step 1 — Name Normalization

Converts raw lawyer name strings into a clean, comparable form:
  - lowercase
  - remove punctuation
  - strip known titles / honorifics
  - normalize whitespace
  - collapse duplicate tokens (e.g. "adv adv" → "adv" before title removal)

Examples
--------
  "P.V. Shetty"         → "p v shetty"
  "Sr. Adv. Rajiv Kumar"→ "rajiv kumar"
  "Adv. AOR N. Sharma"  → "n sharma"
"""

import re
import unicodedata
from typing import Optional

# ── Title prefixes/suffixes to strip ──────────────────────────────────────────
# Order matters: longer phrases must come before shorter ones so "sr adv" is
# removed before a standalone "sr" would be matched.
_TITLE_PATTERNS: list[str] = [
    r"\bsenior\s+advocate\b",
    r"\bsr\.?\s*adv\.?\b",
    r"\bsenior\s+adv\.?\b",
    r"\badvocate[\s\-]on[\s\-]record\b",
    r"\ba\.?\s*o\.?\s*r\.?\b",   # A.O.R / AOR / A O R
    r"\baor\b",
    r"\badvocate\b",
    r"\badv\.?\b",
    r"\bmr\.?\b",
    r"\bms\.?\b",
    r"\bdr\.?\b",
    r"\bsr\.?\b",                 # bare "sr" after longer patterns consumed
]

_TITLE_RE = re.compile(
    "|".join(_TITLE_PATTERNS),
    flags=re.IGNORECASE,
)

# Characters to treat as word separators (keep letters, digits, space)
_PUNCT_RE = re.compile(r"[^\w\s]")

# Collapse runs of whitespace
_SPACE_RE = re.compile(r"\s+")


def _unicode_to_ascii(text: str) -> str:
    """Decompose accented characters and drop the accent marks."""
    return (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def normalize_name(raw: str) -> str:
    """
    Return a normalised form of *raw* suitable for comparison.

    Parameters
    ----------
    raw : str
        The raw name string from appearances.csv.

    Returns
    -------
    str
        Cleaned, lower-cased name with titles and punctuation removed.
        Returns empty string if the input is empty/null.
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = raw.strip()

    # 1. Transliterate accented characters
    text = _unicode_to_ascii(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove punctuation (replace with space to avoid token merging)
    text = _PUNCT_RE.sub(" ", text)

    # 4. Strip titles
    text = _TITLE_RE.sub(" ", text)

    # 5. Collapse whitespace
    text = _SPACE_RE.sub(" ", text).strip()

    return text


def normalize_registry_name(raw: Optional[str]) -> str:
    """
    Same as normalize_name but also handles None gracefully.
    Used when normalising entries from the AOR / SR master files.
    """
    if raw is None:
        return ""
    return normalize_name(str(raw))


# ── Designation label normalisation ───────────────────────────────────────────

_DESIG_PATTERNS: dict[str, re.Pattern] = {
    "SENIOR": re.compile(
        r"senior|sr\.?\s*adv|sr\s*advocate", flags=re.IGNORECASE
    ),
    "AOR": re.compile(
        r"a\.?\s*o\.?\s*r\.?|aor|advocate[\s\-]+on[\s\-]+record",
        flags=re.IGNORECASE,
    ),
    "ADVOCATE": re.compile(
        r"adv\.?|advocate", flags=re.IGNORECASE
    ),
}


def normalize_designation(raw: Optional[str]) -> str:
    """
    Map a noisy designation string to one of: ADVOCATE | AOR | SENIOR.

    Priority: SENIOR > AOR > ADVOCATE (most specific first).
    Defaults to ADVOCATE if nothing matches.
    """
    if not raw or not isinstance(raw, str):
        return "ADVOCATE"

    for label, pattern in _DESIG_PATTERNS.items():
        if pattern.search(raw):
            return label

    return "ADVOCATE"