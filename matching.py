"""
matching.py
===========
Steps 4–6 — Candidate Scoring and Match Decision

Similarity components
---------------------
  exact_match          1.00  if normalised names are identical
  token_set_ratio      difflib SequenceMatcher on sorted token sets
  surname_sim          character-level similarity of surnames
  given_name_sim       compares the non-surname portion of both names
  initials_compat      checks initials are consistent with full tokens
  phonetic_match       +0.15 bonus when Soundex codes agree
  common_surname_pen   penalty for high-freq surname + divergent given names

Hard-reject rules (score → 0 immediately, never merge)
-------------------------------------------------------
  1. same_case_conflict        — co-counsel in the same case
  2. surname similarity < 0.62 — clearly different surnames
  3. hard_initial_conflict     — initial in one name contradicts full token
                                 in the other (e.g. "R. Shetty" vs "P.V. Shetty")
  4. first_name_conflict       — both have a full first name whose edit
                                 similarity is below 0.55 (NEW)
                                 e.g. "Rohit Vikram Choudhary" vs
                                      "Rajeev Kumar Choudhary"
  5. both multi-token + given  — given_name_sim < 0.40 when both sides
     name too divergent          have 2+ full tokens (raised from 0.25)

Thresholds
----------
  MERGE_THRESHOLD          0.78  (default conservative)
  COMMON_SURNAME_THRESHOLD 0.90  (common Indian surnames)
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from features import NameFeatures

# ── Thresholds ────────────────────────────────────────────────────────────────
MERGE_THRESHOLD = 0.78
COMMON_SURNAME_THRESHOLD = 0.90
HIGH_CONFIDENCE = 0.92

_COMMON_SURNAMES: frozenset[str] = frozenset({
    "singh", "kumar", "sharma", "gupta", "verma", "mishra", "pandey",
    "patel", "joshi", "rao", "reddy", "nair", "pillai", "iyer", "menon",
    "chauhan", "bhatia", "jain", "agarwal", "aggarwal", "chaudhary",
    "choudhary", "chaudhari", "dubey", "shukla", "srivastava", "shrivastava",
    "khanna", "bose", "banerjee", "mukherjee", "chatterjee", "ghosh",
    "das", "sen", "roy", "paul", "prasad", "naik", "shah", "mehta",
    "thakur", "saxena", "tiwari", "chaurasia", "chourasia",
})


# ── Score breakdown ───────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    id_a: int
    id_b: int
    norm_a: str
    norm_b: str
    exact_match: float = 0.0
    token_set_ratio: float = 0.0
    surname_sim: float = 0.0
    initials_compat: float = 0.0
    given_name_sim: float = 0.0
    phonetic_match: float = 0.0
    common_surname_penalty: float = 0.0
    same_case_conflict: bool = False
    final_score: float = 0.0
    merged: bool = False

    def explain(self) -> str:
        if self.same_case_conflict:
            return "BLOCKED: same-case conflict"
        if self.exact_match == 1.0:
            return "exact-normalised-match | final=1.000"
        parts = []
        if self.token_set_ratio > 0:
            parts.append(f"token-sim={self.token_set_ratio:.2f}")
        if self.surname_sim > 0:
            parts.append(f"surname-sim={self.surname_sim:.2f}")
        if self.given_name_sim > 0:
            parts.append(f"given-name-sim={self.given_name_sim:.2f}")
        if self.initials_compat > 0:
            parts.append(f"initials-compat={self.initials_compat:.2f}")
        if self.phonetic_match > 0:
            parts.append("phonetic-match")
        if self.common_surname_penalty < 0:
            parts.append(f"common-surname-pen={self.common_surname_penalty:.2f}")
        parts.append(f"final={self.final_score:.3f}")
        return " | ".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seq_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _token_set_ratio(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    jaccard = len(sa & sb) / len(sa | sb)
    seq = _seq_ratio(" ".join(sorted(sa)), " ".join(sorted(sb)))
    return max(jaccard, seq)


def _first_full_token(feat: NameFeatures) -> str:
    """
    Return the first full (non-initial, non-surname) token.
    e.g.  "rohit vikram choudhary" → "rohit"
          "r choudhary"            → ""   (only initial before surname)
          "choudhary"              → ""   (surname only)
    """
    # full_tokens = all tokens with len > 1
    # surname is last full token; given = full_tokens[:-1]
    given = feat.full_tokens[:-1] if len(feat.full_tokens) > 1 else []
    return given[0] if given else ""


def _first_name_conflict(feat_a: NameFeatures, feat_b: NameFeatures) -> bool:
    """
    Return True when BOTH names carry a full first name (not just an initial)
    and those first names are clearly different people.

    Logic:
    - Extract first full token (excluding surname) from each side.
    - If either side has only initials before the surname, we cannot confirm
      a conflict — return False (benefit of the doubt for abbreviations).
    - If both have a full first name AND their edit similarity < 0.55
      AND neither first name starts with the same letter as the other's
      first-name initial → definite different people → True.

    Examples
    --------
    "rohit vikram choudhary" vs "rajeev kumar choudhary"
        fn_a="rohit", fn_b="rajeev"  sim≈0.31  'r'=='r' but full sims low
        → still different: seq_ratio < 0.55 AND token_set of full names
          clearly diverges → True

    "r vikram choudhary" vs "rohit vikram choudhary"
        fn_a=""  (first token is initial 'r')  → False  (abbreviation case)

    "siddhartha dave" vs "siddharth dave"
        fn_a="siddhartha", fn_b="siddharth"  sim≈0.94  → False  (same person)
    """
    fn_a = _first_full_token(feat_a)
    fn_b = _first_full_token(feat_b)

    # One or both sides have no full first name — can't confirm conflict
    if not fn_a or not fn_b:
        return False

    sim = _seq_ratio(fn_a, fn_b)
    if sim >= 0.55:
        return False   # similar enough — could be same person

    # Check if one is an abbreviated form of the other
    # e.g. "raj" could be short for "rajesh" — but we only skip this
    # when one is clearly a prefix of the other
    short, long = (fn_a, fn_b) if len(fn_a) <= len(fn_b) else (fn_b, fn_a)
    if long.startswith(short) and len(short) >= 3:
        return False   # abbreviation / prefix

    # Both have full first names that are clearly different → conflict
    return True


def _hard_initial_conflict(feat_a: NameFeatures, feat_b: NameFeatures) -> bool:
    """
    Return True if an initial in one name contradicts a full token in the other.
    e.g. "P. V. Shetty" (initials p,v) vs "R. Shetty" (initial r):
         'r' is not among the starts of "p v shetty" tokens → conflict.
    """
    a_starts = {t[0] for t in feat_a.tokens if t}
    b_starts = {t[0] for t in feat_b.tokens if t}

    for init in feat_b.initials:
        if init not in a_starts:
            return True
    for init in feat_a.initials:
        if init not in b_starts:
            return True
    return False


def _given_name_similarity(feat_a: NameFeatures, feat_b: NameFeatures) -> float:
    """Compare the non-surname portion of two names."""
    given_a = feat_a.full_tokens[:-1] if len(feat_a.full_tokens) > 1 else []
    given_b = feat_b.full_tokens[:-1] if len(feat_b.full_tokens) > 1 else []
    all_a = given_a + feat_a.initials
    all_b = given_b + feat_b.initials

    if not all_a and not all_b:
        return 0.5   # both single-token (surname only) — neutral
    if not all_a or not all_b:
        return 0.5   # one is surname-only — can't compare

    return _seq_ratio(" ".join(all_a), " ".join(all_b))


def _initials_compatibility(feat_a: NameFeatures, feat_b: NameFeatures) -> float:
    if not feat_a.initials and not feat_b.initials:
        return 0.5

    b_starts = {t[0] for t in feat_b.tokens if t}
    a_starts = {t[0] for t in feat_a.tokens if t}
    match_score = sum(1 for i in feat_a.initials if i in b_starts) + \
                  sum(1 for i in feat_b.initials if i in a_starts)
    total = len(feat_a.initials) + len(feat_b.initials)
    return min(1.0, match_score / max(total, 1))


def _common_surname_penalty(
    feat_a: NameFeatures, feat_b: NameFeatures, given_name_sim: float
) -> float:
    if feat_a.surname not in _COMMON_SURNAMES:
        return 0.0
    if given_name_sim < 0.5:
        return -0.20
    elif given_name_sim < 0.70:
        return -0.10
    return 0.0


# ── Main scoring function ─────────────────────────────────────────────────────

def score_pair(
    id_a: int,
    id_b: int,
    feat_a: NameFeatures,
    feat_b: NameFeatures,
    same_case_conflict: bool = False,
) -> ScoreBreakdown:

    bd = ScoreBreakdown(
        id_a=id_a, id_b=id_b,
        norm_a=feat_a.norm_name, norm_b=feat_b.norm_name,
    )

    # ── Hard block 1: co-counsel ──────────────────────────────────────────
    if same_case_conflict:
        bd.same_case_conflict = True
        return bd   # final_score=0, merged=False

    na, nb = feat_a.norm_name, feat_b.norm_name

    # ── Exact normalised match ────────────────────────────────────────────
    if na == nb:
        bd.exact_match = 1.0
        bd.final_score = 1.0
        bd.merged = True
        return bd

    bd.token_set_ratio = _token_set_ratio(na, nb)

    # ── Hard block 2: surnames too different ──────────────────────────────
    bd.surname_sim = _seq_ratio(feat_a.surname, feat_b.surname)
    if bd.surname_sim < 0.62:
        return bd

    # ── Hard block 3: initial contradicts full token ──────────────────────
    if _hard_initial_conflict(feat_a, feat_b):
        return bd

    # ── Hard block 4: both have full first names that clearly differ ──────
    if _first_name_conflict(feat_a, feat_b):
        return bd

    bd.given_name_sim = _given_name_similarity(feat_a, feat_b)

    # ── Hard block 5: multi-token names with divergent given names ─────────
    # Raised threshold to 0.40 (was 0.25) for stricter separation
    if (
        len(feat_a.full_tokens) >= 2
        and len(feat_b.full_tokens) >= 2
        and bd.given_name_sim < 0.40
    ):
        return bd

    bd.initials_compat = _initials_compatibility(feat_a, feat_b)

    # ── Phonetic bonus ────────────────────────────────────────────────────
    if (
        feat_a.phonetic_surname == feat_b.phonetic_surname
        and feat_a.phonetic_surname != "0000"
    ):
        bd.phonetic_match = 0.15

    # ── Common-surname penalty ────────────────────────────────────────────
    bd.common_surname_penalty = _common_surname_penalty(
        feat_a, feat_b, bd.given_name_sim
    )

    # ── Weighted score ────────────────────────────────────────────────────
    score = (
        0.50 * bd.token_set_ratio
        + 0.18 * bd.surname_sim
        + 0.17 * bd.given_name_sim
        + 0.10 * bd.initials_compat
        + bd.phonetic_match
        + bd.common_surname_penalty
    )

    bd.final_score = max(0.0, min(1.0, score))

    effective_threshold = (
        COMMON_SURNAME_THRESHOLD
        if feat_a.surname in _COMMON_SURNAMES
        else MERGE_THRESHOLD
    )
    bd.merged = bd.final_score >= effective_threshold
    return bd


def score_all_pairs(
    candidate_pairs: set[tuple[int, int]],
    features: dict[int, NameFeatures],
    forbidden_pairs: set[tuple[int, int]],
) -> list[ScoreBreakdown]:
    results: list[ScoreBreakdown] = []
    for id_a, id_b in candidate_pairs:
        feat_a = features.get(id_a)
        feat_b = features.get(id_b)
        if feat_a is None or feat_b is None:
            continue
        conflict = (id_a, id_b) in forbidden_pairs
        bd = score_pair(id_a, id_b, feat_a, feat_b, same_case_conflict=conflict)
        results.append(bd)
    return results