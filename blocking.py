"""
blocking.py
===========
Step 3 — Blocking

Generate candidate pairs for comparison without doing full O(n²)
pairwise comparison.  Three overlapping blocking keys are used so that
different kinds of name variation are covered:

  1. exact_surname     — token-exact last name
  2. phonetic_surname  — Soundex code of surname (catches spelling variants)
  3. initials_surname  — initials-key + surname (catches "P V Shetty" vs
                         "P.V. Shetty" but not "Shetty" alone)

A pair is a candidate if it shares ANY blocking key.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterator

import pandas as pd

from features import NameFeatures


def _build_block_index(
    features: dict[int, NameFeatures],
) -> dict[str, list[int]]:
    """
    Build inverted index: blocking_key → [appearance_id, ...].

    Three key namespaces are used (prefixed) to avoid collisions:
      "S:{surname}"
      "P:{phonetic_surname}"
      "IS:{initials_key}:{surname}"
    """
    index: dict[str, list[int]] = defaultdict(list)

    for aid, feat in features.items():
        if not feat.surname:
            continue

        # Key 1: exact surname
        key_s = f"S:{feat.surname}"
        index[key_s].append(aid)

        # Key 2: phonetic surname (only if code is meaningful)
        if feat.phonetic_surname and feat.phonetic_surname != "0000":
            key_p = f"P:{feat.phonetic_surname}"
            index[key_p].append(aid)

        # Key 3: initials + surname (catches dotted-initial variants)
        if feat.initials_key and feat.surname:
            key_is = f"IS:{feat.initials_key}:{feat.surname}"
            index[key_is].append(aid)

    return dict(index)


def generate_candidate_pairs(
    features: dict[int, NameFeatures],
    max_block_size: int = 200,
) -> set[tuple[int, int]]:
    """
    Return deduplicated candidate pairs (id_a, id_b) where id_a < id_b.

    Parameters
    ----------
    features       : mapping from appearance_id to NameFeatures
    max_block_size : blocks larger than this are skipped (generic names like
                     "singh" would generate too many spurious pairs)
    """
    index = _build_block_index(features)
    pairs: set[tuple[int, int]] = set()

    for key, members in index.items():
        if len(members) < 2:
            continue
        if len(members) > max_block_size:
            # Very common surname block — apply tighter filtering inside matching
            # but still include pairs where initials also match
            # We skip the pure-surname and phonetic blocks but keep IS: blocks
            if not key.startswith("IS:"):
                continue

        for a, b in combinations(members, 2):
            pairs.add((min(a, b), max(a, b)))

    return pairs


def build_same_case_index(df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Build mapping: case_id → set of normalised names that appear in it.

    Used to enforce the co-counsel constraint: two normalised names that
    appear in the same case cannot be the same identity.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        index[str(row["case_id"])].add(row["norm_name"])
    return dict(index)


def build_same_case_pair_set(df: pd.DataFrame) -> set[tuple[int, int]]:
    """
    Return set of (id_a, id_b) pairs where both IDs appeared in the same
    case_id.  These pairs are FORBIDDEN from merging.
    """
    case_groups: dict[str, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        case_groups[str(row["case_id"])].append(int(row["appearance_id"]))

    forbidden: set[tuple[int, int]] = set()
    for members in case_groups.values():
        for a, b in combinations(members, 2):
            forbidden.add((min(a, b), max(a, b)))

    return forbidden
