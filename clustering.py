"""
clustering.py
=============
Steps 7–8 — Graph Clustering and Canonical Name Selection

Canonical name selection priority:
  1. Registry spelling — fuzzy-matched against AOR or SR master
  2. Longest complete name (most full tokens, fewest initials)
  3. Most frequent raw name in the cluster

Registry matching is fuzzy (token-set similarity ≥ 0.80) so a cluster
containing only "Yogeshwaran" still gets matched to the registry entry
"P. V. Yogeshwaran" as long as the surname and enough tokens overlap.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

from matching import ScoreBreakdown


# ── Graph construction ────────────────────────────────────────────────────────

def build_match_graph(score_results: list[ScoreBreakdown]) -> nx.Graph:
    G = nx.Graph()
    for bd in score_results:
        G.add_node(bd.id_a)
        G.add_node(bd.id_b)
        if bd.merged:
            G.add_edge(bd.id_a, bd.id_b, weight=bd.final_score)
    return G


def get_clusters(G: nx.Graph, all_ids: list[int]) -> dict[int, int]:
    for aid in all_ids:
        if aid not in G:
            G.add_node(aid)
    id_to_cluster: dict[int, int] = {}
    components = sorted(nx.connected_components(G), key=lambda c: min(c))
    for cluster_id, component in enumerate(components, start=1):
        for aid in component:
            id_to_cluster[aid] = cluster_id
    return id_to_cluster


# ── Registry lookup ───────────────────────────────────────────────────────────

def _token_set_sim(a: str, b: str) -> float:
    """Jaccard similarity on word-token sets."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_registry_entries(
    aor_df: pd.DataFrame,
    sr_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """
    Return list of (normalised_name, raw_registry_name) for every entry
    in both registries.  Used for fuzzy lookup against cluster members.
    """
    from normalize import normalize_registry_name

    entries: list[tuple[str, str]] = []
    for _, row in aor_df.iterrows():
        raw = str(row.get("full_name", "")).strip()
        norm = normalize_registry_name(raw)
        if norm:
            entries.append((norm, raw))
    for _, row in sr_df.iterrows():
        raw = str(row.get("full_name", "")).strip()
        norm = normalize_registry_name(raw)
        if norm:
            entries.append((norm, raw))
    return entries


def find_registry_match(
    cluster_norm_names: list[str],
    registry_entries: list[tuple[str, str]],
    sim_threshold: float = 0.80,
) -> tuple[str, float] | tuple[None, float]:
    """
    Search registry entries for the best fuzzy match to any cluster member.

    Matching strategy (in priority order):
      1. Exact normalised match         → score 1.0
      2. Surname exact + token-set ≥ threshold → score = token_set_sim
      3. Best token-set match ≥ threshold

    Returns (registry_raw_name, score) or (None, 0.0) if no match found.
    """
    # Build surname set for quick filtering
    cluster_surnames: set[str] = set()
    for nm in cluster_norm_names:
        tokens = nm.split()
        full = [t for t in tokens if len(t) > 1]
        if full:
            cluster_surnames.add(full[-1])

    best_raw: str | None = None
    best_score: float = 0.0

    for reg_norm, reg_raw in registry_entries:
        # --- Priority 1: exact ---
        if reg_norm in cluster_norm_names:
            return reg_raw, 1.0

        # --- Quick surname filter ---
        reg_tokens = reg_norm.split()
        reg_full = [t for t in reg_tokens if len(t) > 1]
        if not reg_full:
            continue
        reg_surname = reg_full[-1]
        if reg_surname not in cluster_surnames:
            continue   # different surname — skip

        # --- Fuzzy match against each cluster member ---
        for nm in cluster_norm_names:
            sim = _token_set_sim(nm, reg_norm)
            if sim > best_score:
                best_score = sim
                best_raw = reg_raw

    if best_score >= sim_threshold and best_raw:
        return best_raw, best_score

    return None, 0.0


# ── Canonical name selection ──────────────────────────────────────────────────

def _split_initial_penalty(raw_name: str) -> int:
    """
    Count "split-initial" transcription artifacts in a raw name.

    A split-initial occurs when a single-character token (an initial)
    appears immediately before a multi-character token starting with the
    SAME letter — e.g. "P. v. Vishwanatha Shetty" where the initial 'v'
    sits right before 'Vishwanatha'.  This pattern means the recorder
    wrote the middle name twice: once abbreviated and once expanded.

    Lower penalty = cleaner name.  0 = no artifact detected.
    """
    tokens = raw_name.lower().replace(".", " ").split()
    penalty = 0
    for i in range(len(tokens) - 1):
        if len(tokens[i]) == 1 and tokens[i + 1].startswith(tokens[i]):
            penalty += 1
    return penalty


def select_canonical_name(
    cluster_norm_names: list[str],
    cluster_raw_names: list[str],
    registry_entries: list[tuple[str, str]],
) -> tuple[str, str]:
    """
    Choose the best canonical name for a cluster.

    Returns (canonical_name, selection_reason).

    Priority
    --------
    1. Registry match (fuzzy) — use official registry spelling
    2. Name scored by: (full_token_count, -split_initial_penalty, frequency)
         - more full tokens wins
         - among ties: fewer transcription artifacts wins
         - final tiebreaker: most frequent
    """
    # Priority 1: registry fuzzy match
    reg_name, reg_score = find_registry_match(cluster_norm_names, registry_entries)
    if reg_name:
        return reg_name, f"registry(sim={reg_score:.2f})"

    if not cluster_raw_names:
        return "", "none"

    freq = Counter(cluster_raw_names)

    import re as _re

    def score(name: str) -> tuple[int, int, int]:
        # Strip punctuation before counting — "P." (len=2) must not count as
        # a full token; only bare alphabetic tokens longer than 1 char qualify.
        clean_tokens = _re.sub(r"[^\w\s]", " ", name.lower()).split()
        n_full = len([t for t in clean_tokens if len(t) > 1])
        penalty = _split_initial_penalty(name)
        frequency = freq[name]
        return (n_full, -penalty, frequency)

    best = max(cluster_raw_names, key=score)
    s = score(best)

    if s[1] < 0:
        reason = "longest(clean)"    # penalty was applied somewhere
    elif s[2] > 1:
        reason = "most_frequent"
    else:
        reason = "longest"

    return best, reason


# ── Cluster summary ───────────────────────────────────────────────────────────

def build_cluster_summary(
    df: pd.DataFrame,
    id_to_cluster: dict[int, int],
    canonical_names: dict[int, str],
    selection_reasons: dict[int, str],
    inferred_designations: dict[int, str],
) -> pd.DataFrame:
    records = []
    cluster_groups: dict[int, list[dict]] = defaultdict(list)

    for _, row in df.iterrows():
        aid = int(row["appearance_id"])
        cid = id_to_cluster.get(aid, -1)
        cluster_groups[cid].append({
            "appearance_id": aid,
            "raw_name": row["raw_name"],
            "norm_name": row["norm_name"],
            "case_id": row["case_id"],
            "hearing_date": row["hearing_date"],
            "designation_raw": row["designation_raw"],
        })

    for cid, members in cluster_groups.items():
        raw_names = [m["raw_name"] for m in members]
        unique_names = list(dict.fromkeys(raw_names))
        records.append({
            "cluster_id": cid,
            "canonical_name": canonical_names.get(cid, ""),
            "inferred_designation": inferred_designations.get(cid, "ADVOCATE"),
            "name_selection_reason": selection_reasons.get(cid, ""),
            "member_count": len(members),
            "unique_raw_names": len(set(raw_names)),
            "unique_cases": len({m["case_id"] for m in members}),
            "all_raw_variants": " | ".join(sorted(set(unique_names))),
        })

    return pd.DataFrame(records).sort_values("cluster_id")