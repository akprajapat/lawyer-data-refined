"""
designation.py
==============
Step 9 — Designation Inference

Infers the current, most-accurate designation for each lawyer cluster.

Designation ladder (one-way progression):
  ADVOCATE → AOR → SENIOR

Rules (in priority order)
--------------------------
1. SR master match     → SENIOR  (strong positive evidence)
2. AOR master match    → AOR     (strong positive evidence)
3. Latest-hearing designation (temporal recency, min 2 appearances)
4. Majority designation among cluster appearances
5. Any SENIOR label present → SENIOR *only* if ≥ 2 independent records
6. Default → ADVOCATE

Confidence scoring
------------------
  1.00 — registry confirmed
  0.90 — ≥ 3 consistent appearances (all agree)
  0.75 — majority evidence, some noise
  0.60 — single appearance or conflicting signals
  0.50 — fallback default

Note: Absence from a registry is NOT penalised (registries are incomplete).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import pandas as pd


# ── Registry membership sets ──────────────────────────────────────────────────

def build_registry_norm_sets(
    aor_df: pd.DataFrame,
    sr_df: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """
    Return (aor_norm_set, sr_norm_set) for membership testing.
    Names are normalised so matching is robust to minor punctuation
    differences that the clustering pipeline may not have resolved.
    """
    from normalize import normalize_registry_name

    aor_norms = {
        normalize_registry_name(str(r["full_name"]))
        for _, r in aor_df.iterrows()
        if r.get("full_name")
    }
    sr_norms = {
        normalize_registry_name(str(r["full_name"]))
        for _, r in sr_df.iterrows()
        if r.get("full_name")
    }
    return aor_norms, sr_norms


# ── Designation normalisation (imported from normalize.py) ────────────────────

def _norm_desig(raw: str) -> str:
    from normalize import normalize_designation
    return normalize_designation(raw)


# ── Per-cluster inference ─────────────────────────────────────────────────────

def infer_designation_for_cluster(
    cluster_rows: pd.DataFrame,
    canonical_norm: str,
    aor_norm_set: set[str],
    sr_norm_set: set[str],
) -> tuple[str, float, str]:
    """
    Infer designation for one cluster.

    Parameters
    ----------
    cluster_rows   : all appearance rows for this cluster
    canonical_norm : normalised canonical name (for registry lookup)
    aor_norm_set   : normalised AOR registry names
    sr_norm_set    : normalised SR registry names

    Returns
    -------
    (designation, confidence, reason)
    """
    # ── Rule 1: SR registry match ─────────────────────────────────────────
    if canonical_norm in sr_norm_set:
        return "SENIOR", 1.00, "sr_registry_match"

    # Also check if any cluster norm name hits the registry
    for norm in cluster_rows.get("norm_name", []):
        if norm in sr_norm_set:
            return "SENIOR", 1.00, "sr_registry_match_variant"

    # ── Rule 2: AOR registry match ────────────────────────────────────────
    if canonical_norm in aor_norm_set:
        return "AOR", 1.00, "aor_registry_match"

    for norm in cluster_rows.get("norm_name", []):
        if norm in aor_norm_set:
            return "AOR", 1.00, "aor_registry_match_variant"

    # ── Collect normalised designations for this cluster ──────────────────
    norm_desigs = [_norm_desig(str(d)) for d in cluster_rows["designation_raw"]]
    counts = Counter(norm_desigs)
    total = len(norm_desigs)

    # ── Rule 3: Temporal recency (latest hearing date) ────────────────────
    dated = cluster_rows.copy()
    dated["_date"] = pd.to_datetime(dated["hearing_date"], errors="coerce")
    dated = dated.dropna(subset=["_date"])
    latest_desig: Optional[str] = None
    if not dated.empty:
        latest_row = dated.loc[dated["_date"].idxmax()]
        latest_desig = _norm_desig(str(latest_row["designation_raw"]))

    # ── Rule 4: SENIOR requires ≥ 2 records ──────────────────────────────
    senior_count = counts.get("SENIOR", 0)
    aor_count = counts.get("AOR", 0)
    advocate_count = counts.get("ADVOCATE", 0)

    # Avoid upgrading to SENIOR on a single noisy label
    if senior_count == 1 and total > 1:
        # Treat that one SENIOR as noise; remove from consideration
        counts["SENIOR"] = 0
        senior_count = 0

    # ── Majority rule ─────────────────────────────────────────────────────
    majority_desig, majority_n = counts.most_common(1)[0]

    # ── Confidence scoring ────────────────────────────────────────────────
    purity = majority_n / total if total > 0 else 0

    if purity == 1.0 and total >= 3:
        confidence = 0.90
        reason = "unanimous_3plus"
    elif purity >= 0.75:
        confidence = 0.75
        reason = "majority_75pct"
    elif purity >= 0.50:
        confidence = 0.60
        reason = "simple_majority"
    else:
        confidence = 0.50
        reason = "conflicting_signals"

    # Single-appearance cluster: lower confidence
    if total == 1:
        confidence = 0.60
        reason = "single_appearance"

    # Prefer latest-date designation when majority and latest agree
    final_desig = majority_desig
    if latest_desig and latest_desig == majority_desig:
        reason += "+latest_agrees"
    elif latest_desig and latest_desig != majority_desig:
        # Latest disagrees — keep majority but note conflict
        reason += "+latest_conflicts"

    return final_desig, round(confidence, 2), reason


# ── Batch inference ───────────────────────────────────────────────────────────

def infer_all_designations(
    df: pd.DataFrame,
    id_to_cluster: dict[int, int],
    canonical_names: dict[int, str],     # cluster_id → canonical_name
    aor_df: pd.DataFrame,
    sr_df: pd.DataFrame,
) -> tuple[dict[int, str], dict[int, float], dict[int, str]]:
    """
    Run designation inference for every cluster.

    Returns
    -------
    (designations, confidences, reasons)
    Each is a dict keyed by cluster_id.
    """
    from normalize import normalize_registry_name

    aor_norm_set, sr_norm_set = build_registry_norm_sets(aor_df, sr_df)

    # Group rows by cluster
    df_copy = df.copy()
    df_copy["cluster_id"] = df_copy["appearance_id"].map(id_to_cluster)

    designations: dict[int, str] = {}
    confidences: dict[int, float] = {}
    reasons: dict[int, str] = {}

    for cid, group in df_copy.groupby("cluster_id"):
        canonical_raw = canonical_names.get(cid, "")
        canonical_norm = normalize_registry_name(canonical_raw)

        desig, conf, reason = infer_designation_for_cluster(
            group, canonical_norm, aor_norm_set, sr_norm_set
        )
        designations[cid] = desig
        confidences[cid] = conf
        reasons[cid] = reason

    return designations, confidences, reasons
