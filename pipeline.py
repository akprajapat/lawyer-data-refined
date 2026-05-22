"""
pipeline.py
===========
Main orchestration pipeline for Indian Supreme Court lawyer entity resolution.

Steps executed in order
-----------------------
  1.  Load data
  2.  Normalize names
  3.  Extract features
  4.  Build same-case forbidden pairs
  5.  Generate candidate pairs (blocking)
  6.  Score all pairs
  7.  Build match graph
  8.  Cluster (connected components)
  9.  Select canonical names
  10. Infer designations
  11. Attach per-appearance confidence scores
  12. Write outputs

Usage
-----
  python pipeline.py \
      --appearances appearances.csv \
      --aor_master  aor_master.csv \
      --sr_master   sr_advocate_master.csv \
      --out_dir     output/

Or simply:
  python pipeline.py        (uses default paths from current directory)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

# ── Internal modules ──────────────────────────────────────────────────────────
from normalize import normalize_name, normalize_designation
from features import extract_features, NameFeatures
from blocking import generate_candidate_pairs, build_same_case_pair_set
from matching import score_all_pairs, ScoreBreakdown, MERGE_THRESHOLD
from clustering import (
    build_match_graph,
    get_clusters,
    build_registry_entries,
    select_canonical_name,
    build_cluster_summary,
)
from designation import infer_all_designations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Step helpers ──────────────────────────────────────────────────────────────

def load_data(
    appearances_path: str,
    aor_path: str,
    sr_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading data …")
    appearances = pd.read_csv(appearances_path, dtype=str)
    appearances["appearance_id"] = appearances["appearance_id"].astype(int)
    aor = pd.read_csv(aor_path, dtype=str)
    sr = pd.read_csv(sr_path, dtype=str)
    log.info(
        "  appearances=%d  |  AOR registry=%d  |  SR registry=%d",
        len(appearances), len(aor), len(sr),
    )
    return appearances, aor, sr


def normalise_and_extract(df: pd.DataFrame) -> pd.DataFrame:
    """Add norm_name column and extract NameFeatures objects."""
    log.info("Normalising names …")
    df = df.copy()
    df["norm_name"] = df["raw_name"].apply(normalize_name)

    log.info("Extracting features …")
    # Build features dict keyed by appearance_id
    return df


def build_features_dict(df: pd.DataFrame) -> dict[int, NameFeatures]:
    features: dict[int, NameFeatures] = {}
    for _, row in df.iterrows():
        aid = int(row["appearance_id"])
        features[aid] = extract_features(
            str(row["raw_name"]), str(row["norm_name"])
        )
    return features




def compute_appearance_confidence(
    bd_by_id: dict[int, list[ScoreBreakdown]],
    id_to_cluster: dict[int, int],
    designation_confidences: dict[int, float],
) -> dict[int, float]:
    """
    Per-appearance confidence = average of:
      - cluster-level designation confidence
      - mean edge weight (similarity score) for edges involving this node

    Ranges from 0 to 1.
    """
    appear_conf: dict[int, float] = {}

    for aid, cid in id_to_cluster.items():
        desig_conf = designation_confidences.get(cid, 0.50)
        edges = bd_by_id.get(aid, [])
        if edges:
            edge_scores = [
                bd.final_score for bd in edges if bd.merged
            ]
            edge_conf = sum(edge_scores) / len(edge_scores) if edge_scores else desig_conf
        else:
            edge_conf = desig_conf
        # Blend
        appear_conf[aid] = round((desig_conf + edge_conf) / 2, 3)

    return appear_conf


def detect_suspicious_merges(
    cluster_summary: pd.DataFrame,
    threshold_size: int = 8,
) -> pd.DataFrame:
    """
    Flag clusters with suspiciously many distinct raw variants — possible
    over-merges.
    """
    suspicious = cluster_summary[
        cluster_summary["unique_raw_names"] >= threshold_size
    ].copy()
    return suspicious


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    appearances_path: str = "appearances.csv",
    aor_path: str = "aor_master.csv",
    sr_path: str = "sr_advocate_master.csv",
    out_dir: str = "output",
) -> pd.DataFrame:
    """
    Execute the full entity-resolution pipeline.

    Returns the final output DataFrame.
    """
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────
    df, aor_df, sr_df = load_data(appearances_path, aor_path, sr_path)

    # ── 2. Normalise ──────────────────────────────────────────────────────
    df = normalise_and_extract(df)

    # ── 3. Feature extraction ─────────────────────────────────────────────
    log.info("Building feature index …")
    features = build_features_dict(df)

    # ── 4. Same-case forbidden pairs ──────────────────────────────────────
    log.info("Computing same-case constraints …")
    forbidden_pairs = build_same_case_pair_set(df)
    log.info("  Forbidden (co-counsel) pairs: %d", len(forbidden_pairs))

    # ── 5. Blocking ───────────────────────────────────────────────────────
    log.info("Generating candidate pairs via blocking …")
    candidate_pairs = generate_candidate_pairs(features, max_block_size=200)
    log.info("  Candidate pairs: %d", len(candidate_pairs))

    # ── 6. Similarity scoring ─────────────────────────────────────────────
    log.info("Scoring candidate pairs …")
    score_results = score_all_pairs(candidate_pairs, features, forbidden_pairs)
    merged_count = sum(1 for bd in score_results if bd.merged)
    log.info(
        "  Scored: %d  |  Merged: %d  |  Threshold: %.2f",
        len(score_results), merged_count, MERGE_THRESHOLD,
    )

    # ── 7. Graph clustering ───────────────────────────────────────────────
    log.info("Building match graph and clustering …")
    G = build_match_graph(score_results)
    all_ids = df["appearance_id"].tolist()
    id_to_cluster = get_clusters(G, all_ids)
    n_clusters = len(set(id_to_cluster.values()))
    log.info("  Clusters: %d  (from %d appearances)", n_clusters, len(df))

    # ── 8. Registry lookup ────────────────────────────────────────────────
    log.info("Building registry lookup …")
    registry_entries = build_registry_entries(aor_df, sr_df)

    # ── 9. Canonical name selection ───────────────────────────────────────
    log.info("Selecting canonical names …")
    df_copy = df.copy()
    df_copy["cluster_id"] = df_copy["appearance_id"].map(id_to_cluster)

    canonical_names: dict[int, str] = {}
    selection_reasons: dict[int, str] = {}

    for cid, group in df_copy.groupby("cluster_id"):
        norm_names = group["norm_name"].tolist()
        raw_names = group["raw_name"].tolist()
        canon, reason = select_canonical_name(norm_names, raw_names, registry_entries)
        canonical_names[cid] = canon
        selection_reasons[cid] = reason

    # ── 10. Designation inference ─────────────────────────────────────────
    log.info("Inferring designations …")
    designations, desig_confidences, desig_reasons = infer_all_designations(
        df_copy, id_to_cluster, canonical_names, aor_df, sr_df
    )

    # ── 11. Per-appearance confidence ─────────────────────────────────────
    log.info("Computing per-appearance confidence …")
    # Index score breakdowns by each appearance_id
    bd_by_id: dict[int, list[ScoreBreakdown]] = defaultdict(list)
    for bd in score_results:
        bd_by_id[bd.id_a].append(bd)
        bd_by_id[bd.id_b].append(bd)

    appear_conf = compute_appearance_confidence(
        bd_by_id, id_to_cluster, desig_confidences
    )

    # ── 12. Assemble final output ─────────────────────────────────────────
    log.info("Assembling output …")
    output_rows = []
    for _, row in df_copy.iterrows():
        aid = int(row["appearance_id"])
        cid = id_to_cluster[aid]
        output_rows.append(
            {
                "appearance_id": aid,
                "cluster_id": cid,
                "canonical_name": canonical_names.get(cid, row["raw_name"]),
                "inferred_designation": designations.get(cid, "ADVOCATE"),
                "confidence_score": appear_conf.get(aid, 0.50),
                # ── Diagnostic columns (helpful for review) ──────────────
                "raw_name": row["raw_name"],
                "norm_name": row["norm_name"],
                "designation_raw": row["designation_raw"],
                "case_id": row["case_id"],
                "hearing_date": row["hearing_date"],
                "appearing_for": row["appearing_for"],
                "desig_inference_reason": desig_reasons.get(cid, ""),
                "name_selection_reason": selection_reasons.get(cid, ""),
            }
        )

    output_df = pd.DataFrame(output_rows).sort_values(
        ["cluster_id", "appearance_id"]
    )

    # ── 13. Build summary and suspicious-merge report ─────────────────────
    cluster_summary = build_cluster_summary(
        df_copy,
        id_to_cluster,
        canonical_names,
        selection_reasons,
        designations,
    )

    suspicious = detect_suspicious_merges(cluster_summary, threshold_size=6)

    # ── 14. Write outputs ─────────────────────────────────────────────────
    main_out = out / "resolved_appearances.csv"
    summary_out = out / "cluster_summary.csv"
    suspicious_out = out / "suspicious_merges.csv"
    match_out = out / "match_pairs.csv"

    output_df.to_csv(main_out, index=False)
    log.info("  Wrote: %s", main_out)

    cluster_summary.to_csv(summary_out, index=False)
    log.info("  Wrote: %s", summary_out)

    if not suspicious.empty:
        suspicious.to_csv(suspicious_out, index=False)
        log.info(
            "  Suspicious merges: %d clusters → %s", len(suspicious), suspicious_out
        )
    else:
        log.info("  No suspicious merges detected.")

    # Write scored pairs for auditability
    pair_records = [
        {
            "id_a": bd.id_a,
            "id_b": bd.id_b,
            "norm_a": bd.norm_a,
            "norm_b": bd.norm_b,
            "final_score": round(bd.final_score, 4),
            "merged": bd.merged,
            "same_case_conflict": bd.same_case_conflict,
            "token_set_ratio": round(bd.token_set_ratio, 4),
            "surname_sim": round(bd.surname_sim, 4),
            "initials_compat": round(bd.initials_compat, 4),
            "phonetic_match": round(bd.phonetic_match, 4),
            "explanation": bd.explain(),
        }
        for bd in score_results
    ]
    pd.DataFrame(pair_records).sort_values(
        "final_score", ascending=False
    ).to_csv(match_out, index=False)
    log.info("  Wrote: %s", match_out)

    # ── Statistics ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("─" * 60)
    log.info("Pipeline complete in %.1f s", elapsed)
    log.info("  Appearances         : %d", len(df))
    log.info("  Unique raw names    : %d", df['raw_name'].nunique())
    log.info("  Clusters formed     : %d", n_clusters)
    log.info("  Candidate pairs     : %d", len(candidate_pairs))
    log.info("  Merged pairs        : %d", merged_count)
    log.info("  Suspicious clusters : %d", len(suspicious))

    desig_breakdown = output_df.drop_duplicates("cluster_id")["inferred_designation"].value_counts()
    log.info("  Designation breakdown (per cluster):")
    for desig, cnt in desig_breakdown.items():
        log.info("    %-10s : %d", desig, cnt)
    log.info("─" * 60)

    return output_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lawyer entity resolution pipeline for Supreme Court appearances"
    )
    parser.add_argument(
        "--appearances", default="appearances.csv",
        help="Path to appearances.csv"
    )
    parser.add_argument(
        "--aor_master", default="aor_master.csv",
        help="Path to aor_master.csv"
    )
    parser.add_argument(
        "--sr_master", default="sr_advocate_master.csv",
        help="Path to sr_advocate_master.csv"
    )
    parser.add_argument(
        "--out_dir", default="output",
        help="Directory for output files"
    )

    args = parser.parse_args()

    run_pipeline(
        appearances_path=args.appearances,
        aor_path=args.aor_master,
        sr_path=args.sr_master,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()