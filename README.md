# Indian Supreme Court Lawyer Entity Resolution Pipeline

A conservative, heuristic-based entity resolution pipeline for
deduplicating Indian Supreme Court lawyer appearance records.

---

## Overview

The pipeline takes three input files:

| File | Description |
|------|-------------|
| `appearances.csv` | Raw appearance records with noisy names and designations |
| `aor_master.csv` | Official (incomplete) AOR registry |
| `sr_advocate_master.csv` | Official (incomplete) Senior Advocate registry |

And produces four output files:

| File | Description |
|------|-------------|
| `resolved_appearances.csv` | Main output: every appearance with cluster, canonical name, designation, confidence |
| `cluster_summary.csv` | One row per cluster with statistics and all raw name variants |
| `match_pairs.csv` | All scored candidate pairs (full audit trail) |
| `suspicious_merges.csv` | Clusters with ≥ 6 distinct raw name variants (review flag) |

---

## Architecture

```
src/
├── normalize.py    Step 1  — Name & designation normalisation
├── features.py     Step 2  — Feature extraction (tokens, initials, Soundex)
├── blocking.py     Step 3  — Candidate pair generation (3 blocking keys)
├── matching.py     Steps 4–6 — Similarity scoring & merge decision
├── clustering.py   Steps 7–8 — Graph clustering & canonical name selection
├── designation.py  Step 9  — Designation inference
└── pipeline.py     Step 10 — Orchestration & output
```

### Step-by-step description

**Step 1 — Normalisation** (`normalize.py`)
- Lowercase, remove punctuation, strip titles (Adv., Sr., AOR, etc.)
- `"Sr. Adv. Rajiv Kumar"` → `"rajiv kumar"`
- `"P.V. Shetty"` → `"p v shetty"`

**Step 2 — Feature Extraction** (`features.py`)
- Tokens, surname (last full token), initials (single-char tokens)
- Initials key (`"p v shetty"` → `"pvs"`)
- Soundex phonetic code of surname (built-in; no external library)

**Step 3 — Blocking** (`blocking.py`)
- Three blocking keys to avoid O(n²) comparisons:
  1. Exact surname → catches obvious same-surname groups
  2. Soundex surname → catches spelling variants
  3. Initials-key + surname → catches `"P V Shetty"` vs `"P.V. Shetty"`
- Blocks with > 200 members (very common surnames via exact/phonetic
  keys) are skipped; only the tight `initials+surname` key is used.
- The same-case constraint is precomputed: two IDs that appear in the
  same `case_id` are co-counsel and can never be merged.

**Step 4-6 — Similarity Scoring** (`matching.py`)

| Component | Weight | Notes |
|-----------|--------|-------|
| `token_set_ratio` | 0.50 | Max of Jaccard and sorted-string SequenceMatcher |
| `surname_sim` | 0.18 | Character-level edit similarity of surnames |
| `given_name_sim` | 0.17 | Non-surname token comparison |
| `initials_compat` | 0.10 | Whether initials are consistent with full tokens |
| `phonetic_match` | +0.15 bonus | Soundex codes agree |
| `common_surname_penalty` | −0.10 to −0.20 | Applied when surname is a very common Indian name (Singh, Kumar, Sharma …) and given names diverge |

Hard-reject rules (score → 0 immediately):
- Same `case_id` (co-counsel)
- Surname similarity < 0.62
- Hard initial conflict (initial in one name contradicts full token in other)
- Both have multi-token names and given names are < 25% similar

Merge thresholds:
- Default: **0.78**
- Common surnames (Singh, Kumar, Sharma, …): **0.90**

**Step 7-8 — Clustering** (`clustering.py`)
- NetworkX connected-components on the match graph
- Canonical name priority: registry spelling → longest complete name → most frequent

**Step 9 — Designation Inference** (`designation.py`)

Priority:
1. SR master match → **SENIOR** (confidence 1.00)
2. AOR master match → **AOR** (confidence 1.00)
3. Majority across cluster appearances (with temporal recency weighting)
4. A lone `"Sr. Adv."` label in a multi-record cluster is treated as noise

Confidence scale:
- 1.00 — registry confirmed
- 0.90 — ≥ 3 unanimous appearances
- 0.75 — clear majority (≥ 75%)
- 0.60 — simple majority or single appearance
- 0.50 — conflicting signals

---

## Usage

```bash
# From the src/ directory
python pipeline.py \
    --appearances ../appearances.csv \
    --aor_master  ../aor_master.csv \
    --sr_master   ../sr_advocate_master.csv \
    --out_dir     ../output
```

Or use defaults (files in the current directory):
```bash
python pipeline.py
```

---

## Results (on the provided dataset)

| Metric | Value |
|--------|-------|
| Total appearances | 3,590 |
| Unique raw names | 859 |
| Clusters formed | 498 |
| Candidate pairs scored | 70,533 |
| Pairs merged | ~43,800 |
| Suspicious clusters flagged | 21 |
| SENIOR clusters | 75 |
| AOR clusters | 259 |
| ADVOCATE clusters | 164 |

---

## Design Decisions

### Why no ML?
Deterministic heuristics are:
- Explainable (every merge has a reason)
- Reproducible (same input → same output)
- Debuggable (match_pairs.csv shows the full scoring trace)
- No training data required

### Why conservative thresholds?
The assignment specification explicitly prioritises **high precision**
over recall. A lawyer wrongly split into two clusters is a minor
inconvenience; a lawyer wrongly merged with another creates systemic
errors downstream.

### Common-surname handling
Indian surnames like "Singh", "Kumar", "Sharma" appear hundreds of times.
Blocking on surname alone would produce massive false-merge groups.
The pipeline applies a stricter merge threshold (0.90 vs 0.78) and a
score penalty when the only shared evidence is a common surname.

### Registry as positive-only evidence
Absence from the AOR or SR registries does not penalise a name — the
registries are explicitly described as incomplete. Only confirmed matches
provide a boost.

---

## Dependencies

```
pandas >= 1.5
networkx >= 2.8
```

No external fuzzy-matching or phonetics libraries are required — Soundex
and token similarity are implemented in pure Python.
