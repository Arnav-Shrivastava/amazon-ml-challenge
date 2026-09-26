"""
data_audit.py — Phase 0 data audit for Amazon ML Challenge 2026.

Produces a text report covering:
  - Row counts per source
  - Null rates per field
  - Country distribution
  - Name / address length distributions
  - Raw noisy example pairs pulled from the data
  - Ground truth statistics (singleton fraction, match-count distribution)

Run from project root:
    python scripts/data_audit.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"

FILES = {
    "source1": TRAIN / "train_source1.tsv",
    "source2": TRAIN / "train_source2.tsv",
    "source3": TRAIN / "train_source3.tsv",
    "gt":      TRAIN / "train_ground_truth.tsv",
}

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS     = ["source1_entity_id", "matched_entity_ids"]

SEP = "\t"


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_source(path: Path, expected_cols: list[str], label: str) -> pd.DataFrame:
    """Load a TSV, verify separator was applied, return DataFrame."""
    df = pd.read_csv(path, sep=SEP, dtype=str, keep_default_na=False, low_memory=False)

    # Single-column bug check: if sep wasn't applied, all data lands in col 0
    if df.shape[1] == 1:
        print(f"  [ERROR] {label}: looks like sep='\\t' wasn't applied — "
              f"only 1 column detected. Raw first row: {df.iloc[0,0][:120]!r}")
        sys.exit(1)

    if list(df.columns) != expected_cols:
        print(f"  [WARN]  {label}: unexpected columns: {list(df.columns)}")
        print(f"          Expected: {expected_cols}")

    return df


def null_rate(df: pd.DataFrame, col: str) -> float:
    """Fraction of rows where col is empty string or whitespace-only."""
    return (df[col].str.strip() == "").mean()


def length_stats(series: pd.Series) -> dict:
    lengths = series.str.len()
    return {
        "min":    int(lengths.min()),
        "p25":    int(lengths.quantile(0.25)),
        "median": int(lengths.median()),
        "p75":    int(lengths.quantile(0.75)),
        "max":    int(lengths.max()),
        "mean":   round(float(lengths.mean()), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_audit() -> None:
    banner("LOADING DATA")

    print("  Loading train_source1.tsv ...")
    s1 = load_source(FILES["source1"], SOURCE_COLS, "source1")
    print(f"    → {len(s1):,} rows, {s1.shape[1]} columns: {list(s1.columns)}")

    print("  Loading train_source2.tsv ...")
    s2 = load_source(FILES["source2"], SOURCE_COLS, "source2")
    print(f"    → {len(s2):,} rows, {s2.shape[1]} columns: {list(s2.columns)}")

    print("  Loading train_source3.tsv ...")
    s3 = load_source(FILES["source3"], SOURCE_COLS, "source3")
    print(f"    → {len(s3):,} rows, {s3.shape[1]} columns: {list(s3.columns)}")

    print("  Loading train_ground_truth.tsv ...")
    gt = load_source(FILES["gt"], GT_COLS, "ground_truth")
    print(f"    → {len(gt):,} rows, {gt.shape[1]} columns: {list(gt.columns)}")

    # ── Row counts ────────────────────────────────────────────────────────────
    banner("ROW COUNTS")
    for label, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3), ("Ground Truth", gt)]:
        print(f"  {label:<15}: {len(df):>10,} rows")

    # ── Null rates ────────────────────────────────────────────────────────────
    banner("NULL RATES (empty or whitespace-only)")
    for label, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        print(f"\n  {label}:")
        for col in ["entity_id", "business_name", "business_address", "country"]:
            rate = null_rate(df, col)
            print(f"    {col:<25}: {rate*100:6.2f}%  ({int(rate * len(df)):,} rows)")

    # ── Country distribution ──────────────────────────────────────────────────
    banner("COUNTRY DISTRIBUTION")
    for label, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        counts = df["country"].value_counts(dropna=False).head(15)
        print(f"\n  {label}:")
        for country, cnt in counts.items():
            print(f"    {str(country):<30}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

    # ── Name / address length distributions ───────────────────────────────────
    banner("NAME LENGTH DISTRIBUTION (non-empty names)")
    for label, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        non_empty = df.loc[df["business_name"].str.strip() != "", "business_name"]
        stats = length_stats(non_empty)
        print(f"  {label}: n={len(non_empty):,}  min={stats['min']}  "
              f"p25={stats['p25']}  median={stats['median']}  "
              f"p75={stats['p75']}  max={stats['max']}  mean={stats['mean']}")

    banner("ADDRESS LENGTH DISTRIBUTION (non-empty addresses)")
    for label, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        non_empty = df.loc[df["business_address"].str.strip() != "", "business_address"]
        stats = length_stats(non_empty)
        print(f"  {label}: n={len(non_empty):,}  min={stats['min']}  "
              f"p25={stats['p25']}  median={stats['median']}  "
              f"p75={stats['p75']}  max={stats['max']}  mean={stats['mean']}")

    # ── Raw noisy examples ────────────────────────────────────────────────────
    banner("RAW NOISY EXAMPLES (S1 vs matched S2/S3)")

    # Parse GT to get a sample of matched pairs
    gt_parsed = gt[gt["matched_entity_ids"].str.strip() != ""].copy()
    gt_parsed["matched_list"] = gt_parsed["matched_entity_ids"].str.split(",")

    # Build lookup maps
    s1_map = s1.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s2_map = s2.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s3_map = s3.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    src_map = {**s2_map, **s3_map}

    print("\n  Showing up to 15 real matched entity pairs from the training data:\n")
    shown = 0
    for _, row in gt_parsed.sample(min(200, len(gt_parsed)), random_state=99).iterrows():
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_map:
            continue
        s1_rec = s1_map[s1_id]
        for sx_id in row["matched_list"]:
            sx_id = sx_id.strip()
            if sx_id not in src_map:
                continue
            sx_rec = src_map[sx_id]
            print(f"  Pair: {s1_id} ↔ {sx_id}")
            print(f"    S1 name   : {s1_rec['business_name']!r}")
            print(f"    Sx name   : {sx_rec['business_name']!r}")
            print(f"    S1 address: {s1_rec['business_address']!r}")
            print(f"    Sx address: {sx_rec['business_address']!r}")
            print(f"    S1 country: {s1_rec['country']!r}  |  Sx country: {sx_rec['country']!r}")
            print()
            shown += 1
            if shown >= 15:
                break
        if shown >= 15:
            break

    # ── Ground truth statistics ───────────────────────────────────────────────
    banner("GROUND TRUTH STATISTICS")

    # GT has only MATCHED rows; singletons are s1 entities NOT in GT
    gt_s1_ids = set(gt["source1_entity_id"])
    all_s1_ids = set(s1["entity_id"])
    singletons = all_s1_ids - gt_s1_ids
    matched_entities = all_s1_ids & gt_s1_ids

    print(f"\n  Total S1 entities     : {len(all_s1_ids):,}")
    print(f"  Singletons (no match) : {len(singletons):,}  ({len(singletons)/len(all_s1_ids)*100:.1f}%)")
    print(f"  Have ≥1 match         : {len(matched_entities):,}  ({len(matched_entities)/len(all_s1_ids)*100:.1f}%)")

    # Match-count distribution
    gt_matched = gt[gt["source1_entity_id"].isin(matched_entities)].copy()
    gt_matched["match_count"] = gt_matched["matched_entity_ids"].apply(
        lambda x: len([i for i in x.split(",") if i.strip()])
    )

    print("\n  Match-count distribution (among matched S1 entities):")
    for bucket, label in [(1, "exactly 1"), (2, "exactly 2"), (3, "exactly 3"),
                          (range(4, 6), "4–5"), (range(6, 11), "6–10"),
                          (range(11, 10000), "11+")]:
        if isinstance(bucket, int):
            cnt = (gt_matched["match_count"] == bucket).sum()
        else:
            cnt = gt_matched["match_count"].isin(bucket).sum()
        pct = cnt / len(gt_matched) * 100 if len(gt_matched) else 0
        print(f"    {label:<15}: {cnt:>7,}  ({pct:.1f}%)")

    print(f"\n  Mean matches per matched entity  : {gt_matched['match_count'].mean():.2f}")
    print(f"  Median matches per matched entity: {gt_matched['match_count'].median():.1f}")
    print(f"  Max matches per matched entity   : {gt_matched['match_count'].max()}")

    # Total matched IDs breakdown by source
    all_matched_ids = []
    for ids in gt["matched_entity_ids"]:
        all_matched_ids.extend([i.strip() for i in ids.split(",") if i.strip()])
    s2_cnt = sum(1 for i in all_matched_ids if i.startswith("S2-"))
    s3_cnt = sum(1 for i in all_matched_ids if i.startswith("S3-"))
    print(f"\n  Total matched ID references: {len(all_matched_ids):,}")
    print(f"    From Source 2 : {s2_cnt:,}  ({s2_cnt/len(all_matched_ids)*100:.1f}%)")
    print(f"    From Source 3 : {s3_cnt:,}  ({s3_cnt/len(all_matched_ids)*100:.1f}%)")

    banner("AUDIT COMPLETE")
    print("  Review the numbers above before proceeding to Phase 1.\n")


if __name__ == "__main__":
    run_audit()
