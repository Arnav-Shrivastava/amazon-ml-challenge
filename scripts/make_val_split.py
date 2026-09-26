"""
make_val_split.py — Phase 1: Create reproducible train/val split.

Splits at the Source-1-entity level so no S1 entity's ground truth leaks
between train and val. Stratifies by match-count bucket to preserve the
singleton / 1-match / 2+-match ratio.

Outputs (in dataset/splits/):
    val_source1_ids.txt    — S1 IDs held out for validation
    train_source1_ids.txt  — S1 IDs kept for training

Run from project root:
    python scripts/make_val_split.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
TRAIN_DIR   = ROOT / "dataset" / "train"
SPLITS_DIR  = ROOT / "dataset" / "splits"

VAL_FRACTION = 0.20
RANDOM_SEED  = 42

GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"
S1_FILE = TRAIN_DIR / "train_source1.tsv"


def match_count_bucket(n: int) -> int:
    """Map match count to stratification bucket: 0=singleton, 1=one, 2=two+."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return 2


def make_split() -> None:
    print("Loading data for val split...")

    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, usecols=["entity_id"],
                     keep_default_na=False)
    gt = pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False)

    all_s1_ids = set(s1["entity_id"].tolist())

    # Build per-S1 match count
    gt_counts: dict[str, int] = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_counts[s1_id] = len(matched)

    # Build DataFrame with bucket labels
    records = []
    for s1_id in all_s1_ids:
        cnt = gt_counts.get(s1_id, 0)
        records.append({"entity_id": s1_id, "match_count": cnt,
                        "bucket": match_count_bucket(cnt)})

    df = pd.DataFrame(records)

    print(f"  Total S1 entities : {len(df):,}")
    print(f"  Bucket distribution:")
    for b, label in [(0, "singleton"), (1, "1 match"), (2, "2+ matches")]:
        cnt = (df["bucket"] == b).sum()
        print(f"    {label:<12}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                 random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(df, df["bucket"]))

    train_ids = df.iloc[train_idx]["entity_id"].tolist()
    val_ids   = df.iloc[val_idx]["entity_id"].tolist()

    print(f"\n  Train entities    : {len(train_ids):,}")
    print(f"  Val   entities    : {len(val_ids):,}")

    # Verify no leakage
    assert len(set(train_ids) & set(val_ids)) == 0, "LEAKAGE: train/val overlap!"

    # Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    (SPLITS_DIR / "train_source1_ids.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (SPLITS_DIR / "val_source1_ids.txt").write_text("\n".join(val_ids), encoding="utf-8")

    # Also save val ground truth for evaluation
    val_id_set = set(val_ids)
    gt_val = gt[gt["source1_entity_id"].isin(val_id_set)]
    gt_train = gt[~gt["source1_entity_id"].isin(val_id_set)]

    gt_val.to_csv(SPLITS_DIR / "val_ground_truth.tsv", sep="\t", index=False)
    gt_train.to_csv(SPLITS_DIR / "train_ground_truth.tsv", sep="\t", index=False)

    print(f"\n  Saved splits to {SPLITS_DIR}/")
    print(f"    train_source1_ids.txt  ({len(train_ids):,} IDs)")
    print(f"    val_source1_ids.txt    ({len(val_ids):,} IDs)")
    print(f"    train_ground_truth.tsv ({len(gt_train):,} matched rows)")
    print(f"    val_ground_truth.tsv   ({len(gt_val):,} matched rows)")
    print("\n  Val split ready. No ground truth leaked between train/val.")


if __name__ == "__main__":
    make_split()
