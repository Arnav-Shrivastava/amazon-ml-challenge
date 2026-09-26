"""
data_audit.py — Phase 0 data audit for Amazon ML Challenge 2026.

Streams each file in chunks so it completes quickly even on large TSVs.
Produces:
  - Row counts per source
  - Null rates per field
  - Country distribution
  - Name / address length distributions (percentiles via reservoir sampling)
  - 15 real noisy matched-pair examples pulled from the data
  - Ground truth statistics (singleton fraction, match-count distribution)

Run from project root:
    python scripts/data_audit.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"

FILES = {
    "source1": TRAIN / "train_source1.tsv",
    "source2": TRAIN / "train_source2.tsv",
    "source3": TRAIN / "train_source3.tsv",
    "gt":      TRAIN / "train_ground_truth.tsv",
}

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS     = ["source1_entity_id", "matched_entity_ids"]
SEP         = "\t"
CHUNKSIZE   = 100_000
RESERVOIR_N = 50_000   # reservoir size for length-percentile sampling

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def percentile_from_reservoir(samples: list[int], q: float) -> int:
    arr = np.array(samples, dtype=np.int32)
    return int(np.percentile(arr, q))


def scan_source(path: Path, label: str) -> dict:
    """Stream-scan a source TSV, collecting stats without loading all into RAM."""
    print(f"  Scanning {label} ({path.name}) ...")

    row_count     = 0
    null_counts   = Counter()
    country_ctr   = Counter()
    name_res, addr_res = [], []   # reservoir samples for length percentiles
    name_total_len = addr_total_len = 0

    # Check header
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split(SEP)

    if len(header) == 1:
        print(f"  [ERROR] {label}: only 1 column detected — sep='\\t' likely not applied!")
        print(f"          Raw header: {header[0][:120]!r}")
        sys.exit(1)

    print(f"    Columns ({len(header)}): {header}")

    for chunk in pd.read_csv(
        path, sep=SEP, dtype=str, keep_default_na=False,
        chunksize=CHUNKSIZE, encoding="utf-8", on_bad_lines="skip"
    ):
        # Standardise column names
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk_len = len(chunk)
        row_count += chunk_len

        for col in ["business_name", "business_address", "country"]:
            if col in chunk.columns:
                null_counts[col] += (chunk[col].str.strip() == "").sum()

        if "country" in chunk.columns:
            country_ctr.update(chunk["country"].str.strip().tolist())

        # Reservoir sampling for name/address lengths
        if "business_name" in chunk.columns:
            lengths = chunk["business_name"].str.len().tolist()
            name_total_len += sum(lengths)
            for l in lengths:
                if len(name_res) < RESERVOIR_N:
                    name_res.append(l)
                else:
                    j = random.randint(0, row_count)
                    if j < RESERVOIR_N:
                        name_res[j] = l

        if "business_address" in chunk.columns:
            lengths = chunk["business_address"].str.len().tolist()
            addr_total_len += sum(lengths)
            for l in lengths:
                if len(addr_res) < RESERVOIR_N:
                    addr_res.append(l)
                else:
                    j = random.randint(0, row_count)
                    if j < RESERVOIR_N:
                        addr_res[j] = l

    # Compute stats
    name_stats = {
        "p25": percentile_from_reservoir(name_res, 25),
        "median": percentile_from_reservoir(name_res, 50),
        "p75": percentile_from_reservoir(name_res, 75),
        "max": max(name_res) if name_res else 0,
        "mean": round(name_total_len / row_count, 1) if row_count else 0,
    }
    addr_stats = {
        "p25": percentile_from_reservoir(addr_res, 25),
        "median": percentile_from_reservoir(addr_res, 50),
        "p75": percentile_from_reservoir(addr_res, 75),
        "max": max(addr_res) if addr_res else 0,
        "mean": round(addr_total_len / row_count, 1) if row_count else 0,
    }

    return {
        "label": label,
        "row_count": row_count,
        "null_counts": dict(null_counts),
        "country_ctr": country_ctr,
        "name_stats": name_stats,
        "addr_stats": addr_stats,
    }


def scan_gt(path: Path) -> dict:
    """Stream-scan ground truth TSV."""
    print(f"  Scanning ground truth ({path.name}) ...")
    row_count   = 0
    match_counts: list[int] = []
    s1_ids_seen = set()

    for chunk in pd.read_csv(
        path, sep=SEP, dtype=str, keep_default_na=False,
        chunksize=CHUNKSIZE, encoding="utf-8", on_bad_lines="skip"
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        row_count += len(chunk)
        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            ids = [x.strip() for x in matched.split(",") if x.strip()]
            match_counts.append(len(ids))
            s1_ids_seen.add(s1_id)

    return {
        "row_count": row_count,
        "s1_ids_seen": s1_ids_seen,
        "match_counts": match_counts,
    }


def sample_noisy_pairs(path_s1: Path, path_gt: Path, src_paths: dict[str, Path], n: int = 15) -> None:
    """Load a small random sample of GT rows and print real matched-pair examples."""
    print(f"\n  Sampling GT for noisy pair examples...")

    # Read first 5000 GT rows for sampling speed
    gt_sample = pd.read_csv(path_gt, sep=SEP, dtype=str, keep_default_na=False,
                            nrows=5000, encoding="utf-8", on_bad_lines="skip")
    gt_sample.columns = [c.strip() for c in gt_sample.columns]
    gt_sample = gt_sample[gt_sample["matched_entity_ids"].str.strip() != ""]
    gt_sample = gt_sample.sample(min(500, len(gt_sample)), random_state=7)

    # Collect all entity IDs we need to look up
    s1_needed = set(gt_sample["source1_entity_id"])
    sx_needed: set[str] = set()
    pair_map: dict[str, list[str]] = {}
    for _, row in gt_sample.iterrows():
        ids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        pair_map[row["source1_entity_id"]] = ids
        sx_needed.update(ids[:3])  # limit to first 3 matches per entity

    # Load only needed records from S1
    s1_recs: dict[str, dict] = {}
    for chunk in pd.read_csv(path_s1, sep=SEP, dtype=str, keep_default_na=False,
                              chunksize=CHUNKSIZE, encoding="utf-8", on_bad_lines="skip"):
        chunk.columns = [c.strip() for c in chunk.columns]
        hits = chunk[chunk["entity_id"].isin(s1_needed)]
        for _, r in hits.iterrows():
            s1_recs[r["entity_id"]] = r.to_dict()
        if len(s1_recs) >= len(s1_needed):
            break

    # Load needed records from S2/S3
    sx_recs: dict[str, dict] = {}
    for src_path in src_paths.values():
        for chunk in pd.read_csv(src_path, sep=SEP, dtype=str, keep_default_na=False,
                                  chunksize=CHUNKSIZE, encoding="utf-8", on_bad_lines="skip"):
            chunk.columns = [c.strip() for c in chunk.columns]
            hits = chunk[chunk["entity_id"].isin(sx_needed)]
            for _, r in hits.iterrows():
                sx_recs[r["entity_id"]] = r.to_dict()
        if len(sx_recs) >= len(sx_needed):
            break

    # Print examples
    shown = 0
    print(f"\n  {n} real matched-pair examples (S1 entity vs its matched S2/S3 record):\n")
    for s1_id, sx_ids in pair_map.items():
        if s1_id not in s1_recs:
            continue
        s1r = s1_recs[s1_id]
        for sx_id in sx_ids[:2]:
            if sx_id not in sx_recs:
                continue
            sxr = sx_recs[sx_id]
            print(f"  Pair: {s1_id} <-> {sx_id}")
            print(f"    S1 name   : {s1r.get('business_name','')!r}")
            print(f"    Sx name   : {sxr.get('business_name','')!r}")
            print(f"    S1 address: {s1r.get('business_address','')!r}")
            print(f"    Sx address: {sxr.get('business_address','')!r}")
            print(f"    country   : S1={s1r.get('country','')!r}  Sx={sxr.get('country','')!r}")
            print()
            shown += 1
            if shown >= n:
                return


# ── Main ──────────────────────────────────────────────────────────────────────

def run_audit() -> None:
    random.seed(42)

    banner("SCANNING SOURCE FILES (streaming, chunked)")
    s1_stats = scan_source(FILES["source1"], "Source 1")
    s2_stats = scan_source(FILES["source2"], "Source 2")
    s3_stats = scan_source(FILES["source3"], "Source 3")
    gt_stats = scan_gt(FILES["gt"])

    # ── Row counts ────────────────────────────────────────────────────────────
    banner("ROW COUNTS")
    for stats in [s1_stats, s2_stats, s3_stats]:
        print(f"  {stats['label']:<12}: {stats['row_count']:>10,} rows")
    print(f"  Ground Truth: {gt_stats['row_count']:>10,} rows (matched S1 entities only)")

    # ── Null rates ────────────────────────────────────────────────────────────
    banner("NULL / EMPTY RATES PER FIELD")
    for stats in [s1_stats, s2_stats, s3_stats]:
        n = stats["row_count"]
        print(f"\n  {stats['label']}  (n={n:,})")
        for col in ["business_name", "business_address", "country"]:
            cnt = stats["null_counts"].get(col, 0)
            print(f"    {col:<25}: {cnt/n*100:6.2f}%  ({cnt:,} rows)")

    # ── Country distribution ──────────────────────────────────────────────────
    banner("COUNTRY DISTRIBUTION (top 10 per source)")
    for stats in [s1_stats, s2_stats, s3_stats]:
        print(f"\n  {stats['label']}:")
        n = stats["row_count"]
        for country, cnt in stats["country_ctr"].most_common(10):
            print(f"    {str(country):<30}: {cnt:>8,}  ({cnt/n*100:.1f}%)")

    # ── Length distributions ──────────────────────────────────────────────────
    banner("NAME LENGTH DISTRIBUTION (chars)")
    print(f"  {'Source':<12} {'p25':>5} {'median':>7} {'p75':>5} {'max':>6} {'mean':>6}")
    for stats in [s1_stats, s2_stats, s3_stats]:
        s = stats["name_stats"]
        print(f"  {stats['label']:<12} {s['p25']:>5} {s['median']:>7} {s['p75']:>5} {s['max']:>6} {s['mean']:>6}")

    banner("ADDRESS LENGTH DISTRIBUTION (chars)")
    print(f"  {'Source':<12} {'p25':>5} {'median':>7} {'p75':>5} {'max':>6} {'mean':>6}")
    for stats in [s1_stats, s2_stats, s3_stats]:
        s = stats["addr_stats"]
        print(f"  {stats['label']:<12} {s['p25']:>5} {s['median']:>7} {s['p75']:>5} {s['max']:>6} {s['mean']:>6}")

    # ── GT statistics ─────────────────────────────────────────────────────────
    banner("GROUND TRUTH STATISTICS")
    s1_total = s1_stats["row_count"]
    gt_s1_count = len(gt_stats["s1_ids_seen"])
    singleton_count = s1_total - gt_s1_count

    print(f"\n  Total S1 entities        : {s1_total:,}")
    print(f"  S1 with >= 1 match (GT)  : {gt_s1_count:,}  ({gt_s1_count/s1_total*100:.1f}%)")
    print(f"  Singletons (no GT entry) : {singleton_count:,}  ({singleton_count/s1_total*100:.1f}%)")

    mc = gt_stats["match_counts"]
    match_arr = np.array(mc, dtype=np.int32)
    print(f"\n  Match-count distribution (among {len(mc):,} GT rows):")
    for label, mask in [
        ("exactly 1",  match_arr == 1),
        ("exactly 2",  match_arr == 2),
        ("exactly 3",  match_arr == 3),
        ("4-5",        (match_arr >= 4) & (match_arr <= 5)),
        ("6-10",       (match_arr >= 6) & (match_arr <= 10)),
        ("11+",        match_arr >= 11),
    ]:
        cnt = int(mask.sum())
        print(f"    {label:<12}: {cnt:>7,}  ({cnt/len(mc)*100:.1f}%)")

    print(f"\n  Mean matches per GT row  : {match_arr.mean():.2f}")
    print(f"  Median                   : {float(np.median(match_arr)):.1f}")
    print(f"  Max                      : {int(match_arr.max())}")

    all_ids_flat: list[str] = []
    for chunk in pd.read_csv(FILES["gt"], sep=SEP, dtype=str, keep_default_na=False,
                              chunksize=CHUNKSIZE, encoding="utf-8", on_bad_lines="skip"):
        chunk.columns = [c.strip() for c in chunk.columns]
        for ids_str in chunk["matched_entity_ids"]:
            all_ids_flat.extend([x.strip() for x in ids_str.split(",") if x.strip()])
    s2_ref = sum(1 for i in all_ids_flat if i.startswith("S2-"))
    s3_ref = sum(1 for i in all_ids_flat if i.startswith("S3-"))
    print(f"\n  Total matched ID refs     : {len(all_ids_flat):,}")
    print(f"    From Source 2           : {s2_ref:,}  ({s2_ref/len(all_ids_flat)*100:.1f}%)")
    print(f"    From Source 3           : {s3_ref:,}  ({s3_ref/len(all_ids_flat)*100:.1f}%)")

    # ── Noisy pair examples ───────────────────────────────────────────────────
    banner("NOISY MATCHED-PAIR EXAMPLES")
    sample_noisy_pairs(
        FILES["source1"], FILES["gt"],
        {"source2": FILES["source2"], "source3": FILES["source3"]},
        n=15,
    )

    banner("AUDIT COMPLETE")


if __name__ == "__main__":
    run_audit()
