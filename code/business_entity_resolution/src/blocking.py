"""
blocking.py — Candidate pair generation (Phase 2).

Implements 8 complementary blocking strategies, all unioned into a single
candidate set per S1 entity. Uses a pandas-merge approach: each strategy
builds a (entity_id, key) table for S2+S3, then merges with the S1 key
table to find matches — avoiding full pairwise comparison.

Blocking strategies
-------------------
B1  Sorted first-3 significant name tokens (robust to reordering)
B2  Soundex phonetic key on first 2 name tokens (typo/variant resilience)
B3  Postal/PIN code + country (exact address match)
B4  City tokens + country  (per comma-segment, multi-valued)
B5  Country + normalized name prefix (6 chars)
B6  Address house/building number + country (3+ digit number)
B7  Country + Soundex (sparse-address fallback — empty/missing address)
B8  Non-ASCII script key (Devanagari / Hindi exact rendering match)

Public API
----------
generate_candidates(source1_df, source2_df, source3_df, output_path=None)
    → dict[str, set[str]]

evaluate_blocking(candidates, gt_dict)
    → dict with recall_ceiling, avg_candidates, etc.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from normalize import preprocess_df


# ── Blocking key functions ────────────────────────────────────────────────────
# Each function receives a preprocessed DataFrame and returns a
# (entity_id, key) DataFrame — potentially multiple rows per entity.

def _keys_B1(df: pd.DataFrame) -> pd.DataFrame:
    """B1: sorted first-3 significant name tokens."""
    keys = "B1:" + df["_sorted_name_tokens"]
    mask = df["_sorted_name_tokens"].str.len() > 2  # at least a 3-char key
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values,
                         "key": keys[mask].values})


def _keys_B2(df: pd.DataFrame) -> pd.DataFrame:
    """B2: Soundex phonetic key on first 2 name tokens."""
    keys = "B2:" + df["_soundex_key"]
    mask = df["_soundex_key"].str.len() > 0
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values,
                         "key": keys[mask].values})


def _keys_B3(df: pd.DataFrame) -> pd.DataFrame:
    """B3: postal/PIN code + country."""
    mask = df["_postal"].str.len() > 0
    keys = "B3:" + df["_postal"][mask] + "_" + df["_country"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values,
                         "key": keys.values})


def _keys_B4(df: pd.DataFrame) -> pd.DataFrame:
    """B4: city tokens + country (multi-valued — one row per city token)."""
    mask = df["_city_tokens"].str.len() > 0
    sub = df[mask][["entity_id", "_city_tokens", "_country"]].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["entity_id", "key"])
    sub = sub.assign(city=sub["_city_tokens"].str.split("|")).explode("city")
    sub["city"] = sub["city"].str.strip()
    sub = sub[sub["city"].str.len() > 3]
    sub["key"] = "B4:" + sub["city"] + "_" + sub["_country"]
    return sub[["entity_id", "key"]].reset_index(drop=True)


def _keys_B5(df: pd.DataFrame) -> pd.DataFrame:
    """B5: country + normalized name prefix (first 6 chars)."""
    prefix = df["_norm_name_stripped"].str[:6]
    mask = prefix.str.len() >= 4
    keys = "B5:" + df["_country"][mask] + "_" + prefix[mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values,
                         "key": keys.values})


def _keys_B6(df: pd.DataFrame) -> pd.DataFrame:
    """B6: address house/building number + country."""
    mask = df["_house_num"].str.len() >= 3
    # Strip leading zeros for consistent matching (08111 == 8111)
    house = df["_house_num"][mask].str.lstrip("0")
    house = house.where(house.str.len() > 0, df["_house_num"][mask])
    keys = "B6:" + house + "_" + df["_country"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values,
                         "key": keys.values})


def _keys_B7(df: pd.DataFrame) -> pd.DataFrame:
    """B7: country + Soundex — fallback for records with sparse/empty address."""
    # Activate only for records with very short address (≤10 chars normalized)
    sparse = df["_norm_addr"].str.len() <= 10
    mask   = sparse & (df["_soundex_key"].str.len() > 0)
    keys   = "B7:" + df["_country"][mask] + "_" + df["_soundex_key"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values,
                         "key": keys.values})


def _keys_B8(df: pd.DataFrame) -> pd.DataFrame:
    """B8: non-ASCII script key (Devanagari / Hindi rendering)."""
    mask = df["_script_key"].str.len() > 3
    keys = "B8:" + df["_script_key"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values,
                         "key": keys.values})


# ── Strategy registry ─────────────────────────────────────────────────────────

_STRATEGIES: list[tuple[str, Callable]] = [
    ("B1_name_token_sort",   _keys_B1),
    ("B2_phonetic_soundex",  _keys_B2),
    ("B3_postal_code",       _keys_B3),
    ("B4_city_token",        _keys_B4),
    ("B5_country_prefix",    _keys_B5),
    ("B6_house_number",      _keys_B6),
    ("B7_sparse_fallback",   _keys_B7),
    ("B8_script_key",        _keys_B8),
]


# ── Generic strategy runner ───────────────────────────────────────────────────

def _run_strategy(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    key_fn: Callable[[pd.DataFrame], pd.DataFrame],
    candidates: dict[str, set[str]],
    strategy_name: str,
    chunk_size: int = 50_000,
) -> int:
    """
    Run one blocking strategy; update candidates in-place.

    Algorithm:
    1. Compute (entity_id, key) table for S23.
    2. In chunks of S1 entities, compute their keys, inner-join with S23 table.
    3. For each matched (s1_id, s23_id) pair, add s23_id to candidates[s1_id].

    Memory is bounded by s23_key_df size (~300MB typical) + chunk merge (~100MB).
    """
    print(f"  [{strategy_name}] Building S23 index...", end=" ", flush=True)
    s23_keys = key_fn(s23_df)
    s23_keys = s23_keys[s23_keys["key"].str.len() > 0].reset_index(drop=True)
    print(f"{len(s23_keys):,} S23 keys")

    if len(s23_keys) == 0:
        return 0

    n_chunks = max(1, (len(s1_df) + chunk_size - 1) // chunk_size)
    new_pairs = 0

    for chunk_i in tqdm(range(n_chunks), desc=f"    {strategy_name}", leave=False):
        s1_chunk = s1_df.iloc[chunk_i * chunk_size: (chunk_i + 1) * chunk_size]
        s1_keys  = key_fn(s1_chunk)
        s1_keys  = s1_keys[s1_keys["key"].str.len() > 0]

        if len(s1_keys) == 0:
            continue

        # Inner join on key
        pairs = pd.merge(
            s1_keys.rename(columns={"entity_id": "s1_id"}),
            s23_keys.rename(columns={"entity_id": "s23_id"}),
            on="key",
            how="inner",
        )[["s1_id", "s23_id"]].drop_duplicates()

        if len(pairs) == 0:
            continue

        # Update candidates dict — tight loop for performance
        s1_arr  = pairs["s1_id"].values
        s23_arr = pairs["s23_id"].values
        for s1_id, s23_id in zip(s1_arr, s23_arr):
            candidates[s1_id].add(s23_id)

        new_pairs += len(pairs)
        del pairs

    del s23_keys
    gc.collect()
    print(f"  [{strategy_name}] -> {new_pairs:,} candidate pair hits")
    return new_pairs


# ── Public API ────────────────────────────────────────────────────────────────

def generate_candidates(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    output_path: str | None = None,
    chunk_size: int = 50_000,
) -> dict[str, set[str]]:
    """
    Generate candidate pairs for all S1 entities.

    Same function used for train (val evaluation) and test (final submission).
    Every S1 entity gets an entry in the returned dict — singletons have empty sets.

    Parameters
    ----------
    source1_df  : S1 DataFrame (entity_id, business_name, business_address, country)
    source2_df  : S2 DataFrame (same schema)
    source3_df  : S3 DataFrame (same schema)
    output_path : if given, writes candidate_pairs.tsv to this path
    chunk_size  : S1 entities processed per merge chunk (tune for RAM)

    Returns
    -------
    dict[str, set[str]]  — {s1_entity_id: {candidate_s2_or_s3_ids}}
    """
    print("=" * 60)
    print("CANDIDATE GENERATION (blocking)")
    print("=" * 60)

    # 1. Preprocess all DataFrames
    print("\nStep 1/3 — Preprocessing source DataFrames...")
    print("  Preprocessing S1...", end=" ", flush=True)
    s1_pre  = preprocess_df(source1_df)
    print(f"done ({len(s1_pre):,} entities)")

    print("  Preprocessing S2...", end=" ", flush=True)
    s2_pre  = preprocess_df(source2_df)
    print(f"done ({len(s2_pre):,} entities)")

    print("  Preprocessing S3...", end=" ", flush=True)
    s3_pre  = preprocess_df(source3_df)
    print(f"done ({len(s3_pre):,} entities)")

    print("  Concatenating S2 + S3...", end=" ", flush=True)
    s23_pre = pd.concat([s2_pre, s3_pre], ignore_index=True)
    del s2_pre, s3_pre
    gc.collect()
    print(f"done ({len(s23_pre):,} entities)")

    # 2. Initialize candidates dict — one entry per S1 entity (including singletons)
    candidates: dict[str, set[str]] = defaultdict(set)
    for eid in s1_pre["entity_id"]:
        candidates[eid]  # touch each key so singletons appear in the dict

    # 3. Run all blocking strategies
    print(f"\nStep 2/3 — Running {len(_STRATEGIES)} blocking strategies...\n")
    stats: dict[str, int] = {}
    for name, fn in _STRATEGIES:
        hit_count = _run_strategy(s1_pre, s23_pre, fn, candidates, name, chunk_size)
        stats[name] = hit_count

    del s23_pre
    gc.collect()

    # 4. Summary statistics
    n_total     = len(candidates)
    n_with_cands = sum(1 for v in candidates.values() if v)
    total_cands  = sum(len(v) for v in candidates.values())
    avg_cands    = total_cands / n_total if n_total else 0.0

    print(f"\n{'=' * 60}")
    print("BLOCKING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  S1 entities           : {n_total:,}")
    print(f"  Entities with ≥1 cand : {n_with_cands:,}  ({n_with_cands/n_total*100:.1f}%)")
    print(f"  Total candidate pairs : {total_cands:,}")
    print(f"  Avg candidates/entity : {avg_cands:.1f}")
    print(f"\n  Strategy hit counts:")
    for name, cnt in stats.items():
        print(f"    {name:<28}: {cnt:>12,}")

    # 5. Write output TSV (if requested)
    if output_path is not None:
        _write_candidate_tsv(candidates, output_path)

    return dict(candidates)


def _write_candidate_tsv(candidates: dict[str, set[str]], path: str) -> None:
    """Write candidate_pairs.tsv in submission format."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 3/3 — Writing candidate_pairs.tsv → {out_path} ...", end=" ", flush=True)
    rows = []
    for s1_id in sorted(candidates.keys()):
        cands = sorted(candidates[s1_id])
        rows.append(f"{s1_id}\t{','.join(cands)}")

    out_path.write_text("source1_entity_id\tcandidate_entity_ids\n" + "\n".join(rows),
                        encoding="utf-8")
    print(f"done ({len(rows):,} rows)")


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_blocking(
    candidates: dict[str, set[str]],
    gt_dict: dict[str, set[str]],
) -> dict:
    """
    Measure blocking quality on a ground-truth dict.

    Parameters
    ----------
    candidates : output of generate_candidates()
    gt_dict    : {s1_id: set of true match IDs}
                 Singletons have empty sets.

    Returns
    -------
    dict with:
        recall_ceiling        : fraction of true GT pairs present in candidates
        n_gt_pairs_total      : total GT pairs across all matched entities
        n_gt_pairs_found      : GT pairs recovered by blocking
        total_candidate_pairs : |union of all candidate sets|
        avg_candidates        : average candidate set size per S1 entity
        n_entities            : total S1 entities evaluated
        n_singletons          : S1 entities with no true match
    """
    total_gt   = 0
    found_gt   = 0
    total_cand = 0
    n_singletons = 0

    for s1_id, true_matches in gt_dict.items():
        cands = candidates.get(s1_id, set())
        total_cand += len(cands)

        if not true_matches:
            n_singletons += 1
            continue

        total_gt += len(true_matches)
        found_gt += len(true_matches & cands)

    n_entities    = len(gt_dict)
    recall_ceil   = found_gt / total_gt if total_gt > 0 else 0.0
    avg_cands     = total_cand / n_entities if n_entities > 0 else 0.0

    return {
        "recall_ceiling":        recall_ceil,
        "n_gt_pairs_total":      total_gt,
        "n_gt_pairs_found":      found_gt,
        "total_candidate_pairs": total_cand,
        "avg_candidates":        avg_cands,
        "n_entities":            n_entities,
        "n_singletons":          n_singletons,
    }


# ── Standalone run (for testing on val split) ─────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT     = Path(__file__).resolve().parent.parent.parent.parent
    TRAIN    = ROOT / "dataset" / "train"
    SPLITS   = ROOT / "dataset" / "splits"
    OUTPUT   = ROOT / "output"

    print("Loading data...")
    s1 = pd.read_csv(TRAIN / "train_source1.tsv", sep="\t", dtype=str,
                     keep_default_na=False, low_memory=False)
    s2 = pd.read_csv(TRAIN / "train_source2.tsv", sep="\t", dtype=str,
                     keep_default_na=False, low_memory=False)
    s3 = pd.read_csv(TRAIN / "train_source3.tsv", sep="\t", dtype=str,
                     keep_default_na=False, low_memory=False)
    gt_raw = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t", dtype=str,
                         keep_default_na=False)

    # Use val split only for speed during dev
    val_ids = set(Path(SPLITS / "val_source1_ids.txt").read_text().splitlines())
    s1_val  = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"Running blocking on {len(s1_val):,} val S1 entities...\n")
    cands = generate_candidates(s1_val, s2, s3,
                                output_path=str(OUTPUT / "candidate_pairs_val.tsv"))

    # Build GT dict for val
    gt_dict: dict[str, set[str]] = {eid: set() for eid in val_ids}
    for _, row in gt_raw[gt_raw["source1_entity_id"].isin(val_ids)].iterrows():
        ids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_dict[row["source1_entity_id"]] = set(ids)

    # Evaluate
    metrics = evaluate_blocking(cands, gt_dict)
    print("\nBLOCKING EVALUATION (val split):")
    print(f"  Recall ceiling       : {metrics['recall_ceiling']*100:.2f}%")
    print(f"  GT pairs total       : {metrics['n_gt_pairs_total']:,}")
    print(f"  GT pairs found       : {metrics['n_gt_pairs_found']:,}")
    print(f"  Total candidate pairs: {metrics['total_candidate_pairs']:,}")
    print(f"  Avg candidates/entity: {metrics['avg_candidates']:.1f}")
    print(f"  Singletons in val    : {metrics['n_singletons']:,}")
