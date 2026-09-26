"""
blocking.py — Candidate pair generation (Phase 2) — CHUNKED & SINGLE-PASS.

Implements 8 complementary blocking strategies. Uses a chunked, single-pass
streaming approach to prevent Out-Of-Memory (OOM) and massive redundant
computation. Reads S2 and S3 in chunks exactly once, computes all 8 keys
simultaneously per chunk, and holds only the compact (id, key) tables in memory.

Public API
----------
generate_candidates(source1_df, s2_path, s3_path, output_path=None)
    → dict[str, set[str]]
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
    mask = df["_sorted_name_tokens"].str.len() > 2
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values, "key": keys[mask].values})


def _keys_B2(df: pd.DataFrame) -> pd.DataFrame:
    """B2: Soundex phonetic key on first 2 name tokens."""
    keys = "B2:" + df["_soundex_key"]
    mask = df["_soundex_key"].str.len() > 0
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values, "key": keys[mask].values})


def _keys_B3(df: pd.DataFrame) -> pd.DataFrame:
    """B3: postal/PIN code + country."""
    mask = df["_postal"].str.len() > 0
    keys = "B3:" + df["_postal"][mask] + "_" + df["_country"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values, "key": keys.values})


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
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values, "key": keys.values})


def _keys_B6(df: pd.DataFrame) -> pd.DataFrame:
    """B6: address house/building number + country."""
    mask = df["_house_num"].str.len() >= 3
    house = df["_house_num"][mask].str.lstrip("0")
    house = house.where(house.str.len() > 0, df["_house_num"][mask])
    keys = "B6:" + house + "_" + df["_country"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values, "key": keys.values})


def _keys_B7(df: pd.DataFrame) -> pd.DataFrame:
    """B7: country + Soundex — fallback for records with sparse/empty address."""
    sparse = df["_norm_addr"].str.len() <= 10
    mask   = sparse & (df["_soundex_key"].str.len() > 0)
    keys   = "B7:" + df["_country"][mask] + "_" + df["_soundex_key"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values, "key": keys.values})


def _keys_B8(df: pd.DataFrame) -> pd.DataFrame:
    """B8: non-ASCII script key (Devanagari / Hindi rendering)."""
    mask = df["_script_key"].str.len() > 3
    keys = "B8:" + df["_script_key"][mask]
    return pd.DataFrame({"entity_id": df["entity_id"][mask].values, "key": keys.values})


_STRATEGIES: list[tuple[str, Callable]] = [
    ("B1_name_token_sort",   _keys_B1),
    ("B2_phonetic_soundex",  _keys_B2),
    ("B3_postal_code",       _keys_B3),
    # ("B4_city_token",        _keys_B4),  # Disabled: Causes memory explosion during join due to low cardinality
    ("B5_country_prefix",    _keys_B5),
    ("B6_house_number",      _keys_B6),
    ("B7_sparse_fallback",   _keys_B7),
    ("B8_script_key",        _keys_B8),
]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_candidates(
    source1_df: pd.DataFrame,
    s2_path: Path,
    s3_path: Path,
    output_path: str | None = None,
    chunksize: int = 150_000,
) -> dict[str, set[str]]:
    """
    Generate candidate pairs for all S1 entities (Single-Pass Streaming implementation).
    """
    print("=" * 60)
    print("CANDIDATE GENERATION (blocking - single pass)")
    print("=" * 60)

    print("\nStep 1/4 — Preprocessing S1 DataFrame in memory...")
    s1_pre = preprocess_df(source1_df)
    print(f"done ({len(s1_pre):,} S1 entities)")

    candidates: dict[str, set[str]] = defaultdict(set)
    for eid in s1_pre["entity_id"]:
        candidates[eid]

    print("\nStep 2/4 — Streaming S2/S3 to extract all blocking keys...")
    s23_key_chunks: dict[str, list[pd.DataFrame]] = {name: [] for name, _ in _STRATEGIES}

    for path in [s2_path, s3_path]:
        if not path.exists():
            continue
        print(f"  Reading {path.name}...")
        for chunk in tqdm(
            pd.read_csv(
                path, sep="\t", dtype=str, keep_default_na=False,
                chunksize=chunksize, encoding="utf-8", on_bad_lines="skip"
            ),
            desc=f"  {path.name}", leave=False
        ):
            chunk.columns = [c.strip() for c in chunk.columns]
            pre = preprocess_df(chunk)
            
            for name, fn in _STRATEGIES:
                keys = fn(pre)
                keys = keys[keys["key"].str.len() > 0]
                if len(keys) > 0:
                    s23_key_chunks[name].append(keys)
            
            del pre, chunk
    
    gc.collect()

    print(f"\nStep 3/4 — Joining S1 against S2/S3 index for each strategy...\n")
    stats: dict[str, int] = {}
    s1_chunk_size = 50_000

    for name, fn in _STRATEGIES:
        print(f"  [{name}] Consolidating index...", end=" ", flush=True)
        chunks = s23_key_chunks[name]
        if not chunks:
            print("0 keys")
            stats[name] = 0
            continue
            
        s23_keys = pd.concat(chunks, ignore_index=True)
        s23_key_chunks[name] = []  # Free list memory immediately
        gc.collect()
        
        print(f"{len(s23_keys):,} keys")
        
        n_chunks = max(1, (len(s1_pre) + s1_chunk_size - 1) // s1_chunk_size)
        new_pairs = 0

        for chunk_i in tqdm(range(n_chunks), desc=f"    {name}", leave=False):
            s1_chunk = s1_pre.iloc[chunk_i * s1_chunk_size: (chunk_i + 1) * s1_chunk_size]
            s1_keys  = fn(s1_chunk)
            s1_keys  = s1_keys[s1_keys["key"].str.len() > 0]

            if len(s1_keys) == 0:
                continue

            pairs = pd.merge(
                s1_keys.rename(columns={"entity_id": "s1_id"}),
                s23_keys.rename(columns={"entity_id": "s23_id"}),
                on="key",
                how="inner",
            )[["s1_id", "s23_id"]].drop_duplicates()

            if len(pairs) == 0:
                continue

            s1_arr  = pairs["s1_id"].values
            s23_arr = pairs["s23_id"].values
            for s1_id, s23_id in zip(s1_arr, s23_arr):
                if len(candidates[s1_id]) < 100:
                    candidates[s1_id].add(s23_id)

            new_pairs += len(pairs)
            del pairs

        del s23_keys
        gc.collect()
        print(f"  [{name}] -> {new_pairs:,} candidate pair hits")
        stats[name] = new_pairs

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

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nStep 4/4 — Writing candidate_pairs.tsv → {out_path} ...", end=" ", flush=True)
        rows = []
        for s1_id in sorted(candidates.keys()):
            cands = sorted(candidates[s1_id])
            rows.append(f"{s1_id}\t{','.join(cands)}")
        out_path.write_text("source1_entity_id\tcandidate_entity_ids\n" + "\n".join(rows), encoding="utf-8")
        print(f"done ({len(rows):,} rows)")

    return dict(candidates)


def evaluate_blocking(candidates: dict[str, set[str]], gt_dict: dict[str, set[str]]) -> dict:
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
    n_entities = len(gt_dict)
    return {
        "recall_ceiling":        found_gt / total_gt if total_gt > 0 else 0.0,
        "n_gt_pairs_total":      total_gt,
        "n_gt_pairs_found":      found_gt,
        "total_candidate_pairs": total_cand,
        "avg_candidates":        total_cand / n_entities if n_entities > 0 else 0.0,
        "n_entities":            n_entities,
        "n_singletons":          n_singletons,
    }


if __name__ == "__main__":
    import sys
    ROOT     = Path(__file__).resolve().parent.parent.parent.parent
    TRAIN    = ROOT / "dataset" / "train"
    SPLITS   = ROOT / "dataset" / "splits"
    OUTPUT   = ROOT / "output"
    
    s2_path = TRAIN / "train_source2.tsv"
    s3_path = TRAIN / "train_source3.tsv"

    print("Loading S1 data...")
    s1 = pd.read_csv(TRAIN / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)

    val_ids = set(Path(SPLITS / "val_source1_ids.txt").read_text().splitlines())
    s1_val  = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"Running blocking on {len(s1_val):,} val S1 entities...\n")
    cands = generate_candidates(s1_val, s2_path, s3_path, output_path=str(OUTPUT / "candidate_pairs_val.tsv"))

    gt_raw = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_dict: dict[str, set[str]] = {eid: set() for eid in val_ids}
    for _, row in gt_raw[gt_raw["source1_entity_id"].isin(val_ids)].iterrows():
        ids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_dict[row["source1_entity_id"]] = set(ids)

    metrics = evaluate_blocking(cands, gt_dict)
    print("\nBLOCKING EVALUATION (val split):")
    print(f"  Recall ceiling       : {metrics['recall_ceiling']*100:.2f}%")
    print(f"  GT pairs total       : {metrics['n_gt_pairs_total']:,}")
    print(f"  GT pairs found       : {metrics['n_gt_pairs_found']:,}")
    print(f"  Total candidate pairs: {metrics['total_candidate_pairs']:,}")
    print(f"  Avg candidates/entity: {metrics['avg_candidates']:.1f}")
    print(f"  Singletons in val    : {metrics['n_singletons']:,}")
