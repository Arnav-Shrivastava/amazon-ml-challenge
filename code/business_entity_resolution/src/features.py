"""
features.py — Pairwise feature engineering (Phase 3).

For every (S1, candidate) pair produced by blocking.py, computes a
fixed-width numeric feature vector used to train the LightGBM matcher.

Feature groups (20 features total)
------------------------------------
Name features (8)
    name_lev_norm          Levenshtein normalized similarity (rapidfuzz)
    name_jaro_winkler      Jaro-Winkler similarity (rapidfuzz)
    name_jaccard_tokens    Jaccard on word-token sets
    name_token_sort_ratio  rapidfuzz token_sort_ratio / 100
    name_token_set_ratio   rapidfuzz token_set_ratio / 100
    name_prefix_match      fraction of S1 name tokens found in Sx name
    name_suffix_match      1.0 if legal suffix tokens identical, else 0
    name_script_exact      1.0 if raw (pre-norm) names are character-identical

Address features (7)
    addr_lev_norm          Levenshtein normalized similarity
    addr_jaccard_tokens    Jaccard on word-token sets
    addr_token_sort_ratio  rapidfuzz token_sort_ratio / 100
    addr_postal_match      1.0 exact, 0.5 one side missing, 0.0 mismatch
    addr_city_match        1.0 if any city token overlaps
    addr_subset            1.0 if one token set ⊆ the other
    addr_house_num_match   1.0 if extracted house numbers match (zero-stripped)

Structural features (5)
    country_exact          1.0 if country strings match
    source_is_s3           1.0 if candidate is S3 (S2=0), else 0
    name_len_ratio         min/max of raw name lengths
    addr_len_ratio         min/max of raw address lengths
    addr_both_empty        1.0 if both addresses are empty/very short

Public API
----------
build_feature_matrix(s1_df, sx_df, candidate_pairs) -> pd.DataFrame
    Returns DataFrame with one row per candidate pair and one column per feature,
    plus 's1_id' and 'sx_id' identifier columns.

FEATURE_COLS : list[str]  — ordered list of all 20 feature names.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
import rapidfuzz.distance as rfd
import rapidfuzz.fuzz as rff

from normalize import (
    preprocess_df,
    strip_legal_suffix_series,
    normalize_series,
)

# Canonical ordered feature column list (used by trainer and predictor)
FEATURE_COLS: list[str] = [
    # Name
    "name_lev_norm",
    "name_jaro_winkler",
    "name_jaccard_tokens",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_prefix_match",
    "name_suffix_match",
    "name_script_exact",
    # Address
    "addr_lev_norm",
    "addr_jaccard_tokens",
    "addr_token_sort_ratio",
    "addr_postal_match",
    "addr_city_match",
    "addr_subset",
    "addr_house_num_match",
    # Structural
    "country_exact",
    "source_is_s3",
    "name_len_ratio",
    "addr_len_ratio",
    "addr_both_empty",
]


# ── Helper: pairwise string features ─────────────────────────────────────────

def _lev_norm(a: str, b: str) -> float:
    return rfd.Levenshtein.normalized_similarity(a, b)


def _jaro_winkler(a: str, b: str) -> float:
    return rfd.JaroWinkler.similarity(a, b)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _prefix_match(a: str, b: str) -> float:
    """Fraction of a-tokens found in b-tokens."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def _token_sort(a: str, b: str) -> float:
    return rff.token_sort_ratio(a, b) / 100.0


def _token_set(a: str, b: str) -> float:
    return rff.token_set_ratio(a, b) / 100.0


def _is_subset(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return 1.0 if (sa <= sb or sb <= sa) else 0.0


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if max(la, lb) == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


def _house_num_match(a: str, b: str) -> float:
    """1.0 if both have the same house number (leading-zero stripped)."""
    if not a or not b:
        return 0.5  # one side missing — uncertain
    a_stripped = a.lstrip("0") or "0"
    b_stripped = b.lstrip("0") or "0"
    return 1.0 if a_stripped == b_stripped else 0.0


def _postal_match(a: str, b: str) -> float:
    if a and b:
        return 1.0 if a == b else 0.0
    elif not a and not b:
        return 0.5  # both missing — neutral
    else:
        return 0.5  # one side missing — uncertain


def _city_match(a_tokens: str, b_tokens: str) -> float:
    """1.0 if any city candidate token overlaps between a and b."""
    sa = set(a_tokens.split("|")) - {""}
    sb = set(b_tokens.split("|")) - {""}
    if not sa or not sb:
        return 0.0
    return 1.0 if sa & sb else 0.0


def _script_exact(a: str, b: str) -> float:
    """1.0 if the raw names are character-identical after lowercasing."""
    return 1.0 if a.lower().strip() == b.lower().strip() else 0.0


def _suffix_match(s1_norm: str, sx_norm: str) -> float:
    """
    Compare legal suffix tokens between two normalized names.
    1.0  → both have the same suffix (or both have none)
    0.5  → one has a suffix, the other doesn't
    0.0  → both have suffixes but they differ
    """
    from normalize import _LEGAL_SUFFIXES  # type: ignore[attr-defined]

    def _extract_suffix(name: str) -> str:
        tokens = name.split()
        for n in [3, 2, 1]:
            if len(tokens) > n:
                tail = " ".join(tokens[-n:])
                if tail in _LEGAL_SUFFIXES:
                    return tail
        if tokens and tokens[-1] in _LEGAL_SUFFIXES:
            return tokens[-1]
        return ""

    suf_a = _extract_suffix(s1_norm)
    suf_b = _extract_suffix(sx_norm)
    if not suf_a and not suf_b:
        return 1.0
    if not suf_a or not suf_b:
        return 0.5
    return 1.0 if suf_a == suf_b else 0.0


# ── Row-level feature computation ─────────────────────────────────────────────

def _compute_row_features(
    s1_norm_name: str,
    sx_norm_name: str,
    s1_raw_name: str,
    sx_raw_name: str,
    s1_norm_addr: str,
    sx_norm_addr: str,
    s1_postal: str,
    sx_postal: str,
    s1_city: str,
    sx_city: str,
    s1_house: str,
    sx_house: str,
    s1_country: str,
    sx_country: str,
    sx_is_s3: int,
    s1_raw_addr: str,
    sx_raw_addr: str,
) -> list[float]:
    """Compute all 20 features for one (S1, Sx) pair."""

    # Name features
    name_lev        = _lev_norm(s1_norm_name, sx_norm_name)
    name_jw         = _jaro_winkler(s1_norm_name, sx_norm_name)
    name_jac        = _jaccard(s1_norm_name, sx_norm_name)
    name_tsort      = _token_sort(s1_norm_name, sx_norm_name)
    name_tset       = _token_set(s1_norm_name, sx_norm_name)
    name_prefix     = _prefix_match(s1_norm_name, sx_norm_name)
    name_suffix     = _suffix_match(s1_norm_name, sx_norm_name)
    name_script     = _script_exact(s1_raw_name, sx_raw_name)

    # Address features
    addr_lev        = _lev_norm(s1_norm_addr, sx_norm_addr)
    addr_jac        = _jaccard(s1_norm_addr, sx_norm_addr)
    addr_tsort      = _token_sort(s1_norm_addr, sx_norm_addr)
    addr_postal     = _postal_match(s1_postal, sx_postal)
    addr_city       = _city_match(s1_city, sx_city)
    addr_subset     = _is_subset(s1_norm_addr, sx_norm_addr)
    addr_house      = _house_num_match(s1_house, sx_house)

    # Structural features
    country_match   = 1.0 if s1_country == sx_country else 0.0
    is_s3           = float(sx_is_s3)
    name_len_r      = _len_ratio(s1_raw_name, sx_raw_name)
    addr_len_r      = _len_ratio(s1_raw_addr, sx_raw_addr)
    addr_both_empty = 1.0 if (len(s1_norm_addr) <= 5 and len(sx_norm_addr) <= 5) else 0.0

    return [
        name_lev, name_jw, name_jac, name_tsort, name_tset,
        name_prefix, name_suffix, name_script,
        addr_lev, addr_jac, addr_tsort, addr_postal, addr_city, addr_subset, addr_house,
        country_match, is_s3, name_len_r, addr_len_r, addr_both_empty,
    ]


# ── Batch feature computation ─────────────────────────────────────────────────

def build_feature_matrix(
    s1_df: pd.DataFrame,
    sx_df: pd.DataFrame,
    candidate_pairs: dict[str, set[str]],
    batch_size: int = 100_000,
    label_col: str | None = None,
    gt_dict: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """
    Build the pairwise feature matrix for all candidate pairs.

    Parameters
    ----------
    s1_df           : Source-1 DataFrame (raw, will be preprocessed internally)
    sx_df           : Source-2 + Source-3 combined DataFrame (raw)
    candidate_pairs : output of blocking.generate_candidates()
    batch_size      : pairs per processing batch (tune for RAM)
    label_col       : if given, add a 'label' column (1=match, 0=no-match)
    gt_dict         : {s1_id: set of true match ids} — required if label_col set

    Returns
    -------
    pd.DataFrame with columns: s1_id, sx_id, [FEATURE_COLS], [label_col]
    """
    from tqdm import tqdm

    print("Preprocessing DataFrames for feature computation...")
    s1_pre = preprocess_df(s1_df)
    sx_pre = preprocess_df(sx_df)

    # Build fast lookup dicts (indexed by entity_id)
    s1_lookup = s1_pre.set_index("entity_id")
    sx_lookup = sx_pre.set_index("entity_id")

    # Determine which pairs are S3 (for source_is_s3 feature)
    s3_ids = set(sx_df["entity_id"][sx_df["entity_id"].str.startswith("S3-")])

    # Flatten candidate pairs into a list of (s1_id, sx_id) tuples
    print("Flattening candidate pairs...")
    pairs: list[tuple[str, str]] = []
    for s1_id, sx_set in candidate_pairs.items():
        for sx_id in sx_set:
            if sx_id in sx_lookup.index:
                pairs.append((s1_id, sx_id))

    total_pairs = len(pairs)
    print(f"Total pairs to featurize: {total_pairs:,}")

    # Build feature matrix in batches
    all_rows: list[list] = []
    s1_ids_out: list[str] = []
    sx_ids_out: list[str] = []

    for batch_start in tqdm(range(0, total_pairs, batch_size), desc="Feature batches"):
        batch = pairs[batch_start: batch_start + batch_size]

        for s1_id, sx_id in batch:
            try:
                r1 = s1_lookup.loc[s1_id]
                rx = sx_lookup.loc[sx_id]
            except KeyError:
                continue

            feats = _compute_row_features(
                s1_norm_name = str(r1["_norm_name"]),
                sx_norm_name = str(rx["_norm_name"]),
                s1_raw_name  = str(r1["business_name"]),
                sx_raw_name  = str(rx["business_name"]),
                s1_norm_addr = str(r1["_norm_addr"]),
                sx_norm_addr = str(rx["_norm_addr"]),
                s1_postal    = str(r1["_postal"]),
                sx_postal    = str(rx["_postal"]),
                s1_city      = str(r1["_city_tokens"]),
                sx_city      = str(rx["_city_tokens"]),
                s1_house     = str(r1["_house_num"]),
                sx_house     = str(rx["_house_num"]),
                s1_country   = str(r1["_country"]),
                sx_country   = str(rx["_country"]),
                sx_is_s3     = 1 if sx_id in s3_ids else 0,
                s1_raw_addr  = str(r1["business_address"]),
                sx_raw_addr  = str(rx["business_address"]),
            )
            all_rows.append(feats)
            s1_ids_out.append(s1_id)
            sx_ids_out.append(sx_id)

    print(f"Computed {len(all_rows):,} feature vectors.")

    feat_df = pd.DataFrame(all_rows, columns=FEATURE_COLS)
    feat_df.insert(0, "s1_id", s1_ids_out)
    feat_df.insert(1, "sx_id", sx_ids_out)

    # Optionally add labels
    if label_col is not None and gt_dict is not None:
        feat_df[label_col] = [
            1 if sx_id in gt_dict.get(s1_id, set()) else 0
            for s1_id, sx_id in zip(s1_ids_out, sx_ids_out)
        ]
        n_pos = feat_df[label_col].sum()
        n_neg = len(feat_df) - n_pos
        print(f"Labels: {n_pos:,} positives, {n_neg:,} negatives  (ratio 1:{n_neg/max(n_pos,1):.1f})")

    return feat_df


# ── Standalone sanity check ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    import pandas as pd

    s1 = pd.DataFrame({
        "entity_id":        ["S1-001"],
        "business_name":    ["FHW Copley Inc"],
        "business_address": ["103 Gifford Parkway, Syracuse, NY"],
        "country":          ["US"],
    })
    sx = pd.DataFrame({
        "entity_id":        ["S2-101", "S2-999"],
        "business_name":    ["FHW Copleo Inc", "Unrelated Business"],
        "business_address": ["103 Gifford Parkway, NY, SYRACUSE", "99 Different Blvd, Miami, FL"],
        "country":          ["US", "US"],
    })
    cands = {"S1-001": {"S2-101", "S2-999"}}
    gt    = {"S1-001": {"S2-101"}}

    df = build_feature_matrix(s1, sx, cands, label_col="label", gt_dict=gt)
    print("\nFeature matrix (transposed for readability):")
    print(df.set_index(["s1_id", "sx_id"]).T.to_string())
    print(f"\nPASS — {len(FEATURE_COLS)} features computed for {len(df)} pairs.")
