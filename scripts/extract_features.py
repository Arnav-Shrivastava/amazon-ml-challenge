"""
extract_features.py — Extract features for generated candidates.
"""

import argparse
from pathlib import Path
import pandas as pd
import sys
import os

# Add src to pythonpath so we can import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code', 'business_entity_resolution', 'src'))
from features import build_feature_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True, help="Path to train_source1.tsv")
    parser.add_argument("--s2", required=True, help="Path to train_source2.tsv")
    parser.add_argument("--s3", required=True, help="Path to train_source3.tsv")
    parser.add_argument("--cands", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--gt", required=False, help="Path to ground truth TSV (to add labels)")
    parser.add_argument("--out", required=True, help="Output feature matrix (.parquet)")
    parser.add_argument("--sample", required=False, help="Number of S1 entities to sample")
    args = parser.parse_args()
    
    print("Loading S1 data...")
    s1_df = pd.read_csv(args.s1, sep="\t", dtype=str, keep_default_na=False)
    
    print("Loading candidate pairs...")
    cands_df = pd.read_csv(args.cands, sep="\t", dtype=str, keep_default_na=False)
    
    # Filter S1 to only those with candidates
    cands = {}
    for _, row in cands_df.iterrows():
        eid = row["source1_entity_id"]
        c_str = row["candidate_entity_ids"]
        if c_str:
            cands[eid] = set(c_str.split(","))
            
    # Subsample if requested
    if args.sample:
        n_sample = int(args.sample)
        import random
        keys = list(cands.keys())
        random.seed(42)
        random.shuffle(keys)
        sampled_keys = keys[:n_sample]
        cands = {k: cands[k] for k in sampled_keys}

    # Subset S1
    s1_df = s1_df[s1_df["entity_id"].isin(cands.keys())].reset_index(drop=True)
    
    gt_dict = None
    if args.gt:
        print("Loading ground truth...")
        gt_df = pd.read_csv(args.gt, sep="\t", dtype=str, keep_default_na=False)
        gt_dict = {}
        for _, row in gt_df.iterrows():
            eid = row["source1_entity_id"]
            if eid in cands:
                m_str = row["matched_entity_ids"]
                if m_str:
                    gt_dict[eid] = set(m_str.split(","))
                    
    print(f"Total S1 entities to featurize: {len(s1_df):,}")
    print("\nExtracting Features...")
    feat_df = build_feature_matrix(
        s1_df=s1_df,
        s2_path=Path(args.s2),
        s3_path=Path(args.s3),
        candidate_pairs=cands,
        label_col="label" if gt_dict else None,
        gt_dict=gt_dict
    )
    
    print(f"\nSaving features to {args.out}...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(args.out, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
