"""
pipeline.py — Phase 5: End-to-End Inference Pipeline.

Runs the full entity resolution pipeline on the test set:
1. Blocking (generate_candidates via streaming)
2. Feature Engineering (build_feature_matrix via streaming)
3. Inference (LightGBM predict_proba)
4. Thresholding and Submission Formatting

Outputs:
  output/submission.csv — formatted for the competition
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from tqdm import tqdm

from blocking import generate_candidates
from features import build_feature_matrix, FEATURE_COLS


def run_inference_pipeline(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    model_path: Path,
    threshold: float,
    output_path: Path,
):
    print("=" * 60)
    print("END-TO-END INFERENCE PIPELINE")
    print("=" * 60)

    # 1. Load S1
    print(f"Loading S1 from: {s1_path}")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded {len(s1_df):,} S1 entities.")

    # 2. Blocking
    cands = generate_candidates(
        source1_df=s1_df,
        s2_path=s2_path,
        s3_path=s3_path,
        output_path=None,  # We don't need to save intermediate candidates for test
    )

    n_cands = sum(len(v) for v in cands.values())
    print(f"\nGenerated {n_cands:,} candidate pairs to evaluate.")

    if n_cands == 0:
        print("WARNING: No candidates generated. Writing empty predictions.")
        _write_submission(cands.keys(), {}, output_path)
        return

    # 3. Features
    print("\nExtracting Pairwise Features...")
    feat_df = build_feature_matrix(
        s1_df=s1_df,
        s2_path=s2_path,
        s3_path=s3_path,
        candidate_pairs=cands,
        label_col=None,
    )

    # 4. Model Inference
    print(f"\nLoading Model: {model_path}")
    model = lgb.Booster(model_file=str(model_path))

    print(f"Predicting match probabilities (Threshold = {threshold:.3f})...")
    X = feat_df[FEATURE_COLS]
    probs = model.predict(X)  # Booster.predict returns probabilities for binary

    feat_df["prob"] = probs
    feat_df["is_match"] = (probs >= threshold).astype(bool)

    # 5. Build submission dict
    print("\nFormatting predictions...")
    matches = feat_df[feat_df["is_match"]]
    
    predictions: dict[str, list[str]] = {s1_id: [] for s1_id in s1_df["entity_id"]}
    
    # Group matches
    grouped = matches.groupby("s1_id")["sx_id"].apply(list).to_dict()
    for s1_id, matched_ids in grouped.items():
        predictions[s1_id] = matched_ids

    # 6. Write to disk
    _write_submission(s1_df["entity_id"], predictions, output_path)


def _write_submission(s1_ids, predictions: dict[str, list[str]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing submission to: {output_path}")
    rows = []
    for s1_id in s1_ids:
        preds = predictions.get(s1_id, [])
        # The submission requires comma-separated matched entity IDs.
        # If no matches, leave the column empty.
        preds_str = ",".join(sorted(preds))
        rows.append(f"{s1_id}\t{preds_str}")
        
    output_path.write_text("source1_entity_id\tpredicted_match_ids\n" + "\n".join(rows), encoding="utf-8")
    
    # Quick stats
    n_total = len(s1_ids)
    n_linked = sum(1 for p in predictions.values() if len(p) > 0)
    print(f"\nPipeline Complete!")
    print(f"Total S1 entities evaluated : {n_total:,}")
    print(f"Entities linked to >=1 match: {n_linked:,} ({n_linked/max(1,n_total)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=str, required=True, help="Path to test_source1.tsv")
    parser.add_argument("--s2", type=str, required=True, help="Path to test_source2.tsv")
    parser.add_argument("--s3", type=str, required=True, help="Path to test_source3.tsv")
    parser.add_argument("--model", type=str, required=True, help="Path to lgbm_model.txt")
    parser.add_argument("--threshold", type=str, required=True, help="Path to threshold.txt or float value")
    parser.add_argument("--out", type=str, default="output/test_predictions.csv")
    args = parser.parse_args()

    try:
        t_val = float(args.threshold)
    except ValueError:
        t_val = float(Path(args.threshold).read_text().strip())

    run_inference_pipeline(
        s1_path=Path(args.s1),
        s2_path=Path(args.s2),
        s3_path=Path(args.s3),
        model_path=Path(args.model),
        threshold=t_val,
        output_path=Path(args.out)
    )
