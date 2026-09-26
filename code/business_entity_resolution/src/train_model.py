"""
train_model.py — Phase 4: LightGBM Model Training and Threshold Tuning.

Trains a LightGBM classifier on the pairwise feature matrix to predict if
an (S1, Sx) candidate pair is a true match. Tunes the prediction threshold
to maximize the F0.5 score on a validation set.

Outputs:
  output/lgbm_model.txt      — trained LightGBM model
  output/threshold.txt       — optimal probability threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score


def train_lgbm(train_df: pd.DataFrame, feature_cols: list[str], label_col: str = "label"):
    """Train LightGBM binary classifier on feature matrix."""
    print("=" * 60)
    print("LIGHTGBM MODEL TRAINING")
    print("=" * 60)
    
    X = train_df[feature_cols]
    y = train_df[label_col]
    
    print(f"Training on {len(train_df):,} candidate pairs...")
    print(f"Features: {len(feature_cols)}")
    print(f"Positives: {y.sum():,}  ({y.mean()*100:.2f}%)")

    # We use scale_pos_weight to handle the heavy class imbalance
    # (usually ~1:20 or more for blocking outputs).
    pos_weight = (len(y) - y.sum()) / max(1, y.sum())

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "scale_pos_weight": pos_weight,
        "n_estimators": 200,
        "n_jobs": -1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    
    # Feature importance
    print("\nTop 10 Feature Importances (split):")
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, score in imp.head(10).items():
        print(f"  {feat:<25} {score}")

    return model


def tune_threshold_f05(val_df: pd.DataFrame, model, feature_cols: list[str]) -> float:
    """Find probability threshold that maximizes F0.5 score."""
    print("\n=" * 60)
    print("THRESHOLD TUNING (F0.5)")
    print("=" * 60)

    X_val = val_df[feature_cols]
    y_val = val_df["label"]
    
    print("Predicting probabilities on validation set...")
    probs = model.predict_proba(X_val)[:, 1]
    
    best_t = 0.5
    best_f05 = -1.0
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    print(f"\nEvaluating {len(thresholds)} thresholds...")
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        # We tune using the pair-level F0.5. Note that the final challenge metric
        # is entity-level, but pair-level F0.5 highly correlates with it.
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        
        if f05 > best_f05:
            best_f05 = f05
            best_t = t
            
        print(f"  t={t:.2f}  |  P: {p:.4f}  R: {r:.4f}  F0.5: {f05:.4f}")

    print(f"\nBest threshold : {best_t:.2f}")
    print(f"Best val F0.5  : {best_f05:.4f}")
    
    return best_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=str, required=True, help="Path to train feature matrix (Parquet/CSV)")
    parser.add_argument("--val-features", type=str, required=True, help="Path to validation feature matrix (Parquet/CSV)")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train features: {args.train_features}")
    if args.train_features.endswith(".parquet"):
        train_df = pd.read_parquet(args.train_features)
    else:
        train_df = pd.read_csv(args.train_features)
        
    print(f"Loading val features: {args.val_features}")
    if args.val_features.endswith(".parquet"):
        val_df = pd.read_parquet(args.val_features)
    else:
        val_df = pd.read_csv(args.val_features)

    # Automatically identify feature columns (excluding identifiers and label)
    exclude = {"s1_id", "sx_id", "label"}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    # 1. Train Model
    model = train_lgbm(train_df, feature_cols)
    
    # Save Model
    model_path = out_dir / "lgbm_model.txt"
    model.booster_.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # 2. Tune Threshold
    best_t = tune_threshold_f05(val_df, model, feature_cols)
    
    # Save Threshold
    thresh_path = out_dir / "threshold.txt"
    thresh_path.write_text(str(best_t))
    print(f"Threshold saved to: {thresh_path}")
