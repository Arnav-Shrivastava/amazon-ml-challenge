"""
eval_f05.py — Macro-averaged F₀.₅ evaluator.

Formula (per entity):
    P  = |predicted ∩ ground_truth| / |predicted|       (0 if predicted empty)
    R  = |predicted ∩ ground_truth| / |ground_truth|    (1 if ground_truth empty & predicted empty)
    F₀.₅ = (1 + 0.5²) × P × R / (0.5² × P + R)        (0 if P=R=0)

Singletons (ground_truth = ∅):
    - Predicted ∅  → P=1, R=1, F₀.₅ = 1.0
    - Predicted ≠ ∅ → P=0, R=undefined → F₀.₅ = 0.0

Macro-average: mean of per-entity F₀.₅ across ALL S1 entities (including singletons).
"""

from __future__ import annotations

import math
from typing import Dict, Set


def f05_score(predicted: Set[str], ground_truth: Set[str]) -> float:
    """Compute F₀.₅ for a single S1 entity.

    Parameters
    ----------
    predicted:    set of predicted matched IDs (S2-/S3- prefix)
    ground_truth: set of true matched IDs (empty set for singletons)

    Returns
    -------
    float in [0.0, 1.0]
    """
    is_singleton = len(ground_truth) == 0

    if is_singleton:
        # Correctly predicted no match → perfect score
        return 1.0 if len(predicted) == 0 else 0.0

    if len(predicted) == 0:
        # Missed all true matches
        return 0.0

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = 0.25  # β = 0.5, β² = 0.25
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute macro-averaged F₀.₅ across all S1 entities.

    Parameters
    ----------
    predictions:  {source1_entity_id: set of predicted IDs}
    ground_truth: {source1_entity_id: set of true matched IDs}
                  Keys must cover ALL S1 entities (singletons have empty sets).

    Returns
    -------
    dict with keys: 'macro_f05', 'macro_precision', 'macro_recall',
                    'n_entities', 'n_singletons', 'n_matched'
    """
    f05_scores, precisions, recalls = [], [], []

    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        is_singleton = len(gt_set) == 0

        score = f05_score(pred_set, gt_set)
        f05_scores.append(score)

        if is_singleton:
            p = 1.0 if len(pred_set) == 0 else 0.0
            r = 1.0 if len(pred_set) == 0 else 0.0
        elif len(pred_set) == 0:
            p, r = 0.0, 0.0
        else:
            tp = len(pred_set & gt_set)
            p = tp / len(pred_set)
            r = tp / len(gt_set)

        precisions.append(p)
        recalls.append(r)

    n = len(ground_truth)
    n_singletons = sum(1 for v in ground_truth.values() if len(v) == 0)

    return {
        "macro_f05": sum(f05_scores) / n if n else 0.0,
        "macro_precision": sum(precisions) / n if n else 0.0,
        "macro_recall": sum(recalls) / n if n else 0.0,
        "n_entities": n,
        "n_singletons": n_singletons,
        "n_matched": n - n_singletons,
    }


# ── Unit tests ────────────────────────────────────────────────────────────────

def _run_unit_tests() -> None:
    """Verify against the worked example in the problem statement."""
    import sys, io
    # Ensure stdout handles Unicode on Windows (cp1252 terminals reject subscripts)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    tol = 0.001

    # Worked example: predicted [S2-00047, S2-00193, S3-00812]
    #                 ground truth [S2-00047, S3-00812]
    # P = 2/3, R = 2/2 = 1.0
    # F₀.₅ = 1.25 * (2/3 * 1.0) / (0.25 * (2/3) + 1.0)
    #       = 1.25 * 0.6667 / (0.1667 + 1.0)
    #       = 0.8333 / 1.1667 ≈ 0.7143
    score = f05_score(
        predicted={"S2-00047", "S2-00193", "S3-00812"},
        ground_truth={"S2-00047", "S3-00812"},
    )
    expected = 0.7143
    assert math.isclose(score, expected, abs_tol=tol), (
        f"Worked example FAILED: got {score:.4f}, expected {expected:.4f}"
    )
    print(f"  [PASS] Worked example: F0.5 = {score:.4f} (expected ~{expected})")

    # Singleton — correct prediction (empty → empty)
    score_singleton_correct = f05_score(predicted=set(), ground_truth=set())
    assert score_singleton_correct == 1.0, "Singleton correct FAILED"
    print(f"  [PASS] Singleton correct: F0.5 = {score_singleton_correct}")

    # Singleton — wrong prediction (empty GT but predicted something)
    score_singleton_wrong = f05_score(predicted={"S2-00001"}, ground_truth=set())
    assert score_singleton_wrong == 0.0, "Singleton wrong FAILED"
    print(f"  [PASS] Singleton wrong: F0.5 = {score_singleton_wrong}")

    # Perfect match
    score_perfect = f05_score(predicted={"S2-00001", "S3-00002"}, ground_truth={"S2-00001", "S3-00002"})
    assert score_perfect == 1.0, "Perfect match FAILED"
    print(f"  [PASS] Perfect match: F0.5 = {score_perfect}")

    # No overlap (all wrong)
    score_none = f05_score(predicted={"S2-99999"}, ground_truth={"S2-00001"})
    assert score_none == 0.0, "No overlap FAILED"
    print(f"  [PASS] No overlap: F0.5 = {score_none}")

    # Missed all (false negatives only)
    score_missed = f05_score(predicted=set(), ground_truth={"S2-00001"})
    assert score_missed == 0.0, "Missed all FAILED"
    print(f"  [PASS] Missed all (FN only): F0.5 = {score_missed}")

    print("\n  All unit tests PASSED.")


if __name__ == "__main__":
    print("Running eval_f05.py unit tests...")
    _run_unit_tests()
