"""smoke_test_blocking.py — Quick sanity check for normalize.py + blocking.py."""
import sys
sys.path.insert(0, "code/business_entity_resolution/src")

import pandas as pd
from normalize import preprocess_df
from blocking import generate_candidates, evaluate_blocking

# Small synthetic dataset based on real noisy examples from the audit
s1 = pd.DataFrame({
    "entity_id":        ["S1-001", "S1-002", "S1-003"],
    "business_name":    ["FHW Copley Inc",
                         "Apex Trading Pvt Ltd",
                         "H 2 Star Schwab LLC"],
    "business_address": ["103 Gifford Parkway, Syracuse, NY",
                         "D/1/0031 Sushant Golf City, Lucknow, Uttar Pradesh",
                         "315 98 Street, New York, NY"],
    "country":          ["US", "India", "US"],
})
s2 = pd.DataFrame({
    "entity_id":        ["S2-101", "S2-102", "S2-103"],
    "business_name":    ["FHW Copleo Inc",
                         "Apex Trading Pvt Limited",
                         "h2star.com (ID: 50633)"],
    "business_address": ["103 Gifford Parkway, NY, SYRACUSE",
                         "D/1/0031/1 SUSHANT GOLF CITY, ANSAL API, LUCKNOW",
                         "315 98 ST, NEW YORK, NY"],
    "country":          ["US", "India", "US"],
})
s3 = pd.DataFrame({
    "entity_id":        ["S3-201", "S3-202"],
    "business_name":    ["FHW Copley Incorporated",
                         "Apex Trading Private Limited"],
    "business_address": ["103 1/2 Gifford Pkwy, Syracuse, New York",
                         "LUCKNOW, D/1/0031/1 SUSHANT GOLF CITY"],
    "country":          ["US", "India"],
})

print("Running generate_candidates on smoke-test data...")
cands = generate_candidates(s1, s2, s3)

print("\nCandidate sets:")
for k, v in sorted(cands.items()):
    print(f"  {k}: {sorted(v)}")

# Ground truth
gt = {
    "S1-001": {"S2-101", "S3-201"},
    "S1-002": {"S2-102", "S3-202"},
    "S1-003": {"S2-103"},
}
m = evaluate_blocking(cands, gt)

print(f"\nRecall ceiling : {m['recall_ceiling']*100:.1f}%")
print(f"Avg candidates : {m['avg_candidates']:.1f}")

# Assert expected recall for the known noisy pairs
missing = []
for s1_id, true_set in gt.items():
    cand_set = cands.get(s1_id, set())
    for match_id in true_set:
        if match_id not in cand_set:
            missing.append((s1_id, match_id))

if missing:
    print(f"\nWARNING — {len(missing)} true match(es) NOT in candidates:")
    for s1_id, match_id in missing:
        print(f"  {s1_id} -> {match_id}")
else:
    print("\nAll true matches found in candidate sets. Smoke test PASSED.")
