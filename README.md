# Amazon ML Challenge 2026 — Business Entity Resolution
## Team 2135

### Requirements (Python 3.10+)

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
lightgbm==4.5.0
rapidfuzz==3.9.7
jellyfish==1.1.0
tqdm==4.66.5
scipy==1.13.1
joblib==1.4.2
```

Install:
```bash
pip install -r requirements.txt
```

### Reproduce Output

```bash
# 1. Data audit
python scripts/data_audit.py

# 2. Make val split
python scripts/make_val_split.py

# 3. Run full pipeline on train/val (blocking + features + model training)
python code/business_entity_resolution/src/train_model.py

# 4. Run full test inference
python code/business_entity_resolution/src/pipeline.py

# 5. Validate submission
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
