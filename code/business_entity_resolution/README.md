# Business Entity Resolution — Code Package
## Team 2135 | Amazon ML Challenge 2026

### Source Layout

```
src/
├── normalize.py      — Text normalization and abbreviation expansion
├── blocking.py       — Candidate generation (7 blocking strategies, unioned)
├── features.py       — Pairwise feature engineering (19 features)
├── train_model.py    — LightGBM trainer + F₀.₅ threshold tuning
├── predict.py        — Inference on test set
├── pipeline.py       — End-to-end orchestrator for test inference
└── eval_f05.py       — Evaluation metric (macro F₀.₅)
```

### Reproduce (from project root)

```bash
pip install -r code/business_entity_resolution/requirements.txt

# Audit data
python scripts/data_audit.py

# Create val split
python scripts/make_val_split.py

# Train model (blocking → features → LightGBM → threshold tuning)
python code/business_entity_resolution/src/train_model.py

# Generate test outputs
python code/business_entity_resolution/src/pipeline.py

# Validate
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### Model Details

- **Matching model:** LightGBM binary classifier (MIT license), CPU-only
- **Features:** 19 string/structural similarity features (no external model or API)
- **Blocking:** Union of 7 strategies (name bigrams, phonetic, postal code, city token, TF-IDF ANN, country-prefix, sparse fallback)
- **Threshold:** Tuned on validation split to maximize macro F₀.₅

### License Compliance

All libraries used: MIT or BSD-3 (permissive). No models with restricted licenses. No external data or API calls of any kind.
