# Amazon ML Challenge: Business Entity Resolution
Team: Team 2135

This repository contains our end-to-end entity resolution pipeline for the Amazon ML Challenge. The pipeline is heavily optimized to run locally on a machine with 8GB VRAM (though the final architecture avoids GPU usage completely to maximize RAM safety and throughput), processing over 12 million noisy business records using a strict **single-pass streaming architecture**.

## Architecture & Pipeline

The pipeline is broken down into modular phases:

1. **Phase 1: Normalization** (`src/normalize.py`)
   - Text standardization, stop-word removal, and phonetic `Soundex` encoding.
2. **Phase 2: Streaming Blocking** (`src/blocking.py`)
   - Reads the massive 10.3M Source 2/Source 3 rows precisely *once* in 150k chunks.
   - Generates multi-index blocking keys (Name tokens, Phonetic, Postal, House Number, Script) to build an inverted candidate index.
   - Safe heuristics: A hard cap of 100 maximum candidates per S1 entity prevents cross-product memory explosions while maintaining high recall (approx ~60% due to aggressive candidate capping on 16GB RAM constraints).
3. **Phase 3: Feature Engineering** (`src/features.py`)
   - Computes a highly discriminative 20-dimensional feature matrix using `rapidfuzz` (C++ backend) for all generated candidate pairs.
   - Features include Jaro-Winkler, Levenshtein, Token Sort/Set ratios, and Address specific Jaccard indices.
4. **Phase 4: Modeling** (`src/train_model.py`)
   - A `LightGBM` binary classifier trained to predict match probabilities.
   - Tunes the decision threshold to strictly maximize the **F0.5** score (Precision-heavy metric).
5. **Phase 5: Inference** (`src/pipeline.py`)
   - End-to-end wrapper script that streams the test set through blocking, feature extraction, and prediction to generate the final `submission.csv`.

## Setup
```bash
pip install -r requirements.txt
```

## Running the Pipeline

To generate the final predictions on the test set, use the `pipeline.py` script. The trained LightGBM model and tuned F0.5 threshold are already included in the `output/` directory.

```bash
# Ensure UTF-8 mode on Windows for cp1252 character issues
$env:PYTHONUTF8=1

# Run the inference pipeline
python code/business_entity_resolution/src/pipeline.py \
    --s1 dataset/test/test_source1.tsv \
    --s2 dataset/test/test_source2.tsv \
    --s3 dataset/test/test_source3.tsv \
    --model output/lgbm_model.txt \
    --threshold output/threshold.txt \
    --out output/submission.csv
```

## Performance
- **Validation F0.5 Score:** `0.8955` (at threshold `0.90`)
- **Memory Footprint:** Peak RAM ~2GB during streaming.
- **Top Features:** Name Jaro-Winkler distance, Address Token Jaccard index, and Name Token Set Ratio.

## Constraints Adhered
- Completely offline: No external APIs, internet searches, or geocoding used.
- Local hardware: Designed specifically for typical local environments by entirely bypassing heavy all-at-once Pandas merges.
