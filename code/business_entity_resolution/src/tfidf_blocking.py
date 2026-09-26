"""
tfidf_blocking.py — Sparse TF-IDF Candidate Generation

Replaces manual pandas joins with memory-efficient Sparse Matrix Multiplication.
This avoids the OOM limits, removing the 100-candidate cap, and allowing us 
to achieve near 100% recall for the 0.99 F0.5 score.
"""

import gc
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from normalize import preprocess_df

def get_text_for_tfidf(df: pd.DataFrame) -> list[str]:
    """Combines name and address components into a single document."""
    # df must be preprocessed first so that it has 'name', 'address', 'city'
    df = df.fillna("")
    texts = []
    for row in df.to_dict(orient='records'):
        # Name is weighted heavily, followed by city and address
        components = [
            str(row.get('_norm_name', '')), str(row.get('_norm_name', '')), # Duplicate name to give it 2x TF-IDF weight
            str(row.get('_norm_addr', '')),
            str(row.get('_city_tokens', ''))
        ]
        text = " ".join(c for c in components if c.strip()).lower()
        texts.append(text)
    return texts

def build_tfidf_candidates(s1_path: Path, s2_path: Path, s3_path: Path, output_path: Path, top_k: int = 50):
    print("=" * 60)
    print("TF-IDF SPARSE BLOCKING (HIGH RECALL)")
    print("=" * 60)
    
    # 1. Load S1
    print("Loading Source 1...")
    s1_df = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    s1_pre = preprocess_df(s1_df)
    s1_texts = get_text_for_tfidf(s1_pre)
    s1_ids = s1_df["entity_id"].tolist()
    
    # 2. Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer on S1 (Char N-Grams)...")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=500000)
    s1_matrix = vectorizer.fit_transform(s1_texts)
    
    # 3. Stream S2 & S3 into Sparse Matrices
    print("\nStreaming S2 and S3 into Sparse Matrices...")
    sx_matrices = []
    sx_ids = []
    
    def process_sx(filepath):
        print(f"  Transforming {filepath.name}...")
        chunk_size = 200000
        reader = pd.read_csv(filepath, sep="\t", chunksize=chunk_size, keep_default_na=False)
        for chunk in reader:
            chunk_pre = preprocess_df(chunk)
            texts = get_text_for_tfidf(chunk_pre)
            sparse_chunk = vectorizer.transform(texts)
            sx_matrices.append(sparse_chunk)
            sx_ids.extend(chunk["entity_id"].tolist())
            print(f"    Loaded {len(sx_ids):,} total candidates...", end="\r")
        print()

    process_sx(s2_path)
    process_sx(s3_path)
    
    print("\nStacking sparse matrices...")
    sx_matrix = vstack(sx_matrices)
    del sx_matrices
    gc.collect()
    
    print(f"Total Sx Candidates: {sx_matrix.shape[0]:,}")
    print(f"Feature Space (N-grams): {sx_matrix.shape[1]:,}")
    
    # 4. Batch Cosine Similarity Search
    print("\nComputing Cosine Similarity in batches to find top-K candidates...")
    candidate_pairs = {}
    
    batch_size = 2000  # Smaller batch to avoid RAM explosion during dense conversion
    
    for i in tqdm(range(0, s1_matrix.shape[0], batch_size), desc="Searching"):
        end_idx = min(i + batch_size, s1_matrix.shape[0])
        s1_batch = s1_matrix[i:end_idx]
        batch_ids = s1_ids[i:end_idx]
        
        # Dot product of sparse matrices is highly optimized
        # Shape: (batch_size, sx_total)
        sim_scores = s1_batch.dot(sx_matrix.T)
        
        # We need the top K indices per row. Since sim_scores is sparse, we can convert 
        # to dense row by row, or use argpartition. 
        # Better: loop over each row in the batch.
        for row_idx in range(sim_scores.shape[0]):
            row_sparse = sim_scores.getrow(row_idx)
            
            # If no matches found at all, skip
            if row_sparse.nnz == 0:
                continue
                
            # Get data and column indices
            scores = row_sparse.data
            indices = row_sparse.indices
            
            # Find top K
            if len(scores) > top_k:
                # np.argpartition is O(N) instead of O(N log N)
                top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
                best_indices = indices[top_k_idx]
            else:
                best_indices = indices
                
            matched_sx_ids = [sx_ids[idx] for idx in best_indices]
            if matched_sx_ids:
                candidate_pairs[batch_ids[row_idx]] = set(matched_sx_ids)
                
    # 5. Save Output
    print(f"\nWriting {len(candidate_pairs):,} S1 candidates to disk...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for s1_id, sx_set in candidate_pairs.items():
        rows.append(f"{s1_id}\t{','.join(sx_set)}")
        
    output_path.write_text("source1_entity_id\tcandidate_entity_ids\n" + "\n".join(rows), encoding="utf-8")
    
    total_pairs = sum(len(v) for v in candidate_pairs.values())
    print(f"Completed! Generated {total_pairs:,} highly precise candidate pairs.")
    return candidate_pairs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()
    
    build_tfidf_candidates(Path(args.s1), Path(args.s2), Path(args.s3), Path(args.out), args.k)
