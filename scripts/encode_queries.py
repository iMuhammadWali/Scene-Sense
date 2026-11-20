"""Encode all Charades-STA test captions once (single text projection, see sseval.data._encode_with_team) and write the cache
SSEVAL_CACHE/query_emb_fixed.npz (or query_emb_teamtext.npz with SSEVAL_TEAM_TEXT=1). Rows align with the parquet row order.
Usage: python scripts/encode_queries.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sseval import config as C, data
t0 = time.time()
df = data.load_annotations()
emb = data.load_query_emb(df)       # encodes + caches if the cache is missing/stale
assert emb.shape == (len(df), C.EMB_DIM) and np.isfinite(emb).all()
print(f"{len(df)} queries, {df['caption'].nunique()} unique captions; norm range "
      f"{np.linalg.norm(emb, axis=1).min():.4f}-{np.linalg.norm(emb, axis=1).max():.4f}")
print("cache:", C.QUERY_EMB_CACHE, f"{time.time()-t0:.0f}s")
