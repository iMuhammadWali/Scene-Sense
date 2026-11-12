"""Encode all Charades-STA test captions once with the team's CLIPEncoder.encode_text.
Output: cache/query_emb.npz  (emb[row_index] = L2-normalised 512-d; rows align with the parquet row order)."""
import time
from common import *
from app.pipeline.clip_encoder import CLIPEncoder
df = load_queries()
enc = CLIPEncoder.get(); enc.load()
uniq = sorted(df["caption"].unique()); print(f"{len(df)} queries, {len(uniq)} unique captions", flush=True)
cache = {}; t0 = time.time()
for i, c in enumerate(uniq):
    cache[c] = enc.encode_text(c)
    if (i + 1) % 500 == 0: print(f"  {i+1}/{len(uniq)} {time.time()-t0:.0f}s", flush=True)
emb = np.stack([cache[c] for c in df["caption"]]).astype(np.float32)
assert emb.shape == (len(df), 512) and np.isfinite(emb).all()
print("norm range", np.linalg.norm(emb, axis=1).min(), np.linalg.norm(emb, axis=1).max())
np.savez(QCACHE, emb=emb, caption=df["caption"].values)
print("saved", QCACHE, emb.shape, f"{time.time()-t0:.0f}s")
