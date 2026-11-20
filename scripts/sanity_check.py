"""Post-setup sanity check. Usage: python scripts/sanity_check.py [n_videos=20]
Loads (or encodes) frame embeddings for N videos, encodes their queries with the app's CLIPEncoder.encode_text, and reports
  * mean max image-text cosine   (expect ~0.28-0.32; ~0.05-0.10 means the double text-projection bug)
  * Hit@1 of the top frame       (expect roughly 40-55% on a small sample; ~25% is chance)
plus library versions. Needs Charades-STA annotations + videos (see docs/DATASET.md)."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sseval import config as C
C.setup_team()
import numpy as np
import torch, transformers, faiss
from sseval import data

print(f"torch {torch.__version__} | transformers {transformers.__version__} | faiss {faiss.__version__} | python {sys.version.split()[0]}")
if not transformers.__version__.startswith("4.57."):
    print(f"WARNING: transformers {transformers.__version__} is not 4.57.x - numbers may differ from the paper. pip install 'transformers==4.57.6'")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
df = data.load_annotations()
vids = sorted(df["vid"].unique())
have = data.list_available_videos(df)[:n]
if len(have) < n:                                   # encode the missing ones (resumable, same code as scripts/encode_all.py)
    from app.pipeline.frame_extraction import extract_candidates
    from app.pipeline.clip_encoder import CLIPEncoder
    import os
    enc = CLIPEncoder.get(); enc.load(); C.EMB_DIR.mkdir(parents=True, exist_ok=True)
    for v in vids:
        if len(have) >= n: break
        if v in have: continue
        src = C.VIDEOS_DIR / f"{v}.mp4"
        if not src.exists(): continue
        t0 = time.time(); frames, info = extract_candidates(str(src), C.FPS)
        emb = enc.encode_images([f.image for f in frames], batch_size=32)
        out = C.EMB_DIR / f"{v}.npz"; tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, emb=emb.astype(np.float32), ts=np.array([f.timestamp for f in frames], dtype=np.float32),
                 duration=info["video_duration_seconds"])
        os.replace(tmp, out); have = sorted(set(have) | {v}); have = have[:n] if len(have) > n else have
        print(f"encoded {v} ({len(frames)} frames, {time.time()-t0:.1f}s)")
    have = data.list_available_videos(df)[:n]
if not have:
    sys.exit(f"No videos found. Check SSEVAL_VIDEOS={C.VIDEOS_DIR} and SSEVAL_ANN={C.ANNOTATIONS}")

from app.pipeline.clip_encoder import CLIPEncoder
enc = CLIPEncoder.get(); enc.load()
sub = df[df["vid"].isin(have)]
maxcos, hits = [], []
for vid, g in sub.groupby("vid"):
    emb, ts, _ = data.load_video(vid)
    for _, r in g.iterrows():
        q = enc.encode_text(r["caption"]).astype(np.float32); q /= np.linalg.norm(q)
        sims = emb @ q
        maxcos.append(float(sims.max()))
        t = float(ts[int(sims.argmax())]); hits.append(r["gs"] <= t <= r["ge"])
mc, h1 = float(np.mean(maxcos)), 100 * float(np.mean(hits))
print(f"{len(have)} videos, {len(maxcos)} queries")
print(f"mean max image-text cosine: {mc:.3f}   (expect ~0.28-0.32)")
print(f"Hit@1 of top frame:         {h1:.1f}%   (expect ~40-55%; chance ~25%)")
EXP = json.loads((Path(__file__).with_name("expected_numbers.json")).read_text(encoding="utf-8"))["small_sample_sanity_check"]
lo, hi = EXP["mean_max_cosine_range"]
ok_cos, ok_hit = lo <= mc <= hi, h1 >= EXP["hit1_pct_min"]
print(f"cosine in [{lo}, {hi}]: {'PASS' if ok_cos else 'FAIL'} | Hit@1 >= {EXP['hit1_pct_min']}% (chance ~{EXP['chance_hit1_pct']}%): {'PASS' if ok_hit else 'FAIL'}")
print("OVERALL:", "PASS" if ok_cos and ok_hit else "FAIL")
if mc < 0.15:
    print("WARNING: cosine is far too low - the app's CLIPEncoder.encode_text is probably double-projecting (old encoder?).")
sys.exit(0 if ok_cos and ok_hit else 1)
