"""Encode every Charades-STA test video ONCE at 5 FPS with the team's own extract_candidates + CLIPEncoder.
Resumable: one .npz per video in SSEVAL_EMB (default data/emb).
Usage: python scripts/encode_all.py [time_budget_seconds]"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sseval import config as C
C.setup_team()
import numpy as np, pandas as pd
from app.pipeline.frame_extraction import extract_candidates
from app.pipeline.clip_encoder import CLIPEncoder

OUT, VID = C.EMB_DIR, C.VIDEOS_DIR
OUT.mkdir(parents=True, exist_ok=True)
deadline = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else 1e18
vids = sorted(pd.read_parquet(C.ANNOTATIONS)['video'].unique())
enc = CLIPEncoder.get(); enc.load()
done = skipped = 0
for v in vids:
    if time.time() > deadline: break
    out = OUT / v.replace('.mp4', '.npz')
    if out.exists(): done += 1; continue
    src = VID / v
    if not src.exists(): skipped += 1; continue
    try:
        t0 = time.perf_counter(); frames, info = extract_candidates(str(src), 5.0); t1 = time.perf_counter()
        emb = enc.encode_images([f.image for f in frames], batch_size=32); t2 = time.perf_counter()
        tmp = out.with_name(out.stem + '.tmp.npz')
        np.savez(tmp, emb=emb.astype(np.float32),
                 ts=np.array([f.timestamp for f in frames], dtype=np.float32),
                 duration=info['video_duration_seconds'], extract_s=t1 - t0, clip_s=t2 - t1)
        os.replace(tmp, out)
        done += 1
        if done % 25 == 0: print(f'{time.strftime("%H:%M:%S")} encoded {done}/{len(vids)}', flush=True)
    except Exception as e:
        print('FAIL', v, e, flush=True)
print(f'PASS END done={done} missing={skipped} total={len(vids)}', flush=True)
