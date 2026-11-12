"""Encode every Charades-STA test video ONCE at 5 FPS with the team's own
extract_candidates + CLIPEncoder. Resumable: one .npz per video."""
import os, sys, time, json
sys.path.insert(0, '/home/claude/fyp/backend')
import numpy as np, pandas as pd
from app.pipeline.frame_extraction import extract_candidates
from app.pipeline.clip_encoder import CLIPEncoder

ANN='/home/claude/datasets/charades_sta/test.parquet'
VID='/home/claude/datasets/charades_sta/videos'
OUT='/home/claude/work/emb'
deadline = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else 1e18
vids = sorted(pd.read_parquet(ANN)['video'].unique())
enc = CLIPEncoder.get(); enc.load()
done = skipped = 0
for v in vids:
    if time.time() > deadline: break
    out = os.path.join(OUT, v.replace('.mp4', '.npz'))
    if os.path.exists(out): done += 1; continue
    src = os.path.join(VID, v)
    if not os.path.exists(src): skipped += 1; continue
    try:
        t0 = time.perf_counter(); frames, info = extract_candidates(src, 5.0); t1 = time.perf_counter()
        emb = enc.encode_images([f.image for f in frames], batch_size=32); t2 = time.perf_counter()
        np.savez(out + '.tmp.npz', emb=emb.astype(np.float32),
                 ts=np.array([f.timestamp for f in frames], dtype=np.float32),
                 duration=info['video_duration_seconds'], extract_s=t1 - t0, clip_s=t2 - t1)
        os.replace(out + '.tmp.npz', out)
        done += 1
        if done % 25 == 0: print(f'{time.strftime("%H:%M:%S")} encoded {done}/{len(vids)}', flush=True)
    except Exception as e:
        print('FAIL', v, e, flush=True)
print(f'PASS END done={done} missing={skipped} total={len(vids)}', flush=True)
