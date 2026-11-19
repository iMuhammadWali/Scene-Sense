"""E10 - re-rank the fast path's top candidates with BLIP-ITM (no generative VLM, no API, no LLM).

Candidates: E8/E9 protocol on the PRUNED index (cos_last_drop tau 0.95, "C2"): top-20 FAISS hits -> team's group_temporally (gap 2 s) ->
groups ranked by best hit -> 5 candidates (padded with next-best peaks >= 3 s apart); candidate time = the group's peak CLIP frame.
Same 300 positive + 100 negative query sample as E8/E9 (seed 8). Candidate 0..4 are in CLIP order (the first 5 of a 10-candidate list
are identical to the 5-candidate list; verified against E8's per_query.csv cand_times).
Scorer: Salesforce/blip-itm-base-coco, BlipForImageTextRetrieval ITM head -> P(match) and logit (match - no-match) for (frame, query).
Frames are decoded from the raw video with OpenCV at the needed timestamps (peak frame; K=3: peak +-1 s).

Stages (each resumable; scores are cached per query in <out>/raw/<kind>_<qid>.json under key "res|variant|cand|offset"):
  s1  384 px, candidates 0-4, offset 0 (K=1), all queries            [main result]
  s2  384 px, candidates 0-4, offsets -1,+1 (-> K=3), first 150 pos + 50 neg
  s3  288 px, candidates 0-4, offset 0, first 150 pos + 50 neg       [resolution ablation]
  s4  384 px, candidates 5-9, offset 0, first 150 pos                [top-10 re-ranking]
Both query variants ("asis", "strip" = leading person / a person removed) are scored from the same image embeddings.
Usage:  run.py [--stages s1,s2,s3,s4] [--limit N] [--aggregate-only] [--timing]
"""
from __future__ import annotations
import sys, os, json, time, importlib.util, argparse
from pathlib import Path
os.environ.setdefault("HF_HOME", "/home/claude/work/models/hf_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd

from sseval import config as C, data, runutil, rerank as R

EXP = "e10_blip_rerank"
MODEL_ID = "Salesforce/blip-itm-base-coco"
N_CAND_MAX = 10
SEED = 8
METHOD, PARAMS = "cos_last_drop", {"tau": 0.95}
VARIANTS = ("asis", "strip")
STAGES = {                                    # name: (res, cand range, offsets, n_pos, n_neg)
    "s1": (384, range(0, 5), (0.0,), 300, 100),
    "s2": (384, range(0, 5), (-1.0, 1.0), 150, 50),
    "s3": (288, range(0, 5), (0.0,), 150, 50),
    "s4": (384, range(5, 10), (0.0,), 150, 0),
}

def load_e8():
    spec = importlib.util.spec_from_file_location("e8_run", Path(__file__).resolve().parents[2] / "experiments" / "e8_slowpath" / "run.py")
    m = importlib.util.module_from_spec(spec); sys.modules["e8_run"] = m; spec.loader.exec_module(m)
    return m

# ----------------------------------------------------------------------------------------------------------------- candidates
def build_items(limit: int | None = None):
    """-> (e8 module, items (pos then neg), per-query plan dict qid -> dict(q, cands (<=10), dur))."""
    e8 = load_e8()
    e8.LETTERS = "ABCDEFGHIJ"        # build_candidates labels candidates (E8 only needed 5)
    from sseval.pruning import select
    from sseval.retrieval import build_index, search
    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df))
    items = e8.sample_queries(df, QE, list(vids), SEED)
    if limit:
        items = [x for x in items if x["kind"] == "pos"][:limit] + [x for x in items if x["kind"] == "neg"][:max(1, limit // 3)]
    plans = {}
    for q in items:
        emb, ts, dur = vids[q["vid"]]
        sel = select(METHOD, emb, ts, dur, **PARAMS)
        index, mapping = build_index(sel)
        qe = q["qe"] / max(np.linalg.norm(q["qe"]), 1e-12)
        hits = search(index, qe, mapping, e8.TOP_HITS)
        cands = e8.build_candidates(sel, hits, sel.emb @ qe, dur, n=N_CAND_MAX)
        plans[(q["kind"], q["qid"])] = dict(q=q, cands=cands, dur=float(dur))
    return e8, items, plans

# ----------------------------------------------------------------------------------------------------------------- frames
def decode_frames(video: Path, times: list[float]):
    """RGB uint8 frames (full resolution) at each timestamp (with E8's seek back-off if a read fails)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    out = []
    try:
        for t in times:
            img = None
            for back in (0.0, 0.2, 0.5, 1.0):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - back) * 1000.0)
                ok, fr = cap.read()
                if ok and fr is not None:
                    img = fr; break
            if img is None:
                raise RuntimeError(f"cannot decode {video.name} at {t}s")
            out.append(img[:, :, ::-1].copy())
    finally:
        cap.release()
    return out

# ----------------------------------------------------------------------------------------------------------------- BLIP
class Blip:
    def __init__(self, threads: int = 2):
        import torch
        from transformers import BlipProcessor, BlipForImageTextRetrieval
        torch.set_num_threads(threads)
        self.torch = torch
        self.proc = BlipProcessor.from_pretrained(MODEL_ID)
        self.model = BlipForImageTextRetrieval.from_pretrained(MODEL_ID).eval()

    def score(self, frames, texts: list[str], res: int = 384):
        """frames: list of RGB arrays (B); texts: query strings (one per variant). One vision pass shared by all texts.
        -> (logit [n_texts, B], prob [n_texts, B], t_vision_s, t_text_s). logit = itm_score[match] - itm_score[no match]."""
        torch = self.torch
        kw = {} if res == 384 else dict(size={"height": res, "width": res})
        t0 = time.perf_counter()
        pix = self.proc.image_processor(images=list(frames), return_tensors="pt", **kw)["pixel_values"]
        with torch.no_grad():
            emb = self.model.vision_model(pixel_values=pix, interpolate_pos_encoding=(res != 384)).last_hidden_state
            t1 = time.perf_counter()
            att = torch.ones(emb.shape[:-1], dtype=torch.long)
            lg = []
            for t in texts:
                tok = self.proc.tokenizer([t] * len(frames), return_tensors="pt", padding=True, truncation=True, max_length=35)
                q = self.model.text_encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                                            encoder_hidden_states=emb, encoder_attention_mask=att).last_hidden_state
                s = self.model.itm_head(q[:, 0, :])
                lg.append((s[:, 1] - s[:, 0]).numpy().astype(np.float64))
        t2 = time.perf_counter()
        lg = np.stack(lg)
        return lg, 1.0 / (1.0 + np.exp(-lg)), t1 - t0, t2 - t1

# ----------------------------------------------------------------------------------------------------------------- caching / running
def key(res, variant, cand, off) -> str:
    return f"{res}|{variant}|{cand}|{off:+.0f}"

def raw_path(out: Path, kind: str, qid: int) -> Path:
    return out / "raw" / f"{kind}_{qid}.json"

def load_raw(out: Path, kind: str, qid: int) -> dict:
    p = raw_path(out, kind, qid)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"scores": {}, "timing": {}}

def save_raw(out: Path, kind: str, qid: int, rec: dict):
    p = raw_path(out, kind, qid); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); tmp.rename(p)

def stage_queries(items, n_pos, n_neg):
    return [x for x in items if x["kind"] == "pos"][:n_pos] + [x for x in items if x["kind"] == "neg"][:n_neg]

def run_stage(name, blip, items, plans, out: Path):
    res, crange, offsets, n_pos, n_neg = STAGES[name]
    qs = stage_queries(items, n_pos, n_neg)
    t0, done, ran = time.time(), 0, 0
    for q in qs:
        pl = plans[(q["kind"], q["qid"])]
        rec = load_raw(out, q["kind"], q["qid"])
        need = [(c, o) for c in crange if c < len(pl["cands"]) for o in offsets
                if any(key(res, v, c, o) not in rec["scores"] for v in VARIANTS)]
        done += 1
        if not need:
            continue
        times = [R.frame_times(pl["cands"][c]["t"], pl["dur"], (o,))[0] for c, o in need]
        td = time.perf_counter()
        frames = decode_frames(C.VIDEOS_DIR / f"{q['vid']}.mp4", times)
        t_dec = time.perf_counter() - td
        texts = [q["caption"], R.strip_person(q["caption"])]
        lg, pr, tv, tt = blip.score(frames, texts, res)
        for vi, v in enumerate(VARIANTS):
            for j, (c, o) in enumerate(need):
                rec["scores"][key(res, v, c, o)] = [float(lg[vi, j]), float(pr[vi, j])]
        rec["timing"][f"{name}"] = dict(n_img=len(frames), decode_s=t_dec, vision_s=tv, text_s=tt)
        rec["vid"], rec["caption"], rec["cand_t"] = q["vid"], q["caption"], [c["t"] for c in pl["cands"]]
        save_raw(out, q["kind"], q["qid"], rec); ran += 1
        if ran % 10 == 0:
            el = time.time() - t0
            print(f"[{name}] {done}/{len(qs)} queries ({ran} computed) {el:.0f}s; last: vision {tv:.2f}s text {tt:.2f}s decode {t_dec:.2f}s for {len(frames)} imgs", flush=True)
    print(f"[{name}] finished: {ran} queries computed, {len(qs) - ran} cached, {time.time() - t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------------------------------------------- timing
def run_timing(blip, items, plans, out: Path, n: int = 30):
    """Dedicated timing on real queries: single text variant, 2 threads, no other work. Per query: decode, ITM for K=1 (5 frames, one batch)
    and K=3 (15 frames, one batch of 15 and three batches of 5), at 384 and 288 px."""
    qs = [x for x in items if x["kind"] == "pos"][:n]
    rows = []
    blip.score(decode_frames(C.VIDEOS_DIR / f"{qs[0]['vid']}.mp4", [1.0] * 5), [qs[0]["caption"]], 384)      # warm-up
    for q in qs:
        pl = plans[("pos", q["qid"])]
        c5 = pl["cands"][:5]
        for K, offs in ((1, (0.0,)), (3, (-1.0, 0.0, 1.0))):
            times = [R.frame_times(c["t"], pl["dur"], offs) for c in c5]
            times = [t for ts_ in times for t in ts_]
            td = time.perf_counter(); fr = decode_frames(C.VIDEOS_DIR / f"{q['vid']}.mp4", times); t_dec = time.perf_counter() - td
            for res in (384, 288):
                t0 = time.perf_counter()
                for i in range(0, len(fr), 5):                       # batches of 5 images
                    blip.score(fr[i:i + 5], [q["caption"]], res)
                t_itm = time.perf_counter() - t0
                rows.append(dict(qid=q["qid"], K=K, res=res, n_img=len(fr), decode_s=t_dec, itm_s=t_itm, per_pair_s=t_itm / len(fr)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "timing.csv", index=False)
    print(df.groupby(["K", "res"])[["decode_s", "itm_s", "per_pair_s"]].mean().round(3).to_string(), flush=True)

def main():
    ap = argparse.ArgumentParser(description="E10 BLIP-ITM re-ranking of fast-path candidates")
    ap.add_argument("--stages", default="s1,s2,s3,s4")
    ap.add_argument("--limit", type=int, default=None, help="N positives (and N//3 negatives) per stage, for a quick run")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else C.results_dir(EXP)
    out.mkdir(parents=True, exist_ok=True)
    e8, items, plans = build_items(a.limit)
    print(f"[e10] {sum(i['kind'] == 'pos' for i in items)} pos / {sum(i['kind'] == 'neg' for i in items)} neg queries; out={out}", flush=True)
    if not a.aggregate_only:
        blip = Blip()
        if a.timing:
            run_timing(blip, items, plans, out); return
        for s in a.stages.split(","):
            run_stage(s, blip, items, plans, out)
    runutil.write_meta(out, limit=a.limit, model=MODEL_ID, seed=SEED, stages=a.stages)
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).parent))
    import_module("aggregate").aggregate(out, items, plans)

if __name__ == "__main__":
    main()
