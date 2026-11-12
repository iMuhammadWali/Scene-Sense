"""E6 - where the time goes: run the team's REAL VideoProcessor + QueryPipeline on a sample of Charades videos/queries.
Usage: python -W ignore experiments/e6_latency/run.py [--limit N_VIDEOS] [--max-queries Q_PER_VIDEO] [--seed S] [--force]
No OpenRouter key: COMPLEX-routed queries fail at the VLM stage; the stages completed before the failure are kept, the failure is
recorded. Sessions are written to a scratch data dir (not the shared one) and removed afterwards.
Timings are wall-clock on a shared 2-core CPU (record loadavg); the first video / first query are warm-ups reported separately."""
import os, sys, time, threading, shutil
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
# scratch session store; must be set BEFORE the team's settings are first read
_results = Path(os.environ.get("SSEVAL_RESULTS_DIR", REPO / "results"))
SCRATCH = Path(os.environ.get("SSEVAL_E6_DATA", "/home/claude/work/e6_data"))
os.environ["SCENESENSE_DATA_DIR"] = str(SCRATCH)
import numpy as np, pandas as pd, psutil
from sseval import config as C, data, runutil
C.setup_team()
from app.config import get_settings
from app.pipeline.clip_encoder import CLIPEncoder
from app.pipeline.classifier import QueryClassifier
from app.pipeline.query_pipeline import QueryPipeline
from app.pipeline.video_processor import VideoProcessor
from app.services.upload import UploadService
from app.storage.session import SessionStore

class PeakRSS:
    """Samples process RSS in a background thread; .peak_mb is the max seen between enter and exit."""
    def __init__(self, every=0.05): self.p = psutil.Process(); self.every = every; self.peak = 0
    def __enter__(self):
        self.peak = self.p.memory_info().rss; self._stop = False
        def loop():
            while not self._stop:
                self.peak = max(self.peak, self.p.memory_info().rss); time.sleep(self.every)
        self.t = threading.Thread(target=loop, daemon=True); self.t.start(); return self
    def __exit__(self, *a): self._stop = True; self.t.join(); self.peak = max(self.peak, self.p.memory_info().rss)
    @property
    def peak_mb(self): return self.peak / 1e6

def rss_mb(): return psutil.Process().memory_info().rss / 1e6

def upload(vid, store):
    src = C.VIDEOS_DIR / f"{vid}.mp4"
    t0 = time.perf_counter()
    with src.open("rb") as f:
        res = UploadService(store).save_upload(src.name, f, size_hint=src.stat().st_size)
    return res["session_id"], time.perf_counter() - t0, res["video_metadata"]

def process(sid, store):
    with PeakRSS() as pk:
        t0 = time.perf_counter(); meta = VideoProcessor(store).process(sid); total = time.perf_counter() - t0
    stages = {s["stage"]: float(s.get("processing_time_seconds", np.nan)) for s in store.read_json(sid, "processing_log.json")["stages"]
              if s.get("stage") in ("frame_extraction", "clip_encoding", "frame_pruning", "visual_memory", "faiss_index")}
    return total, stages, pk.peak_mb, meta

def ask(sid, q, store):
    err = None
    with PeakRSS() as pk:
        t0 = time.perf_counter()
        try:
            res = QueryPipeline(store).ask(sid, q)
        except Exception as e:                                  # COMPLEX path without an OpenRouter key
            err = f"{type(e).__name__}: {str(e)[:160]}"; res = None
        total = time.perf_counter() - t0
    prog = store.get_query_progress(sid)
    stages = {s["stage"]: float(s["processing_time_seconds"]) for s in prog.get("stages", []) if s.get("status") == "completed"
              and "processing_time_seconds" in s}
    routed = prog.get("query_type")
    return total, stages, routed, err, pk.peak_mb, res

def main():
    ap = runutil.base_parser("E6 latency"); ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--seed", type=int, default=C.SEED); ap.add_argument("--keep-sessions", action="store_true"); a = ap.parse_args()
    n_vid = a.limit or 10
    out = C.results_dir("e6_latency")
    if runutil.is_done(out, a.force): print("[e6] DONE exists, skipping"); return
    df = data.load_annotations()
    avail = [v for v in data.list_available_videos(df) if (C.VIDEOS_DIR / f"{v}.mp4").exists()]
    order = [avail[i] for i in np.random.default_rng(a.seed).permutation(len(avail))]
    sample, warm_vid = order[:n_vid], order[n_vid]
    by_vid = {v: g for v, g in df.groupby("vid")}
    print(f"[e6] sample {sample}, warm-up video {warm_vid}; loadavg {os.getloadavg()}", flush=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    settings = get_settings(); store = SessionStore(settings)
    res_rows = [dict(item="rss_at_start_mb", value=rss_mb())]
    # --- model load times, measured separately (cold)
    t0 = time.perf_counter(); CLIPEncoder.get().load(); clip_load = time.perf_counter() - t0
    t0 = time.perf_counter(); QueryClassifier.get().load(); clf_load = time.perf_counter() - t0
    res_rows += [dict(item="clip_load_s", value=clip_load), dict(item="classifier_load_s", value=clf_load), dict(item="rss_after_load_mb", value=rss_mb())]
    print(f"[e6] model load: clip {clip_load:.1f}s, classifier {clf_load:.1f}s", flush=True)

    runs = []   # one row per (video process) or (query)
    def do_video(vid, warm):
        sid, up_s, vmeta = upload(vid, store)
        total, stg, peak, meta = process(sid, store)
        row = dict(kind="video", video=vid, warmup=warm, session=sid, total_s=total, upload_copy_s=up_s, peak_rss_mb=peak,
                   duration_s=vmeta.get("duration_seconds"), n_candidates=meta.get("candidate_frame_count"),
                   n_retained=meta.get("retained_frame_count"), **{f"s_{k}": v for k, v in stg.items()})
        runs.append(row); print(f"[e6] video {vid} {'(warm-up)' if warm else ''}: {total:.1f}s, peak {peak:.0f} MB", flush=True)
        return sid
    # warm-up: one throwaway video + one query (first-call effects: allocator, lazy imports, thread pools)
    wsid = do_video(warm_vid, True)
    wq = by_vid[warm_vid].iloc[0].caption
    tot, stg, routed, err, peak, _ = ask(wsid, wq, store)
    runs.append(dict(kind="query", video=warm_vid, warmup=True, query=wq, routed=routed, error=err, total_s=tot, peak_rss_mb=peak,
                     **{f"s_{k}": v for k, v in stg.items()}))
    for vid in sample:
        sid = do_video(vid, False)
        qs = by_vid[vid].caption.tolist()[: a.max_queries]
        for q in qs:
            tot, stg, routed, err, peak, _ = ask(sid, q, store)
            runs.append(dict(kind="query", video=vid, warmup=False, query=q, routed=routed, error=err, total_s=tot, peak_rss_mb=peak,
                             **{f"s_{k}": v for k, v in stg.items()}))
            print(f"[e6]   q routed={routed} {tot:.2f}s err={'yes' if err else 'no'}: {q[:50]}", flush=True)
    res_rows.append(dict(item="rss_end_mb", value=rss_mb()))
    res_rows.append(dict(item="ru_maxrss_mb", value=__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1e3))
    R = pd.DataFrame(runs)
    R.to_csv(out / "stage_timings.csv", index=False)

    # --- summaries (warm-ups excluded)
    W = R[~R.warmup]
    sm = []
    for kind in ("video", "query"):
        g = W[W.kind == kind]
        cols = [c for c in g.columns if c.startswith("s_")] + ["total_s"]
        for c in cols:
            x = g[c].dropna()
            if len(x):
                sm.append(dict(phase="preprocess (per video)" if kind == "video" else "query", stage=c[2:] if c.startswith("s_") else "TOTAL (wall)",
                               n=len(x), mean_s=x.mean(), median_s=x.median(), std_s=x.std(ddof=0), min_s=x.min(), max_s=x.max()))
    for kind in ("video", "query"):
        for c in [c for c in R.columns if c.startswith("s_")] + ["total_s"]:
            x = R[(R.kind == kind) & R.warmup][c].dropna()
            if len(x):
                sm.append(dict(phase=f"warm-up {kind}", stage=c[2:] if c.startswith("s_") else "TOTAL (wall)", n=len(x), mean_s=x.mean(),
                               median_s=x.median(), std_s=float("nan"), min_s=x.min(), max_s=x.max()))
    for nm, v in (("clip_load", clip_load), ("classifier_load", clf_load)):
        sm.append(dict(phase="model load (cold, once)", stage=nm, n=1, mean_s=v, median_s=v, std_s=float("nan"), min_s=v, max_s=v))
    pd.DataFrame(sm).to_csv(out / "stage_summary.csv", index=False)

    Q = W[W.kind == "query"]
    routing = Q.routed.fillna("unknown").value_counts().rename_axis("route").reset_index(name="n")
    routing["pct"] = 100 * routing.n / routing.n.sum()
    routing["failed_at_vlm"] = routing.route.map(lambda r: int(Q[(Q.routed.fillna("unknown") == r) & Q.error.notna()].shape[0]))
    routing.to_csv(out / "routing_summary.csv", index=False)
    res_rows += [dict(item="video_peak_rss_mb_max", value=W[W.kind == "video"].peak_rss_mb.max()),
                 dict(item="query_peak_rss_mb_max", value=Q.peak_rss_mb.max())]
    pd.DataFrame(res_rows).to_csv(out / "resources.csv", index=False)
    runutil.write_meta(out, n_videos=len(sample), n_queries=len(Q), seed=a.seed, note=
        "shared 2-core CPU; background CLIP encoding job may have been running (see hardware.loadavg); COMPLEX queries fail at the VLM stage (no OpenRouter key)",
        text_encoder_in_pipeline="team CLIPEncoder.encode_text as-is (see report: double text projection under transformers 4.57; affects retrieval content, not timings)")
    if not a.keep_sessions:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    runutil.mark_done(out)
    print(pd.DataFrame(sm).round(3).to_string(index=False)); print(routing.to_string(index=False)); print(pd.DataFrame(res_rows).round(1).to_string(index=False))

if __name__ == "__main__":
    main()
