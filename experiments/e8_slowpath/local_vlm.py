"""E8 local backend: the SAME candidates / prompt / parser / metrics as run.py, with a local SmolVLM2 on CPU instead of the DeepSeek API.

Results go to results/e8_slowpath/local_<model>/ (raw/<cond>_<qid>.json per call -> resumable; per_query.csv; summary.csv; compare.csv).
Usage:
  python -W ignore experiments/e8_slowpath/local_vlm.py --model 500M --conds C1K3,C1K1 --n-pos 5 --n-neg 2     # timing trial
  python -W ignore experiments/e8_slowpath/local_vlm.py --model 2.2B --conds C1K1 --n-pos 100 --n-neg 50 --time-budget-min 55
  python -W ignore experiments/e8_slowpath/local_vlm.py --model 500M --conds C1K1 --aggregate-only
Queries are processed in sample order, 2 positives then 1 negative, so any prefix is a balanced random subset (same qids as DeepSeek C1).
"""
import sys, os, json, time, importlib.util, argparse
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(os.environ.get("SCENESENSE_MODELS_DIR", Path(__file__).resolve().parents[2] / "data" / "models")) / "hf_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
spec = importlib.util.spec_from_file_location("e8_run", Path(__file__).with_name("run.py"))
e8 = importlib.util.module_from_spec(spec); sys.modules["e8_run"] = e8; spec.loader.exec_module(e8)
import numpy as np, pandas as pd
from sseval import config as C, data
from sseval.metrics import bootstrap_indices, bootstrap_ci

MODELS = {"500M": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}
MAX_NEW = 32

def load_model(name: str, longest_edge: int):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(2)
    mid = MODELS[name]
    proc = AutoProcessor.from_pretrained(mid, do_image_splitting=False, size={"longest_edge": longest_edge})
    model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.bfloat16).eval()
    return proc, model

def run_local(proc, model, parts, vid, suffix=None):
    """-> dict(text, input_tokens, output_tokens, latency_s). Images: the same 448 px q80 JPEGs the API run uses; the processor then
    resizes to the configured longest edge (no image splitting)."""
    import torch, io
    from PIL import Image
    content, jpgs = e8.materialize(parts, vid)
    pil = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpgs]
    msg_c = [{"type": "text", "text": p["text"]} if p["type"] == "text" else {"type": "image"} for p in parts]
    if suffix:
        msg_c.append({"type": "text", "text": suffix})
    prompt = proc.apply_chat_template([{"role": "user", "content": msg_c}], add_generation_prompt=True)
    inp = proc(text=prompt, images=[pil], return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
    lat = time.perf_counter() - t0
    n_in = int(inp["input_ids"].shape[1])
    text = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    return dict(text=text, input_tokens=n_in, output_tokens=int(out.shape[1] - n_in), latency_s=lat, n_images=len(pil))

def one_call(proc, model, plan, cname, outdir: Path, meta: dict):
    q = plan["q"]; n_cand = len(plan["cands"])
    calls = []
    r = run_local(proc, model, plan["parts"], q["vid"])
    json_ok = e8.parse_response(r["text"], n_cand)[2] == "ok"           # strict (JSON as instructed)
    ch, cf, st = e8.parse_response(r["text"], n_cand, lenient=True); calls.append((r, ch, cf, st))
    if st != "ok":
        r2 = run_local(proc, model, plan["parts"], q["vid"], suffix=e8.RETRY_SUFFIX)
        ch, cf, st = e8.parse_response(r2["text"], n_cand, lenient=True); calls.append((r2, ch, cf, st))
    first_ok = calls[0][3] == "ok"
    rec = dict(task=f"{cname}_{q['qid']}", cond=cname, qid=q["qid"], kind=q["kind"], vid=q["vid"], req_sha=plan["sha"], K=plan["K"],
               n_images=calls[0][0]["n_images"], cands=[dict(label=c["label"], t=c["t"]) for c in plan["cands"]], choice=ch, confidence=cf,
               status=st, first_try_ok=first_ok, first_try_json=json_ok, n_calls=len(calls), input_tokens=sum(c[0]["input_tokens"] for c in calls),
               output_tokens=sum(c[0]["output_tokens"] for c in calls), latency_s=sum(c[0]["latency_s"] for c in calls),
               cost_actual_usd=0.0, cost_peak_usd=0.0, local=meta,
               calls=[dict(text=c[0]["text"], latency_s=c[0]["latency_s"], tier="local", choice=c[1], status=c[3]) for c in calls])
    p = e8.raw_path(outdir, cname, q["qid"]); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=1, default=lambda o: None if o != o else str(o)), encoding="utf-8"); tmp.replace(p)
    return rec

def paired(a: np.ndarray, b: np.ndarray):
    idx = bootstrap_indices(len(a)); d = a.astype(float) - b.astype(float)
    lo, hi = bootstrap_ci(d, idx); return d.mean(), lo, hi

def compare(outdir: Path, pq: pd.DataFrame, ds: pd.DataFrame, model: str):
    """Local VLM vs fast path vs DeepSeek on the SAME queries (paired)."""
    rows = []
    for cname in sorted(pq.cond.unique()):
        L = pq[(pq.cond == cname) & (pq.kind == "pos")].set_index("qid")
        D = ds[(ds.cond == cname) & (ds.kind == "pos")].set_index("qid")
        common = L.index.intersection(D.index).sort_values()
        if not len(common): continue
        l, d = L.loc[common], D.loc[common]
        ln = pq[(pq.cond == cname) & (pq.kind == "neg")]; dn = ds[(ds.cond == cname) & (ds.kind == "neg") & ds.qid.isin(ln.qid)]
        for metric, col, thr in (("hit1", "hit1", None), ("r1_iou05_wide", "iou_wide", 0.5)):
            f = lambda df, pre: df[f"{pre}_{col}"].values if thr is None else (df[f"{pre}_{col}"].values >= thr)
            r = dict(model=model, condition=cname, metric=metric, n_pos=len(common), fast=f(l, "fast").mean(), local=f(l, "slow").mean(),
                     deepseek=f(d, "slow").mean())
            r["local_minus_fast"], r["lmf_lo"], r["lmf_hi"] = paired(f(l, "slow"), f(l, "fast"))
            r["local_minus_deepseek"], r["lmd_lo"], r["lmd_hi"] = paired(f(l, "slow"), f(d, "slow"))
            r["deepseek_minus_fast"], r["dmf_lo"], r["dmf_hi"] = paired(f(d, "slow"), f(d, "fast"))
            rows.append(r)
        if len(ln):
            rows.append(dict(model=model, condition=cname, metric="correct_none_neg", n_pos=len(ln), fast=0.0, local=(ln.choice == "NONE").mean(),
                             deepseek=(dn.choice == "NONE").mean() if len(dn) else np.nan))
        rows.append(dict(model=model, condition=cname, metric="none_rate_pos", n_pos=len(common), local=(l.choice == "NONE").mean(), deepseek=(d.choice == "NONE").mean()))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--conds", default="C1K1")
    ap.add_argument("--n-pos", type=int, default=5); ap.add_argument("--n-neg", type=int, default=2)
    ap.add_argument("--longest-edge", type=int, default=384); ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--aggregate-only", action="store_true"); ap.add_argument("--seed", type=int, default=8)
    a = ap.parse_args()
    conds = a.conds.split(",")
    e8.CONDITIONS["C1K1"] = (*e8.CONDITIONS["C1K1"][:3], True)     # local runs also test negatives at K=1
    outdir = C.results_dir(f"{e8.EXP}/local_{a.model}")
    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df))
    items_all = e8.sample_queries(df, QE, list(vids), a.seed)
    pos = [x for x in items_all if x["kind"] == "pos"][:a.n_pos]; neg = [x for x in items_all if x["kind"] == "neg"][:a.n_neg]
    order, ni = [], 0
    for i, q in enumerate(pos):
        order.append(q)
        if i % 2 == 1 and ni < len(neg): order.append(neg[ni]); ni += 1
    order += neg[ni:]
    plans_all = e8.prepare(order, vids)
    tasks = [(c, q) for q in order for c in conds if not (q["kind"] == "neg" and not e8.CONDITIONS[c][3])]
    meta = dict(model=MODELS[a.model], dtype="bfloat16", longest_edge=a.longest_edge, image_splitting=False, max_new_tokens=MAX_NEW, threads=2)
    if not a.aggregate_only:
        todo = [(c, q) for c, q in tasks if e8.load_cached(outdir, c, q["qid"], plans_all[(c, q["qid"])]["sha"]) is None]
        print(f"[local {a.model}] {len(tasks)} calls, {len(tasks) - len(todo)} cached, {len(todo)} to run", flush=True)
        if todo:
            proc, model = load_model(a.model, a.longest_edge)
            t0 = time.time()
            for i, (c, q) in enumerate(todo, 1):
                if a.time_budget_min and (time.time() - t0) / 60 > a.time_budget_min:
                    print(f"[local {a.model}] time budget {a.time_budget_min} min reached; stopping after {i - 1} calls", flush=True); break
                rec = one_call(proc, model, plans_all[(c, q["qid"])], c, outdir, meta)
                if i <= 5 or i % 10 == 0:
                    print(f"  [{i}/{len(todo)}] {rec['task']} choice={rec['choice']} in={rec['input_tokens']} lat={rec['latency_s']:.1f}s "
                          f"peakRSS={C.peak_rss_gb():.2f}GB ({time.time() - t0:.0f}s)", flush=True)
            (outdir / "peak_rss_gb.txt").write_text(f"{C.peak_rss_gb():.2f}\n", encoding="utf-8")
    plans = {k: v for k, v in plans_all.items() if k[0] in conds}
    pq = e8.build_per_query(outdir, plans)
    if pq.empty: print("nothing cached"); return
    pq = pq.sort_values(["cond", "kind", "qid"]).reset_index(drop=True)
    lat = {(r["cond"], r["qid"]): r for r in (json.loads(f.read_text(encoding="utf-8")) for f in (outdir / "raw").glob("*.json"))}
    pq["first_try_json"] = [lat[(c, q)].get("first_try_json", True) for c, q in zip(pq.cond, pq.qid)]
    pq["first_try_ok"] = [lat[(c, q)].get("first_try_ok", True) for c, q in zip(pq.cond, pq.qid)]
    pq.to_csv(outdir / "per_query.csv", index=False, encoding="utf-8")
    sm = e8.aggregate(outdir, pq, None)
    sm["parse_fail_first_try"] = sm.condition.map(lambda c: 1 - pq[pq.cond == c].first_try_ok.mean() if c in set(pq.cond) else np.nan)
    sm["json_format_fail_rate"] = sm.condition.map(lambda c: 1 - pq[pq.cond == c].first_try_json.mean() if c in set(pq.cond) else np.nan)
    sm["peak_rss_gb"] = float((outdir / "peak_rss_gb.txt").read_text(encoding="utf-8")) if (outdir / "peak_rss_gb.txt").exists() else np.nan
    sm["sec_per_query"] = sm.condition.map(lambda c: pq[pq.cond == c].latency_s.mean() if c in set(pq.cond) else np.nan)
    sm.to_csv(outdir / "summary.csv", index=False, encoding="utf-8")
    ds = pd.read_csv(C.RESULTS_DIR / e8.EXP / "per_query.csv", encoding="utf-8")
    cmp_ = compare(outdir, pq, ds, a.model); cmp_.to_csv(outdir / "compare.csv", index=False, encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(cmp_.round(3).to_string(index=False))
        print(sm[sm.system == "slow_vlm"][["condition", "n_pos", "hit1", "none_rate_pos", "n_neg", "correct_none_neg", "invalid_rate_pos", "parse_fail_first_try", "json_format_fail_rate", "sec_per_query", "peak_rss_gb"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
