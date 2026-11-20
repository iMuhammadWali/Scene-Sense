"""E9 - small local VLM (SmolVLM2-500M) as a per-clip scorer, instead of E8's one-shot multiple-choice prompt.

Same C1 queries as the E8 local runs (first 100 positives + first 50 negatives of the DeepSeek C1 sample), same fast-path candidates
(5 per query, E8 `prepare`), same frame decoding (K=1: the 1 FPS grid frame nearest the candidate time, 448 px JPEG, processor longest edge 384),
same model loading (bf16, 2 threads). Nothing from E8 is modified; this file only imports it.

Methods
  1. yes/no likelihood: one forward pass per candidate, prompt "Does this image show: '<query>'? Answer Yes or No.",
     p_yes = softmax over the best "Yes" token id vs the best "No" token id at the first answer position (no generation).
  2. rankings: fast top-1 | VLM-only argmax p_yes | fusion z(CLIP)+w*z(logit p_yes), w picked by 2-fold CV | DeepSeek (E8 C1K1) for reference.
  3. absent events: AUROC of max_c p_yes (and of max CLIP) on 100 pos vs 50 neg + a NONE rule at 5% NONE on positives.
  4. position bias: E8's A-E multiple-choice prompt (K=1) with the candidate order shuffled (fixed seed).
  5. timing / peak RAM.

Usage:
  python -W ignore experiments/e9_vlm_scoring/run.py [--limit N] [--skip-posbias] [--aggregate-only] [--out DIR]
Resumable: every (query, candidate) forward pass / shuffled-MC call is cached as one JSON in <out>/raw and <out>/posbias/raw.
"""
from __future__ import annotations
import sys, os, io, json, time, hashlib, importlib.util, argparse
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(os.environ.get("SCENESENSE_MODELS_DIR", Path(__file__).resolve().parents[2] / "data" / "models")) / "hf_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd

EXP = "e9_vlm_scoring"
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
LONGEST_EDGE = 384
N_POS, N_NEG, N_CAND = 100, 50, 5
W_GRID = (0.25, 0.5, 1.0, 2.0)
CV_REPEATS = 20
POSBIAS_SEED = 9
YN_PROMPT = "Does this image show: '{q}'? Answer Yes or No."
YES_WORDS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_WORDS = ["No", " No", "no", " no", "NO", " NO"]
LETTERS = "ABCDE"

# ------------------------------------------------------------------------------------------------------------ pure functions (unit-tested)
def yes_no_prob(logits, yes_ids, no_ids) -> tuple[float, float]:
    """Next-token logits (1-D, vocab) -> (p_yes, logit_diff). Each answer is the MAX logit over its token-id variants
    (e.g. 'Yes' / ' Yes' / 'yes'); p_yes is the softmax over those two values, logit_diff = yes - no (= logit(p_yes), numerically safe)."""
    lg = np.asarray(logits, dtype=np.float64)
    d = float(lg[list(yes_ids)].max() - lg[list(no_ids)].max())
    p = 1.0 / (1.0 + np.exp(-d)) if d >= 0 else float(np.exp(d) / (1.0 + np.exp(d)))
    return float(p), d

def single_token_ids(tokenizer, words) -> list[int]:
    """Token ids of the variants that encode to exactly one token (multi-token variants cannot be scored from one position)."""
    ids = []
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if len(t) == 1 and t[0] not in ids:
            ids.append(int(t[0]))
    return ids

def zscore_rows(x: np.ndarray) -> np.ndarray:
    """z-normalise each row (one query's candidates); a constant row -> zeros."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x, axis=1, keepdims=True); sd = np.nanstd(x, axis=1, keepdims=True)     # NaN = missing candidate (queries with < 5 candidates)
    return np.where(np.isnan(x), np.nan, np.where(sd > 1e-12, (x - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0))

def fuse_choice(clip: np.ndarray, vlm_logit: np.ndarray, w: float) -> np.ndarray:
    """clip, vlm_logit: [n_queries, n_cand]. -> argmax_c z_clip + w * z_vlm per query (ties -> lowest candidate index = best CLIP rank)."""
    return argmax_nan(zscore_rows(clip) + w * zscore_rows(vlm_logit))

def argmax_nan(x: np.ndarray) -> np.ndarray:
    """Row-wise argmax treating NaN (missing candidate) as -inf; ties -> lowest index."""
    return np.argmax(np.where(np.isnan(x), -np.inf, x), axis=1)

def pick_w_cv(clip, vlm_logit, hit, ws, seed: int):
    """2-fold CV over queries. -> (pooled held-out hit vector [n], chosen w per fold [2]). Fit = argmax mean Hit@1 on the training half
    (ties -> smallest w); evaluate on the other half."""
    n = len(clip); perm = np.random.default_rng(seed).permutation(n)
    folds = [perm[: n // 2], perm[n // 2:]]
    hits = np.zeros(n); chosen = []
    H = {w: hit[np.arange(n), fuse_choice(clip, vlm_logit, w)] for w in ws}
    for k in range(2):
        tr, te = folds[1 - k], folds[k]
        best = max(sorted(ws), key=lambda w: (H[w][tr].mean(), -w))
        chosen.append(best); hits[te] = H[best][te]
    return hits, chosen

def none_threshold(pos_scores, rate: float = 0.05) -> float:
    """Threshold t such that 'NONE if score <= t' fires on `rate` of the positives (the ceil(rate*n)-th smallest score)."""
    s = np.sort(np.asarray(pos_scores, dtype=np.float64))
    return float(s[max(0, int(np.ceil(rate * len(s))) - 1)])

# ------------------------------------------------------------------------------------------------------------ E8 reuse
def load_e8():
    spec = importlib.util.spec_from_file_location("e8_run", Path(__file__).resolve().parents[2] / "experiments" / "e8_slowpath" / "run.py")
    m = importlib.util.module_from_spec(spec); sys.modules["e8_run"] = m; spec.loader.exec_module(m)
    return m

def load_local():
    spec = importlib.util.spec_from_file_location("e8_local_vlm", Path(__file__).resolve().parents[2] / "experiments" / "e8_slowpath" / "local_vlm.py")
    m = importlib.util.module_from_spec(spec); sys.modules["e8_local_vlm"] = m; spec.loader.exec_module(m)
    return m

E8_COND = {"C1": ("full_5fps", {}), "C2": ("cos_last_drop", {"tau": 0.95})}      # E9 candidate condition -> E8 pruning method
DS_POS = {"C1": "C1K1", "C2": "C2K3"}      # DeepSeek condition compared against on positives
DS_NEG = {"C1": "C1K3"}                    # E8 only has negatives with K=3 (C1K1 had none)

def build_plans(limit: int | None, n_pos: int = N_POS, n_neg: int = N_NEG, cond: str = "C1"):
    """Same queries and candidates as E8: first n_pos positives + first n_neg negatives of the seed-8 sample (the E8 local runs used 100 + 50)."""
    from sseval import data
    e8 = load_e8()
    method, params = E8_COND[cond]
    e8.CONDITIONS = {cond: (method, params, 1, True)}      # only this index is built; K only affects the (unused here) E8 prompt
    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df))
    items_all = e8.sample_queries(df, QE, list(vids), 8)
    if limit: n_pos, n_neg = limit, max(1, limit // 3)
    pos = [x for x in items_all if x["kind"] == "pos"][:n_pos]; neg = [x for x in items_all if x["kind"] == "neg"][:n_neg]
    items = pos + neg
    plans = e8.prepare(items, vids)
    return e8, items, {q["qid"]: plans[(cond, q["qid"])] for q in items}

def sha(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]

def write_json(p: Path, rec: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=1, default=lambda o: None if o != o else str(o)), encoding="utf-8"); tmp.replace(p)

def read_cached(p: Path, key: str):
    if p.exists():
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("req_sha") == key:
                return r
        except Exception:
            pass
    return None

# ------------------------------------------------------------------------------------------------------------ method 1: yes/no scoring
def score_query(proc, model, e8, plan, yes_ids, no_ids, rawdir: Path, fallback: Path | None = None) -> int:
    """Run (uncached) yes/no forward passes for the 5 candidates of one query. Returns the number of forward passes executed."""
    import torch
    from PIL import Image
    q = plan["q"]; cands = plan["cands"]; n_done = 0
    todo = []
    for c in cands:
        key = sha("yn", MODEL_ID, LONGEST_EDGE, YN_PROMPT, q["caption"], round(c["t"], 3), yes_ids, no_ids)
        if read_cached(rawdir / f"yn_{q['qid']}_{c['label']}.json", key) is None:
            hit = None                                   # C2: the same (query, time) may already be scored under C1 (different label)
            for pth in (fallback.glob(f"yn_{q['qid']}_*.json") if fallback else []):
                hit = read_cached(pth, key)
                if hit is not None: break
            if hit is not None:
                write_json(rawdir / f"yn_{q['qid']}_{c['label']}.json", dict(hit, label=c["label"], reused_from=str(fallback.name)))
            else:
                todo.append((c, key))
    if not todo:
        return 0
    t0 = time.perf_counter()
    times = [e8.clip_times(c["t"], plan["dur"], 1)[0] for c, _ in todo]
    jpgs = e8.decode_frames(e8.C.VIDEOS_DIR / f"{q['vid']}.mp4", times)
    dec = (time.perf_counter() - t0) / len(todo)
    for (c, key), ft, jpg in zip(todo, times, jpgs):
        img = Image.open(io.BytesIO(jpg)).convert("RGB")
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": YN_PROMPT.format(q=q["caption"])}]}]
        prompt = proc.apply_chat_template(msg, add_generation_prompt=True)
        t1 = time.perf_counter()
        inp = proc(text=prompt, images=[[img]], return_tensors="pt")
        t2 = time.perf_counter()
        with torch.no_grad():
            try:
                out = model(**inp, logits_to_keep=1)
            except TypeError:
                out = model(**inp)
        t3 = time.perf_counter()
        lg = out.logits[0, -1].float().numpy()
        p, d = yes_no_prob(lg, yes_ids, no_ids)
        top = int(lg.argmax())
        write_json(rawdir / f"yn_{q['qid']}_{c['label']}.json", dict(
            req_sha=key, qid=q["qid"], kind=q["kind"], vid=q["vid"], label=c["label"], t=c["t"], frame_t=ft, clip_sim=c["sim"],
            p_yes=p, logit_diff=d, yes_logit=float(lg[yes_ids].max()), no_logit=float(lg[no_ids].max()), argmax_token=proc.tokenizer.decode([top]),
            n_input_tokens=int(inp["input_ids"].shape[1]), decode_s=dec, proc_s=t2 - t1, fwd_s=t3 - t2))
        n_done += 1
    return n_done

# ------------------------------------------------------------------------------------------------------------ method 4: shuffled multiple choice
def shuffled_plan(e8, plan):
    """E8's A-E prompt with the candidate order randomly permuted (seeded per qid). Returns (parts, perm) with perm[j] = original index shown j-th."""
    q = plan["q"]
    perm = [int(i) for i in np.random.default_rng([POSBIAS_SEED, q["qid"]]).permutation(len(plan["cands"]))]
    cands = []
    for j, i in enumerate(perm):
        c = dict(plan["cands"][i]); c["label"] = LETTERS[j]; cands.append(c)
    return e8.prompt_parts(q["caption"], cands, 1, plan["dur"]), perm

def posbias_query(local, e8, proc, model, plan, outdir: Path) -> bool:
    q = plan["q"]; parts, perm = shuffled_plan(e8, plan)
    key = sha("mc", MODEL_ID, LONGEST_EDGE, e8.request_sha(parts, 1), perm)
    p = outdir / "posbias" / "raw" / f"mc_{q['qid']}.json"
    if read_cached(p, key) is not None:
        return False
    r = local.run_local(proc, model, parts, q["vid"])
    ch, _, st = e8.parse_response(r["text"], len(perm), lenient=True)
    if st != "ok":
        r2 = local.run_local(proc, model, parts, q["vid"], suffix=e8.RETRY_SUFFIX)
        ch, _, st = e8.parse_response(r2["text"], len(perm), lenient=True)
        r["text"] += " || RETRY: " + r2["text"]; r["latency_s"] += r2["latency_s"]
    write_json(p, dict(req_sha=key, qid=q["qid"], kind=q["kind"], perm=perm, text=r["text"], choice=ch, status=st,
                       input_tokens=r["input_tokens"], latency_s=r["latency_s"]))
    return True

# ------------------------------------------------------------------------------------------------------------ aggregation
def aggregate(out: Path, e8, items, plans, cond: str = "C1"):
    """All metrics on the full sample (set='full') and, if it is larger, on the first 100 pos / 50 neg that the first E9 pass used (set='first100')."""
    from sseval.metrics import bootstrap_indices, bootstrap_ci, auroc
    rawdir = out / ("raw" if cond == "C1" else f"raw_{cond}"); sfx = "" if cond == "C1" else f"_{cond}"
    ci = lambda x, idx: bootstrap_ci(np.asarray(x, dtype=float), idx)
    rows = []                                           # long-format summary: cond, set, section, name, metric, n, value, lo, hi
    def _add(tag, section, name, metric, n, v, lo=np.nan, hi=np.nan):
        rows.append(dict(cond=cond, set=tag, section=section, name=name, metric=metric, n=n, value=v, lo=lo, hi=hi))
    add = lambda *a, **k: _add("full", *a, **k)
    recs = {}
    for q in items:
        pl = plans[q["qid"]]; ok = True; per = []
        for c in pl["cands"]:
            key = sha("yn", MODEL_ID, LONGEST_EDGE, YN_PROMPT, q["caption"], round(c["t"], 3), *YN_IDS)
            r = read_cached(rawdir / f"yn_{q['qid']}_{c['label']}.json", key)
            if r is None: ok = False; break
            per.append(r)
        if ok: recs[q["qid"]] = per
    P = [q for q in items if q["kind"] == "pos" and q["qid"] in recs]; N = [q for q in items if q["kind"] == "neg" and q["qid"] in recs]
    if not P:
        print("[e9] no complete positives cached"); return None
    def mat(qs, f):            # [n, 5]; NaN where a query has fewer than 5 candidates
        m = np.full((len(qs), N_CAND), np.nan)
        for i, q in enumerate(qs):
            for j, r in enumerate(recs[q["qid"]]): m[i, j] = f(r)
        return m
    def hits(qs):
        h = np.zeros((len(qs), N_CAND), bool)
        for i, q in enumerate(qs):
            for j, c in enumerate(plans[q["qid"]]["cands"]): h[i, j] = q["gs"] <= c["t"] <= q["ge"]
        return h
    # DeepSeek reference (E8) on the same queries
    dsa = pd.read_csv(e8.C.RESULTS_DIR / "e8_slowpath" / "per_query.csv", encoding="utf-8")
    dsP = dsa[(dsa.cond == DS_POS[cond]) & (dsa.kind == "pos")].set_index("qid")
    dsN = dsa[(dsa.cond == DS_NEG.get(cond, "-")) & (dsa.kind == "neg")].set_index("qid")
    ctimes = lambda q: ";".join(f"{c['t']:.2f}" for c in plans[q["qid"]]["cands"])
    bad = [q["qid"] for q in P if q["qid"] in dsP.index and dsP.loc[q["qid"], "cand_times"] != ctimes(q)]
    bad += [q["qid"] for q in N if q["qid"] in dsN.index and dsN.loc[q["qid"], "cand_times"] != ctimes(q)]
    print(f"[e9] {cond}: candidate times differ from E8 for {len(bad)} queries" + (f" {bad[:5]}" if bad else " (none)"))
    DS = f"deepseek_{DS_POS[cond]}"

    def analyse(tag, P, N):
        add = lambda *a, **k: _add(tag, *a, **k)
        clipP, lgP, pyP = mat(P, lambda r: r["clip_sim"]), mat(P, lambda r: r["logit_diff"]), mat(P, lambda r: r["p_yes"])
        hitP = hits(P); n = len(P); ar = np.arange(n); idx = bootstrap_indices(n)
        add("diag", "candidates", "queries_with_fewer_than_5_candidates", n, int(np.isnan(clipP).any(axis=1).sum()))
        sys_hit = {"fast_top1": hitP[:, 0].astype(float), "vlm_only": hitP[ar, argmax_nan(lgP)].astype(float)}
        for w in W_GRID:
            sys_hit[f"fusion_w{w:g}"] = hitP[ar, fuse_choice(clipP, lgP, w)].astype(float)
        cv_hit, cv_w = pick_w_cv(clipP, lgP, hitP, W_GRID, seed=0)
        sys_hit["fusion_cv"] = cv_hit
        reps = [pick_w_cv(clipP, lgP, hitP, W_GRID, seed=s) for s in range(CV_REPEATS)]
        w_counts = pd.Series([w for _, ws in reps for w in ws]).value_counts().sort_index()
        sys_hit[DS] = np.array([float(dsP.loc[q["qid"], "slow_hit1"]) if q["qid"] in dsP.index else np.nan for q in P])
        sys_hit["oracle_best_of_5"] = hitP.any(axis=1).astype(float)
        fast = sys_hit["fast_top1"]
        for name, h in sys_hit.items():
            lo, hi = ci(h, idx); add("ranking", name, "hit1", n, np.mean(h), lo, hi)
            if name != "fast_top1":
                d = h - fast; lo, hi = ci(d, idx); add("ranking", name, "hit1_minus_fast", n, d.mean(), lo, hi)
            if name not in ("fast_top1", DS, "oracle_best_of_5"):
                d = h - sys_hit[DS]; lo, hi = ci(d, idx); add("ranking", name, "hit1_minus_deepseek", n, d.mean(), lo, hi)
        d = sys_hit["fusion_cv"] - sys_hit["vlm_only"]; lo, hi = ci(d, idx); add("ranking", "fusion_cv", "hit1_minus_vlm_only", n, d.mean(), lo, hi)
        add("cv", "fusion_cv", "chosen_w_fold1", n, cv_w[0]); add("cv", "fusion_cv", "chosen_w_fold2", n, cv_w[1])
        add("cv", "fusion_cv", "hit1_mean_over_20_random_splits", n, float(np.mean([h.mean() for h, _ in reps])))
        for w, c in w_counts.items(): add("cv", f"w={w:g}", "times_chosen_in_40_folds", n, int(c))
        best_w = max(W_GRID, key=lambda w: (sys_hit[f"fusion_w{w:g}"].mean(), -w)); add("cv", f"w={best_w:g}", "best_in_sample_w", n, sys_hit[f"fusion_w{best_w:g}"].mean())
        add("diag", "vlm", "frac_queries_with_tied_max_logit", n, float(np.mean([(lgP[i] == np.nanmax(lgP[i])).sum() > 1 for i in range(n)])))
        add("diag", "vlm", "mean_p_yes_correct_cands", int(hitP.sum()), float(pyP[hitP].mean()))
        add("diag", "vlm", "mean_p_yes_wrong_cands", int((~hitP & ~np.isnan(pyP)).sum()), float(np.nanmean(np.where(hitP, np.nan, pyP))))
        for nm, M in (("vlm", lgP), ("clip", clipP)):
            add("diag", nm, "pairwise_auc_correct_vs_wrong_cands_within_query", n, float(np.nanmean([
                np.mean([(a > b) + 0.5 * (a == b) for a in M[i][hitP[i]] for b in M[i][~hitP[i] & ~np.isnan(M[i])]]) if hitP[i].any() and (~hitP[i] & ~np.isnan(M[i])).any() else np.nan for i in range(n)])))
        dn = (dsP.reindex([q["qid"] for q in P]).choice == "NONE").dropna()
        if len(dn): add("absent", DS, "none_rate_on_pos", len(dn), float(dn.mean()), *ci(dn.values, bootstrap_indices(len(dn), seed=1)))
        if N:
            clipN, pyN = mat(N, lambda r: r["clip_sim"]), mat(N, lambda r: r["p_yes"])
            nN = len(N); ip, inn = bootstrap_indices(n, seed=1), bootstrap_indices(nN, seed=2)
            for name, sp, sn in (("max_p_yes", np.nanmax(pyP, 1), np.nanmax(pyN, 1)), ("max_clip", np.nanmax(clipP, 1), np.nanmax(clipN, 1))):
                boots = [auroc(sp[ip[b]], sn[inn[b]]) for b in range(len(ip))]
                add("absent", name, "auroc_pos_vs_neg", f"{n}/{nN}", auroc(sp, sn), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
                th = none_threshold(sp, 0.05)
                add("absent", name, "none_threshold_(in_sample)", f"{n}/{nN}", th)
                add("absent", name, "none_rate_on_pos", n, float((sp <= th).mean()), *ci(sp <= th, ip))
                add("absent", name, "true_none_rate_on_neg", nN, float((sn <= th).mean()), *ci(sn <= th, inn))
            dd = (dsN.reindex([q["qid"] for q in N]).choice == "NONE").dropna()
            if len(dd): add("absent", f"deepseek_{DS_NEG[cond]}", "true_none_rate_on_neg", len(dd), float(dd.mean()), *ci(dd.values, bootstrap_indices(len(dd), seed=2)))
        return sys_hit, cv_w, w_counts, clipP, lgP, hitP

    res_full = analyse("full", P, N)
    if len(P) > N_POS:
        analyse("first100", P[:N_POS], N[:N_NEG])
    sys_hit = res_full[0]
    # ---- per-query table (full sample)
    pq = []
    for k, q in enumerate(P + N):
        pl = plans[q["qid"]]; r = recs[q["qid"]]; cl = np.array([[x["clip_sim"] for x in r]]); lg = np.array([[x["logit_diff"] for x in r]])
        row = dict(qid=q["qid"], kind=q["kind"], in_first100=(q in P[:N_POS] or q in N[:N_NEG]), vid=q["vid"], caption=q["caption"], gt_s=q["gs"], gt_e=q["ge"],
                   cand_times=ctimes(q), clip_sims=";".join(f"{x:.4f}" for x in cl[0]),
                   p_yes=";".join(f"{x['p_yes']:.4f}" for x in r), logit_diff=";".join(f"{x['logit_diff']:.3f}" for x in r),
                   max_p_yes=max(x["p_yes"] for x in r), max_clip=float(cl.max()), vlm_choice=LETTERS[int(np.argmax(lg))])
        if q["kind"] == "pos":
            h = [q["gs"] <= c["t"] <= q["ge"] for c in pl["cands"]]
            row.update(correct_cands="".join(LETTERS[i] for i, x in enumerate(h) if x), fast_hit1=h[0], vlm_hit1=h[int(np.argmax(lg))])
            for w in W_GRID: row[f"fusion_w{w:g}_hit1"] = h[int(fuse_choice(cl, lg, w)[0])]
            row["fusion_cv_hit1"] = bool(sys_hit["fusion_cv"][P.index(q)])
            row["deepseek_choice"] = dsP.loc[q["qid"], "choice"] if q["qid"] in dsP.index else None
            row["deepseek_hit1"] = bool(dsP.loc[q["qid"], "slow_hit1"]) if q["qid"] in dsP.index else None
        elif q["qid"] in dsN.index:
            row["deepseek_choice"] = dsN.loc[q["qid"], "choice"]
        pq.append(row)
    pd.DataFrame(pq).to_csv(out / f"per_query{sfx}.csv", index=False, encoding="utf-8")
    # ---- timing (full sample; forward passes only, in whichever run they were measured)
    allq = P + N
    fw = [sum(x["fwd_s"] for x in recs[q["qid"]]) for q in allq]; tot = [sum(x["fwd_s"] + x["proc_s"] for x in recs[q["qid"]]) for q in allq]
    dec = [sum(x["decode_s"] for x in recs[q["qid"]]) for q in allq]
    add("timing", "yes_no_scoring", "forward_s_per_query_(5 passes)", len(fw), float(np.mean(fw)), float(np.percentile(fw, 5)), float(np.percentile(fw, 95)))
    add("timing", "yes_no_scoring", "forward_plus_preprocess_s_per_query", len(tot), float(np.mean(tot)))
    add("timing", "yes_no_scoring", "frame_decode_s_per_query_(not in the above)", len(dec), float(np.mean(dec)))
    add("timing", "yes_no_scoring", "input_tokens_per_pass", len(fw), float(np.mean([x["n_input_tokens"] for q in allq for x in recs[q["qid"]]])))
    rss = out / "peak_rss_gb.txt"
    if rss.exists(): add("timing", "process", "peak_rss_gb", 1, float(rss.read_text(encoding="utf-8").split()[0]))
    if cond == "C1":
        aggregate_posbias(out, e8, plans, P[:N_POS], add, dsP)       # position-bias rerun exists for the first 100 positives only
    sm = pd.DataFrame(rows); sm.to_csv(out / f"summary{sfx}.csv", index=False, encoding="utf-8")
    return sm

def aggregate_posbias(out, e8, plans, P, add, ds):
    from sseval.metrics import bootstrap_indices, bootstrap_ci
    ci = lambda x, idx: bootstrap_ci(np.asarray(x, dtype=float), idx)
    rows = []
    for q in P:
        pl = plans[q["qid"]]; parts, perm = shuffled_plan(e8, pl)
        key = sha("mc", MODEL_ID, LONGEST_EDGE, e8.request_sha(parts, 1), perm)
        r = read_cached(out / "posbias" / "raw" / f"mc_{q['qid']}.json", key)
        if r is None: continue
        ch = r["choice"]; j = LETTERS.index(ch) if ch in LETTERS[:5] else None
        orig = perm[j] if j is not None else None
        hit_shown = [bool(q["gs"] <= pl["cands"][i]["t"] <= q["ge"]) for i in perm]        # correctness by displayed position
        rows.append(dict(qid=q["qid"], perm="".join(map(str, perm)), raw=r["text"].replace("\n", " ")[:80], choice=ch, chosen_orig_idx=orig,
                         picked_first_shown=(j == 0), picked_fast_top1=(orig == 0), hit1=bool(j is not None and hit_shown[j]),
                         first_shown_is_correct=hit_shown[0], fast_top1_shown_pos=LETTERS[perm.index(0)], fast_hit1=bool(hit_shown[perm.index(0)]),
                         n_correct_cands=int(sum(hit_shown)), none=(ch == "NONE")))
    if not rows: return None
    d = pd.DataFrame(rows); d.to_csv(out / "posbias_per_query.csv", index=False, encoding="utf-8")
    n = len(d); idx = bootstrap_indices(n)
    for m in ("hit1", "picked_first_shown", "picked_fast_top1", "first_shown_is_correct", "fast_hit1", "none"):
        lo, hi = ci(d[m], idx); add("posbias_shuffled", "smolvlm2_500m_mc", m, n, float(d[m].mean()), lo, hi)
    x = d.hit1.astype(float).values - d.fast_hit1.astype(float).values; lo, hi = ci(x, idx)
    add("posbias_shuffled", "smolvlm2_500m_mc", "hit1_minus_fast_top1", n, float(x.mean()), lo, hi)
    x = d.hit1.astype(float).values - d.first_shown_is_correct.astype(float).values; lo, hi = ci(x, idx)
    add("posbias_shuffled", "smolvlm2_500m_mc", "hit1_minus_always_pick_first_shown", n, float(x.mean()), lo, hi)
    for L in LETTERS: add("posbias_shuffled", f"letter_{L}", "share_of_answers", n, float((d.choice == L).mean()))
    # same queries, unshuffled (E8 local_500M C1K1)
    f = e8.C.RESULTS_DIR / "e8_slowpath" / "local_500M" / "per_query.csv"
    if f.exists():
        u = pd.read_csv(f, encoding="utf-8"); u = u[(u.cond == "C1K1") & (u.kind == "pos") & u.qid.isin(d.qid)]
        if len(u):
            add("posbias_unshuffled_E8", "smolvlm2_500m_mc", "hit1", len(u), float(u.slow_hit1.mean()), *ci(u.slow_hit1, bootstrap_indices(len(u))))
            add("posbias_unshuffled_E8", "smolvlm2_500m_mc", "picked_first_shown(A)", len(u), float((u.choice == "A").mean()))
    return d

def make_figure(out: Path, path: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    for cond, f in (("C1", out / "summary.csv"), ("C2", out / "summary_C2.csv")):
        if f.exists():
            sm = pd.read_csv(f, encoding="utf-8"); sm = sm[(sm.set == "full") & (sm.section == "ranking") & (sm.metric == "hit1")]
            if len(sm): panels.append((cond, sm.set_index("name")))
    if not panels: return
    spec = [("fast_top1", "Fast path\n(CLIP top-1)", "#9aa0a6"), ("vlm_only", "SmolVLM2-500M\nyes/no only", "#E69F00"),
            ("fusion_cv", "Fusion CLIP+VLM\n(w by CV)", "#0072B2"), ("deepseek", "DeepSeek\n(E8)", "#009E73"), ("oracle_best_of_5", "Oracle\nbest of 5", "#d8d8d8")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.8), squeeze=False, sharey=True)
    for ax, (cond, r) in zip(axes[0], panels):
        r = r.rename(index=lambda x: "deepseek" if str(x).startswith("deepseek_") else x)
        for k, (name, lab, c) in enumerate(spec):
            v, lo, hi = 100 * r.loc[name, "value"], 100 * r.loc[name, "lo"], 100 * r.loc[name, "hi"]
            ax.bar(k, v, 0.62, color=c, yerr=[[v - lo], [hi - v]], capsize=4, error_kw=dict(lw=1, color="#333")); ax.text(k, hi + 1.5, f"{v:.0f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(spec))); ax.set_xticklabels([x[1] for x in spec], fontsize=8); ax.set_ylim(0, 105)
        ax.set_title(f"{cond}: {'full 5 FPS index' if cond == 'C1' else 'pruned index (cos_last_drop tau 0.95)'}, n={int(r.iloc[0]['n'])} positives", fontsize=10)
        ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Hit@1 (%), 95% bootstrap CI")
    fig.suptitle("E9: SmolVLM2-500M as a per-clip yes/no scorer (5 candidates, K=1 frame) vs DeepSeek", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------------------ main
YN_IDS: tuple = ()

def main():
    global YN_IDS
    ap = argparse.ArgumentParser(description="E9 VLM per-clip scoring")
    ap.add_argument("--limit", type=int, default=None, help="N positives (and N//3 negatives); overrides --n-pos/--n-neg")
    ap.add_argument("--n-pos", type=int, default=N_POS); ap.add_argument("--n-neg", type=int, default=N_NEG)
    ap.add_argument("--cond", choices=list(E8_COND), default="C1", help="candidate index: C1 full 5 FPS, C2 pruned cos_last_drop tau .95")
    ap.add_argument("--skip-posbias", action="store_true"); ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None); ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    from sseval import config as C, runutil
    out = a.out or C.results_dir(EXP); out.mkdir(parents=True, exist_ok=True)
    rawdir = out / ("raw" if a.cond == "C1" else f"raw_{a.cond}")
    e8, items, plans = build_plans(a.limit, a.n_pos, a.n_neg, a.cond)
    print(f"[e9] {a.cond}: {sum(i['kind'] == 'pos' for i in items)} pos / {sum(i['kind'] == 'neg' for i in items)} neg; out={out}", flush=True)
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(MODEL_ID).tokenizer
    yes_ids, no_ids = single_token_ids(tok, YES_WORDS), single_token_ids(tok, NO_WORDS)
    YN_IDS = (yes_ids, no_ids)
    print(f"[e9] yes ids {dict(zip(map(tok.decode, yes_ids), yes_ids))}  no ids {dict(zip(map(tok.decode, no_ids), no_ids))}", flush=True)
    if not a.aggregate_only:
        local = load_local()
        proc, model = local.load_model("500M", LONGEST_EDGE)
        t0 = time.time(); n_fw = 0
        for i, q in enumerate(items, 1):
            n_fw += score_query(proc, model, e8, plans[q["qid"]], yes_ids, no_ids, rawdir, out / "raw" if a.cond != "C1" else None)
            if i % 20 == 0 or i == len(items):
                print(f"  [score {i}/{len(items)}] qid={q['qid']} forwards so far={n_fw} peakRSS={C.peak_rss_gb():.2f}GB ({time.time() - t0:.0f}s)", flush=True)
        if not a.skip_posbias:
            t0 = time.time(); n_mc = 0
            for i, q in enumerate([x for x in items if x["kind"] == "pos"], 1):
                n_mc += posbias_query(local, e8, proc, model, plans[q["qid"]], out)
                if i % 10 == 0: print(f"  [posbias {i}] calls so far={n_mc} ({time.time() - t0:.0f}s)", flush=True)
        rss = out / "peak_rss_gb.txt"
        new = C.peak_rss_gb()
        old = float(rss.read_text(encoding="utf-8").split()[0]) if rss.exists() else 0.0
        rss.write_text(f"{max(new, old):.2f}\n", encoding="utf-8")
    sm = aggregate(out, e8, items, plans, a.cond)
    if sm is None: return
    if not a.no_figure:
        fig = C.RESULTS_DIR / "figures"; fig.mkdir(parents=True, exist_ok=True)
        make_figure(out, out / "e9_vlm_scoring.png" if a.out else fig / "e9_vlm_scoring.png")
    runutil.write_meta(out, limit=a.limit, model=MODEL_ID, w_grid=list(W_GRID), cond=a.cond, n_pos=a.n_pos, n_neg=a.n_neg)
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 60):
        print(sm[sm.set == "full"].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
