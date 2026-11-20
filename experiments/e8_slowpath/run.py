"""E8 - slow path: VLM verification of fast-path candidates.

Question: what does a vision-language model add over the fast path (CLIP top-1), how does index pruning change that,
and what does it cost (tokens, $, latency)?

Per query: build 5 candidate moments from the fast path (E1 protocol: top-20 hits -> group_temporally(gap 2 s) -> rank groups by
best hit -> 5 distinct groups, padded with next-best peaks >= 3 s apart), decode K frames per candidate from the RAW video
(+-5 s window at 1 FPS, K evenly spaced, as the real slow path's merge_context_windows/extract_frames_in_range do), and ask the
VLM (DeepSeek API, OpenAI-compatible) which clip shows the event, or NONE.

Conditions: C1K3 = full 5 FPS index, 3 frames/candidate; C2K3 = pruned index (cos_last_drop tau 0.95), K=3;
            C1K1 = full index, 1 frame/candidate (positives only).
Usage:
  python -W ignore experiments/e8_slowpath/run.py --dry-run [--limit N]     # no API, no key needed
  python -W ignore experiments/e8_slowpath/run.py --list-models              # GET /models
  python -W ignore experiments/e8_slowpath/run.py [--limit N]                # real run (resumes from raw/ cache)
  python -W ignore experiments/e8_slowpath/run.py --aggregate-only
The API key is read ONLY from ~/.secrets/deepseek_key (or env DEEPSEEK_API_KEY) and is never logged or stored.
"""
from __future__ import annotations
import sys, os, re, json, time, base64, hashlib, zlib, datetime, threading, argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd

from sseval import config as C, data, runutil
from sseval.metrics import bootstrap_indices, bootstrap_ci, iou_arrays
from sseval.pruning import select
from sseval.retrieval import build_index, search
from app.pipeline.temporal import group_temporally  # team's code, unmodified

EXP = "e8_slowpath"
KEY_FILE = Path(os.environ.get("E8_KEY_FILE", str(Path.home() / ".secrets" / "deepseek_key")))
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
N_POS, N_NEG, N_CAND = 300, 100, 5
TOP_HITS, GROUP_GAP, PAD_SEP, WIDEN = 20, 2.0, 3.0, 4.0
WINDOW_S, MAX_FRAMES_REAL = 5.0, 12
LONG_SIDE, JPEG_Q = 448, 80
TOK_PER_IMAGE_MAX = 1024
PRICE = {"peak": (0.30, 1.20), "off": (0.15, 0.60)}     # USD per 1M tokens (input cache-miss, output)
BUDGET_USD = 1.00
CONCURRENCY = 3
MAX_OUT_TOKENS = 256
THINKING = os.environ.get("E8_THINKING", "disabled")   # DeepSeek: the model reasons by default and burns max_tokens before answering
LETTERS = "ABCDE"
CONDITIONS = {                                           # name -> (pruning method, params, K, include negatives)
    "C1K3": ("full_5fps", {}, 3, True),
    "C2K3": ("cos_last_drop", {"tau": 0.95}, 3, True),
    "C1K1": ("full_5fps", {}, 1, False),
}

# ---------------------------------------------------------------------------------------------------------------- pricing
def price_tier(dt: datetime.datetime) -> str:
    """Peak = Mon-Fri, 01:00-04:00 and 06:00-10:00 UTC (end-exclusive; the boundary hour is our reading of '01-04 and 06-10')."""
    dt = dt.astimezone(datetime.timezone.utc)
    return "peak" if dt.weekday() < 5 and (1 <= dt.hour < 4 or 6 <= dt.hour < 10) else "off"

def cost_usd(in_tok: int, out_tok: int, tier: str) -> float:
    pi, po = PRICE[tier]
    return (in_tok * pi + out_tok * po) / 1e6

# ---------------------------------------------------------------------------------------------------------------- parsing
_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)
_CHOICE_RE = re.compile(r"""["']?choice["']?\s*[:=]\s*["']?\s*(NONE|[A-Z])\b""", re.I)
_CONF_RE = re.compile(r"""["']?confidence["']?\s*[:=]\s*["']?\s*([0-9]*\.?[0-9]+)""", re.I)

def parse_response(text: str | None, n_cand: int = N_CAND, lenient: bool = False):
    """-> (choice, confidence, status). choice in letters[:n_cand] | 'NONE' | None; status 'ok' | 'invalid'.
    Tolerates code fences, prose around the JSON, single quotes, lowercase, and a missing confidence (NaN)."""
    valid = set(LETTERS[:n_cand]) | {"NONE"}
    if not text or not isinstance(text, str):
        return None, float("nan"), "invalid"
    choice, conf = None, float("nan")
    for m in _JSON_RE.finditer(text):
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except Exception:
            try:
                obj = json.loads(blob.replace("'", '"'))
            except Exception:
                continue
        if isinstance(obj, dict):
            lk = {str(k).lower(): v for k, v in obj.items()}
            if "choice" in lk:
                choice = str(lk["choice"]).strip().strip("\"'").upper()
                try: conf = float(lk.get("confidence", "nan"))
                except Exception: conf = float("nan")
                break
    if choice is None and lenient:       # small local VLMs often answer with a bare letter instead of JSON
        mb = re.fullmatch(r"\W*(?:clip\s*)?(NONE|[A-Z])\W*", text.strip(), re.I)
        if mb:
            choice = mb.group(1).upper()
    if choice is None:
        m = _CHOICE_RE.search(text)
        if m:
            choice = m.group(1).upper()
            mc = _CONF_RE.search(text)
            if mc:
                try: conf = float(mc.group(1))
                except Exception: pass
    if choice is not None:
        mm = re.fullmatch(r"(?:CLIP\s*)?([A-Z]|NONE)\.?", choice)
        choice = mm.group(1) if mm else choice
    if choice not in valid:
        return None, float("nan"), "invalid"
    if not np.isnan(conf):
        conf = float(min(1.0, max(0.0, conf)))
    return choice, conf, "ok"

# ---------------------------------------------------------------------------------------------------------------- candidates
def build_candidates(sel, hits: list[dict], sims: np.ndarray, duration: float, n: int = N_CAND,
                     group_gap: float = GROUP_GAP, pad_sep: float = PAD_SEP, widen: float = WIDEN) -> list[dict]:
    """Fast-path candidates for one query against one Selection.
    hits: top-20 search hits (dicts with faiss_index/timestamp/similarity/rank); sims: similarity of EVERY item in sel (for padding).
    Group hits with the team's group_temporally, rank groups by best-hit similarity, keep n; if fewer, pad with the next-best items
    at least pad_sep seconds from every chosen candidate time. Candidate time = group's best-hit timestamp; span = union of the
    group's coverage spans; wide = [t-widen, t+widen] clipped to [0, duration]."""
    groups, _ = group_temporally(hits, group_gap)
    cands = []
    for g in groups:
        best = max(g["frames"], key=lambda f: (f["similarity"], -f["timestamp"]))
        ids = [f["faiss_index"] for f in g["frames"]]
        cands.append(dict(t=float(best["timestamp"]), sim=float(best["similarity"]),
                          span_s=float(sel.span_start[ids].min()), span_e=float(sel.span_end[ids].max()), src="group"))
    cands.sort(key=lambda c: (-c["sim"], c["t"]))
    cands = cands[:n]
    if len(cands) < n:
        for i in np.argsort(-sims, kind="stable"):
            if len(cands) >= n:
                break
            t = float(sel.ts[i])
            if all(abs(t - c["t"]) >= pad_sep for c in cands):
                cands.append(dict(t=t, sim=float(sims[i]), span_s=float(sel.span_start[i]), span_e=float(sel.span_end[i]), src="pad"))
    for k, c in enumerate(cands):
        c["label"] = LETTERS[k]
        c["wide_s"], c["wide_e"] = max(0.0, c["t"] - widen), min(float(duration), c["t"] + widen)
    return cands

def clip_times(t: float, duration: float, K: int, window: float = WINDOW_S) -> list[float]:
    """Frame timestamps for one candidate: the real slow path's +-window at 1 FPS (clipped to the video), then K evenly spaced
    frames from it (K=1 -> the grid frame nearest to t). Chronological."""
    lo, hi = max(0.0, t - window), min(t + window, max(0.0, duration - 0.1))
    grid, x = [], lo
    while x <= hi + 1e-9:
        grid.append(round(x, 3)); x += 1.0
    if not grid:
        return [round(min(t, max(0.0, duration - 0.1)), 3)]
    if K >= len(grid):
        return grid
    if K == 1:
        return [min(grid, key=lambda g: abs(g - t))]
    idx = sorted(set(int(round(i)) for i in np.linspace(0, len(grid) - 1, K)))
    return [grid[i] for i in idx]

# ---------------------------------------------------------------------------------------------------------------- prompt
def prompt_parts(query: str, cands: list[dict], K: int, duration: float) -> list[dict]:
    """Ordered list of ('text', str) / ('img', t_seconds, cand_label) parts (images not yet decoded)."""
    n = len(cands)
    letters = "-".join([LETTERS[0], LETTERS[n - 1]]) if n > 1 else LETTERS[0]
    parts = [dict(type="text", text=(
        f"You are given {n} short clips ({letters}) taken from ONE video. Each clip is shown as up to {K} frame(s) in time order, "
        f"preceded by a label with the clip's time in the video. Base your answer only on what is visible.")),]
    for c in cands:
        parts.append(dict(type="text", text=f"Clip {c['label']} (t={c['t']:.1f} s):"))
        for ft in clip_times(c["t"], duration, K):
            parts.append(dict(type="img", t=ft, label=c["label"]))
    parts.append(dict(type="text", text=(
        f'Which clip best shows the event "{query}"? Respond ONLY with JSON: {{"choice": "{LETTERS[0]}".."{LETTERS[n - 1]}" or "NONE", '
        f'"confidence": 0-1}}. Answer NONE if no clip shows the event.')))
    return parts

RETRY_SUFFIX = "Reply with the JSON object only, e.g. {\"choice\": \"B\", \"confidence\": 0.7}."

def request_sha(parts: list[dict], K: int) -> str:
    blob = json.dumps([MODEL, THINKING, K, MAX_OUT_TOKENS, LONG_SIDE, JPEG_Q, parts], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def estimate_tokens(parts: list[dict]) -> int:
    """Worst case: every image billed at 1,024 tokens; text at ~4 chars/token."""
    return sum(TOK_PER_IMAGE_MAX if p["type"] == "img" else len(p["text"]) // 4 + 1 for p in parts)

# ---------------------------------------------------------------------------------------------------------------- frames
def decode_frames(video: Path, times: list[float]) -> list[bytes]:
    """JPEG bytes (long side LONG_SIDE, quality JPEG_Q) of the raw video at each timestamp."""
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
            h, w = img.shape[:2]
            s = LONG_SIDE / max(h, w)
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            out.append(buf.tobytes())
    finally:
        cap.release()
    return out

def materialize(parts: list[dict], vid: str) -> tuple[list[dict], list[bytes]]:
    """Decode all image parts -> OpenAI content parts. Returns (content, jpeg_bytes_list)."""
    times = [p["t"] for p in parts if p["type"] == "img"]
    jpgs = decode_frames(C.VIDEOS_DIR / f"{vid}.mp4", times)
    content, k = [], 0
    for p in parts:
        if p["type"] == "text":
            content.append({"type": "text", "text": p["text"]})
        else:
            b64 = base64.b64encode(jpgs[k]).decode(); k += 1
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content, jpgs

# ---------------------------------------------------------------------------------------------------------------- API
def load_key() -> str:
    k = os.environ.get("DEEPSEEK_API_KEY", "").strip() or (KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.exists() else "")
    if not k:
        raise RuntimeError(f"no API key: set DEEPSEEK_API_KEY or put it in {KEY_FILE}")
    return k

def scrub(s: str, key: str | None) -> str:
    """Remove the API key (and any Bearer token) from a string before it is logged or stored."""
    s = str(s)
    if key:
        s = s.replace(key, "[REDACTED]")
    return re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", s)

class Api:
    def __init__(self, key: str):
        import requests
        self._key, self._s = key, requests.Session()
        self._h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def models(self) -> list[str]:
        r = self._s.get(f"{BASE_URL}/models", headers=self._h, timeout=30)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def chat(self, content: list[dict], max_retries: int = 6) -> dict:
        """One chat completion with exponential backoff on 429/5xx/network errors.
        -> dict(text, usage, latency_s (final successful HTTP call), wall_s (incl. backoff), attempts, sent_at, error)."""
        import requests
        body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": MAX_OUT_TOKENS}
        if THINKING in ("disabled", "enabled"):
            body["thinking"] = {"type": THINKING}
        t_wall, attempts, err = time.time(), 0, None
        for a in range(max_retries + 1):
            attempts += 1
            sent_at = datetime.datetime.now(datetime.timezone.utc)
            t0 = time.perf_counter()
            try:
                r = self._s.post(f"{BASE_URL}/chat/completions", headers=self._h, json=body, timeout=120)
                lat = time.perf_counter() - t0
                if r.status_code == 429 or r.status_code >= 500:
                    err = f"HTTP {r.status_code}"
                    wait_s = float(r.headers.get("Retry-After", 0) or 0) or min(60.0, 2.0 ** a)
                elif r.status_code != 200:
                    return dict(text=None, usage={}, latency_s=lat, wall_s=time.time() - t_wall, attempts=attempts, sent_at=sent_at,
                                error=scrub(f"HTTP {r.status_code}: {r.text[:300]}", self._key))
                else:
                    j = r.json()
                    ch = (j.get("choices") or [{}])[0]
                    return dict(text=(ch.get("message") or {}).get("content"), usage=j.get("usage") or {}, latency_s=lat,
                                wall_s=time.time() - t_wall, attempts=attempts, sent_at=sent_at, error=None,
                                finish=ch.get("finish_reason"))
            except requests.RequestException as e:
                err = scrub(f"{type(e).__name__}: {e}", self._key)
                wait_s = min(60.0, 2.0 ** a)
            time.sleep(wait_s * (0.75 + 0.5 * np.random.random()))
        return dict(text=None, usage={}, latency_s=float("nan"), wall_s=time.time() - t_wall, attempts=attempts,
                    sent_at=sent_at, error=f"gave up after {attempts} attempts: {err}")

# ---------------------------------------------------------------------------------------------------------------- sampling / plan
def vrng(seed, key):
    return np.random.default_rng([seed, zlib.crc32(str(key).encode())])

def sample_queries(df: pd.DataFrame, QE: np.ndarray, vids_avail: list[str], seed: int, n_pos: int = N_POS, n_neg: int = N_NEG) -> list[dict]:
    """300 positives (<= 1 query per video, seeded) + 100 in-domain negatives (E3 construction: caption of ANOTHER video sharing no
    content word with ANY caption of the target video). Negative targets are videos not used for positives."""
    rng = np.random.default_rng(seed)
    avail = sorted(vids_avail)
    perm = rng.permutation(len(avail))
    pos_vids = [avail[i] for i in perm[:n_pos]]
    neg_vids = [avail[i] for i in perm[n_pos:n_pos + n_neg]]
    by_vid = {v: g.index.values for v, g in df.groupby("vid")}
    items = []
    for v in pos_vids:
        rows = by_vid[v]
        i = int(rows[rng.integers(len(rows))])
        items.append(dict(qid=int(df.qid.iloc[i]), kind="pos", vid=v, caption=df.caption.iloc[i], gs=float(df["gs"].iloc[i]),
                          ge=float(df["ge"].iloc[i]), qe=QE[i], src_qid=int(df.qid.iloc[i])))
    cw = [data.content_words(c) for c in df["caption"]]
    vid_arr = df["vid"].values
    for k, v in enumerate(neg_vids):
        vwords = set().union(*[cw[i] for i in by_vid[v]])
        cand = [i for i in range(len(df)) if vid_arr[i] != v and cw[i] and cw[i].isdisjoint(vwords)]
        i = cand[int(vrng(seed, v).integers(len(cand)))]
        items.append(dict(qid=100000 + k, kind="neg", vid=v, caption=df.caption.iloc[i], gs=float("nan"), ge=float("nan"),
                          qe=QE[i], src_qid=int(df.qid.iloc[i])))
    return items

def interleave(items: list[dict], conds: dict) -> list[tuple[str, dict]]:
    """Execution order = priority order, so that a budget stop drops the least important work first:
    (1) C1K3 positives, (2) C1K3 negatives, (3) C2K3 (positives, then negatives), (4) C1K1 positives.
    Within a phase queries run in (seeded random) sample order, so a partial phase is a random subset."""
    pos = [x for x in items if x["kind"] == "pos"]; neg = [x for x in items if x["kind"] == "neg"]
    phases = [("C1K3", pos), ("C1K3", neg), ("C2K3", pos), ("C2K3", neg), ("C1K1", pos)]
    return [(c, q) for c, qs in phases if c in conds for q in qs if not (q["kind"] == "neg" and not conds[c][3])]

def limit_items(items: list[dict], limit: int | None) -> list[dict]:
    if not limit:
        return items
    pos = [x for x in items if x["kind"] == "pos"][:limit]
    neg = [x for x in items if x["kind"] == "neg"][:max(1, limit // 3)]
    return pos + neg

def prepare(items, vids):
    """For every (condition, query): candidates + prompt parts. Pure CPU, no video decoding, no API."""
    sels, idxs = {}, {}
    plans = {}
    for cname, (method, params, K, _) in CONDITIONS.items():
        for q in items:
            if q["kind"] == "neg" and not CONDITIONS[cname][3]:
                continue
            v = q["vid"]
            emb, ts, dur = vids[v]
            key = (method, tuple(sorted(params.items())), v)
            if key not in sels:
                sel = select(method, emb, ts, dur, **params)
                sels[key] = (sel, *build_index(sel))
            sel, index, mapping = sels[key]
            qe = q["qe"] / max(np.linalg.norm(q["qe"]), 1e-12)
            hits = search(index, qe, mapping, TOP_HITS)
            sims = sel.emb @ qe
            cands = build_candidates(sel, hits, sims, dur)
            parts = prompt_parts(q["caption"], cands, K, dur)
            plans[(cname, q["qid"])] = dict(q=q, cands=cands, parts=parts, K=K, dur=dur, sha=request_sha(parts, K),
                                            n_kept=sel.n_kept, n_frames=len(emb))
        sels = {k: x for k, x in sels.items() if k[0] == method} if False else sels
    return plans

# ---------------------------------------------------------------------------------------------------------------- execution
def raw_path(out: Path, cname: str, qid: int) -> Path:
    return out / "raw" / f"{cname}_{qid}.json"

def load_cached(out: Path, cname: str, qid: int, sha: str):
    p = raw_path(out, cname, qid)
    if p.exists():
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("req_sha") == sha:
                return r
        except Exception:
            pass
    return None

def run_call(api: Api, plan: dict, cname: str, out: Path) -> dict:
    q = plan["q"]
    content, jpgs = materialize(plan["parts"], q["vid"])
    n_cand = len(plan["cands"])
    calls = []
    res = api.chat(content)
    choice, conf, status = parse_response(res["text"], n_cand)
    calls.append((res, choice, conf, status))
    if status != "ok" and res["error"] is None:                 # parse failure -> retry once (error already retried with backoff)
        content2 = content + [{"type": "text", "text": RETRY_SUFFIX}]
        res2 = api.chat(content2)
        choice, conf, status = parse_response(res2["text"], n_cand)
        calls.append((res2, choice, conf, status))
    rec_calls, tot_peak, tot_actual, in_t, out_t, lat = [], 0.0, 0.0, 0, 0, 0.0
    for r, ch, cf, st in calls:
        u = r["usage"] or {}
        it, ot = int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
        tier = price_tier(r["sent_at"])
        tot_peak += cost_usd(it, ot, "peak"); tot_actual += cost_usd(it, ot, tier)
        in_t += it; out_t += ot; lat += (r["latency_s"] if r["latency_s"] == r["latency_s"] else 0.0)
        rec_calls.append(dict(text=r["text"], error=r["error"], finish=r.get("finish"), usage=u, latency_s=r["latency_s"], wall_s=r["wall_s"],
                              attempts=r["attempts"], sent_at=r["sent_at"].isoformat(), tier=tier, choice=ch, confidence=cf, status=st))
    rec = dict(task=f"{cname}_{q['qid']}", cond=cname, qid=q["qid"], kind=q["kind"], vid=q["vid"], req_sha=plan["sha"], model=MODEL, K=plan["K"],
               n_images=len(jpgs), jpeg_bytes=int(sum(len(j) for j in jpgs)),
               cands=[dict(label=c["label"], t=c["t"]) for c in plan["cands"]],
               choice=choice, confidence=conf, status=status, n_calls=len(calls), input_tokens=in_t, output_tokens=out_t,
               latency_s=lat, cost_peak_usd=tot_peak, cost_actual_usd=tot_actual, calls=rec_calls)
    ok_any = any(c["error"] is None for c in rec_calls)
    if ok_any:                                                  # never cache pure transport failures: they should be re-tried on resume
        p = raw_path(out, cname, q["qid"]); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=1, default=lambda o: None if o != o else str(o)), encoding="utf-8"); tmp.replace(p)
    return rec

def execute(tasks, plans, out: Path, api: Api, budget: float):
    """Run uncached tasks, 3 at a time, stopping once estimated cumulative peak-price cost (+ in-flight worst case) reaches budget."""
    # spend of earlier abandoned trials (raw_trial*/ dirs) counts against the same budget
    cum, done_n = sum(json.loads(f.read_text(encoding="utf-8"))["cost_peak_usd"] for f in out.glob("raw_trial*/*.json")), 0
    todo = []
    for cname, q in tasks:
        r = load_cached(out, cname, q["qid"], plans[(cname, q["qid"])]["sha"])
        if r is not None:
            cum += r["cost_peak_usd"]
        else:
            todo.append((cname, q))
    print(f"[e8] {len(tasks)} calls planned, {len(tasks) - len(todo)} cached (spent ${cum:.4f} at peak price), {len(todo)} to run", flush=True)
    costs = []
    stopped = False
    inflight = {}
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        it = iter(todo)
        pending = True
        while pending or inflight:
            while pending and len(inflight) < CONCURRENCY:
                cname, q = next(it, (None, None))
                if cname is None:
                    pending = False; break
                plan = plans[(cname, q["qid"])]
                est = (np.mean(costs) if costs else cost_usd(estimate_tokens(plan["parts"]), 120, "peak"))
                if cum + est * (len(inflight) + 1) > budget:
                    stopped = True; pending = False; break
                inflight[ex.submit(run_call, api, plan, cname, out)] = (cname, q["qid"])
            if not inflight:
                break
            d, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in d:
                cname, qid = inflight.pop(f)
                try:
                    rec = f.result()
                except Exception as e:
                    print(f"[e8] {cname}_{qid} FAILED: {scrub(repr(e), None)[:200]}", flush=True); continue
                cum += rec["cost_peak_usd"]; costs.append(rec["cost_peak_usd"]); done_n += 1
                if done_n % 10 == 0 or done_n <= 5:
                    print(f"  [{done_n}/{len(todo)}] {rec['task']} choice={rec['choice']} in={rec['input_tokens']} out={rec['output_tokens']} "
                          f"lat={rec['latency_s']:.1f}s cum_peak=${cum:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if stopped:
        print(f"[e8] BUDGET STOP: estimated cumulative cost (peak price) ${cum:.4f} reached the ${budget:.2f} cap; "
              f"{len(todo) - done_n} calls NOT issued. Results below cover the completed queries only.", flush=True)
    return cum, stopped

# ---------------------------------------------------------------------------------------------------------------- dry run
def draw_grid(parts, jpgs, title):
    import cv2
    rows, k, cur, imgs = [], 0, None, []
    for p in parts:
        if p["type"] == "img":
            imgs.append((p["label"], p["t"], cv2.imdecode(np.frombuffer(jpgs[k], np.uint8), cv2.IMREAD_COLOR))); k += 1
    labels = list(dict.fromkeys(l for l, _, _ in imgs))
    th = 150
    row_imgs = []
    for lab in labels:
        tiles = []
        for l, t, im in imgs:
            if l != lab: continue
            h, w = im.shape[:2]; s = th / h
            tile = cv2.resize(im, (max(1, int(w * s)), th))
            cv2.putText(tile, f"{t:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2); cv2.putText(tile, f"{t:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            tiles.append(tile)
        lab_tile = np.full((th, 50, 3), 40, np.uint8); cv2.putText(lab_tile, lab, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        row_imgs.append(np.hstack([lab_tile] + tiles))
    W = max(r.shape[1] for r in row_imgs)
    row_imgs = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=40) for r in row_imgs]
    head = np.full((40, W, 3), 255, np.uint8)
    cv2.putText(head, title[:int(W / 9)], (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return np.vstack([head] + row_imgs)

def dry_run(plans_all, plans_lim, tasks_lim, out: Path):
    import cv2
    ex_dir = out / "examples"; ex_dir.mkdir(parents=True, exist_ok=True)
    nb, sizes = 0, []
    chosen, seen = [], set()
    for want in (("C1K3", "pos"), ("C2K3", "pos"), ("C1K3", "neg")):
        for cname, q in tasks_lim:
            if (cname, q["kind"]) == want and (cname, q["qid"]) not in seen:
                chosen.append((cname, q["qid"])); seen.add((cname, q["qid"])); break
    for cname, q in tasks_lim:                                   # decode a sample of requests to validate seeking/encoding
        plan = plans_lim[(cname, q["qid"])]
        content, jpgs = materialize(plan["parts"], q["vid"])
        sizes += [len(j) for j in jpgs]; nb += 1
        if (cname, q["qid"]) in chosen:
            g = draw_grid(plan["parts"], jpgs, f"{cname} qid={q['qid']} [{q['kind']}] \"{q['caption']}\"" + (f"  GT {q['gs']:.1f}-{q['ge']:.1f}s" if q['kind'] == 'pos' else "  (absent event)"))
            cv2.imwrite(str(ex_dir / f"{cname}_{q['qid']}.jpg"), g)
    print(f"[dry-run] decoded {nb} requests OK; mean JPEG {np.mean(sizes) / 1024:.1f} KiB, mean base64 request {np.mean(sizes) * 4 / 3 / 1024:.1f} KiB/image")
    for label, pl in (("--limit subset", plans_lim), ("FULL plan (300 pos + 100 neg)", plans_all)):
        by = {}
        for (cname, qid), p in pl.items():
            b = by.setdefault(cname, [0, 0, 0]); b[0] += 1; b[1] += sum(1 for x in p["parts"] if x["type"] == "img"); b[2] += estimate_tokens(p["parts"])
        print(f"[dry-run] {label}: calls / images / worst-case input tokens (1,024 per image)")
        tot_c = tot = 0.0
        for cname, (n, ni, nt) in by.items():
            c_peak = cost_usd(nt, 40 * n, "peak"); c_off = cost_usd(nt, 40 * n, "off")
            tot += c_peak; tot_c += n
            print(f"    {cname}: {n} calls, {ni} images, {nt:,} tok -> worst-case ${c_peak:.2f} peak / ${c_off:.2f} off-peak (40 output tok/call)")
        print(f"    total worst-case ${tot:.2f} peak; budget cap ${BUDGET_USD:.2f} -> {'FITS' if tot <= BUDGET_USD else 'EXCEEDS (run will stop at the cap, completing a random subset)'}")
        if tot > 0:
            print(f"    if images really bill at X tokens: cost scales ~linearly (e.g. 256 tok/img ~ ${tot * (256 / 1024):.2f} peak)")

# ---------------------------------------------------------------------------------------------------------------- aggregation
def contains(t, gs, ge): return (t >= gs) & (t <= ge)

def build_per_query(out: Path, plans) -> pd.DataFrame:
    rows = []
    for (cname, qid), p in plans.items():
        r = load_cached(out, cname, qid, p["sha"])
        if r is None:
            continue
        q, cands = p["q"], p["cands"]
        ch = r["choice"]; pos = q["kind"] == "pos"
        lab_idx = {c["label"]: i for i, c in enumerate(cands)}
        sel_c = cands[lab_idx[ch]] if ch in lab_idx else None
        a = cands[0]
        row = dict(cond=cname, qid=qid, kind=q["kind"], vid=q["vid"], caption=q["caption"], gt_s=q["gs"], gt_e=q["ge"], K=p["K"],
                   n_cands=len(cands), cand_times=";".join(f"{c['t']:.2f}" for c in cands),
                   cand_src=";".join(c["src"] for c in cands), choice=ch, confidence=r["confidence"], status=r["status"], n_calls=r["n_calls"],
                   input_tokens=r["input_tokens"], output_tokens=r["output_tokens"], latency_s=r["latency_s"],
                   cost_actual_usd=r["cost_actual_usd"], cost_peak_usd=r["cost_peak_usd"], tier=r["calls"][-1]["tier"])
        if pos:
            gs, ge = q["gs"], q["ge"]
            def feats(c, pre):
                row[pre + "_hit1"] = bool(c is not None and gs <= c["t"] <= ge)
                for nm, (s, e) in (("span", (c["span_s"], c["span_e"]) if c else (np.nan, np.nan)), ("wide", (c["wide_s"], c["wide_e"]) if c else (np.nan, np.nan))):
                    iou = float(iou_arrays([s], [e], [gs], [ge])[0]) if c else 0.0
                    row[f"{pre}_iou_{nm}"] = iou
            feats(a, "fast")
            feats(sel_c, "slow")                                 # NONE / invalid -> counted as a miss
            feats(sel_c if sel_c is not None else a, "slowfb")   # NONE / invalid -> fall back to the fast path's top-1
            row["oracle_hit5"] = bool(any(gs <= c["t"] <= ge for c in cands))
            best = [float(iou_arrays([c["wide_s"]], [c["wide_e"]], [gs], [ge])[0]) for c in cands]
            row["oracle_iou_wide"] = max(best)
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate(out: Path, pq: pd.DataFrame, spent_peak: float | None):
    rows = []
    def ci(x, idx): return bootstrap_ci(np.asarray(x, dtype=float), idx)
    for cname in CONDITIONS:
        g = pq[(pq.cond == cname)]
        P, N = g[g.kind == "pos"].sort_values("qid"), g[g.kind == "neg"].sort_values("qid")
        if len(P) == 0:
            continue
        idx = bootstrap_indices(len(P))
        base = dict(condition=cname, K=CONDITIONS[cname][2], index=("full 5 FPS" if CONDITIONS[cname][0] == "full_5fps" else "cos_last_drop tau=0.95"))
        def metric_row(system, pre):
            r = dict(base, system=system, n_pos=len(P))
            r["hit1"] = P[f"{pre}_hit1"].mean(); r["hit1_lo"], r["hit1_hi"] = ci(P[f"{pre}_hit1"], idx)
            for nm in ("span", "wide"):
                for th in (0.3, 0.5):
                    x = P[f"{pre}_iou_{nm}"] >= th
                    r[f"r1_iou{int(th * 10):02d}_{nm}"] = x.mean()
                    if th == 0.5: r[f"r1_iou05_{nm}_lo"], r[f"r1_iou05_{nm}_hi"] = ci(x, idx)
            return r
        fast, slow, slowfb = metric_row("fast_top1", "fast"), metric_row("slow_vlm", "slow"), metric_row("slow_vlm_fallback_to_fast", "slowfb")
        slow["none_rate_pos"] = (P.choice == "NONE").mean(); slow["none_rate_pos_lo"], slow["none_rate_pos_hi"] = ci(P.choice == "NONE", idx)
        slow["invalid_rate_pos"] = (P.status != "ok").mean()
        orc = dict(base, system="oracle_best_of_5", n_pos=len(P), hit1=P.oracle_hit5.mean())
        orc["hit1_lo"], orc["hit1_hi"] = ci(P.oracle_hit5, idx); orc["r1_iou05_wide"] = (P.oracle_iou_wide >= 0.5).mean()
        orc["r1_iou05_wide_lo"], orc["r1_iou05_wide_hi"] = ci(P.oracle_iou_wide >= 0.5, idx)
        # paired differences (slow - fast)
        d = dict(base, system="delta_slow_minus_fast", n_pos=len(P))
        for nm, col in (("hit1", "hit1"), ("r1_iou05_span", "iou_span"), ("r1_iou05_wide", "iou_wide")):
            a = (P[f"slow_{col}"] >= 0.5).astype(float) if col != "hit1" else P["slow_hit1"].astype(float)
            b = (P[f"fast_{col}"] >= 0.5).astype(float) if col != "hit1" else P["fast_hit1"].astype(float)
            d[nm] = (a - b).mean(); d[nm + "_lo"], d[nm + "_hi"] = ci(a.values - b.values, idx)
        # negatives
        if len(N):
            nidx = bootstrap_indices(len(N))
            slow["n_neg"] = len(N); slow["correct_none_neg"] = (N.choice == "NONE").mean(); slow["correct_none_neg_lo"], slow["correct_none_neg_hi"] = ci(N.choice == "NONE", nidx)
            slow["invalid_rate_neg"] = (N.status != "ok").mean()
            fast["n_neg"] = len(N); fast["correct_none_neg"] = 0.0           # fast path always returns a best match
        # cost / latency (all calls of this condition)
        allc = g
        for r in (slow,):
            r["latency_mean_s"], r["latency_p50_s"], r["latency_p95_s"] = allc.latency_s.mean(), allc.latency_s.median(), allc.latency_s.quantile(0.95)
            r["input_tokens_mean"], r["output_tokens_mean"] = allc.input_tokens.mean(), allc.output_tokens.mean()
            r["tokens_per_image_mean"] = allc.input_tokens.sum() / max(1, (allc.K * allc.n_cands).sum())
            r["cost_per_query_actual_usd"], r["cost_per_query_peak_usd"] = allc.cost_actual_usd.mean(), allc.cost_peak_usd.mean()
            r["cost_per_query_offpeak_usd"] = float(np.mean([cost_usd(i, o, "off") for i, o in zip(allc.input_tokens, allc.output_tokens)]))
            r["n_calls_total"] = int(allc.n_calls.sum()); r["n_queries_total"] = len(allc)
        rows += [fast, slow, slowfb, orc, d]
    sm = pd.DataFrame(rows)
    # C2 vs C1 paired on common queries (effect of pruning on the slow path)
    extra = []
    for ca, cb in (("C2K3", "C1K3"),):
        A = pq[(pq.cond == ca) & (pq.kind == "pos")].set_index("qid"); B = pq[(pq.cond == cb) & (pq.kind == "pos")].set_index("qid")
        common = A.index.intersection(B.index)
        if len(common):
            idx = bootstrap_indices(len(common)); r = dict(condition=f"{ca}_minus_{cb}", system="delta_pruned_minus_full(slow)", n_pos=len(common), K=3)
            for nm, col in (("hit1", "slow_hit1"), ("r1_iou05_wide", "slow_iou_wide")):
                x = A.loc[common, col]; y = B.loc[common, col]
                if nm != "hit1": x, y = (x >= 0.5), (y >= 0.5)
                dv = x.astype(float).values - y.astype(float).values
                r[nm] = dv.mean(); r[nm + "_lo"], r[nm + "_hi"] = ci(dv, idx)
            extra.append(r)
    if extra:
        sm = pd.concat([sm, pd.DataFrame(extra)], ignore_index=True)
    if spent_peak is not None:
        sm["run_cost_peak_usd_total"] = spent_peak
    return sm

def make_figure(sm: pd.DataFrame, path: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in ("C1K3", "C2K3", "C1K1") if ((sm.condition == c) & (sm.system == "slow_vlm")).any()]
    if not conds:
        return
    names = {"C1K3": "C1: full index\nK=3 frames", "C2K3": "C2: pruned (tau .95)\nK=3 frames", "C1K1": "C1: full index\nK=1 frame"}
    col = {"fast": "#9aa0a6", "slow": "#0072B2", "fallback": "#56B4E9"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True)
    for ax, (m, lo, hi, ttl) in zip(axes, (("hit1", "hit1_lo", "hit1_hi", "Hit@1 (chosen candidate time inside GT)"),
                                           ("r1_iou05_wide", "r1_iou05_wide_lo", "r1_iou05_wide_hi", "R@1 IoU>=0.5 (candidate widened to +-4 s)"))):
        x = np.arange(len(conds)); w = 0.26
        for j, (sysn, lab, c) in enumerate((("fast_top1", "Fast path (top-1)", col["fast"]), ("slow_vlm", "Slow: VLM (NONE = miss)", col["slow"]),
                                            ("slow_vlm_fallback_to_fast", "Slow: VLM, NONE -> fast top-1", col["fallback"]))):
            vals, errs = [], []
            for cn in conds:
                r = sm[(sm.condition == cn) & (sm.system == sysn)].iloc[0]
                vals.append(100 * r[m]); errs.append([100 * (r[m] - r[lo]), 100 * (r[hi] - r[m])])
            b = ax.bar(x + (j - 1) * w, vals, w, color=c, label=lab, yerr=np.array(errs).T, capsize=3, error_kw=dict(lw=1, color="#333"))
            for xi, v in zip(x + (j - 1) * w, vals): ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([names[c] for c in conds], fontsize=9); ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    lines = []
    for cn in conds:
        r = sm[(sm.condition == cn) & (sm.system == "slow_vlm")].iloc[0]
        lines.append(f"{cn}: ${r['cost_per_query_actual_usd']:.4f}/query, {r['latency_mean_s']:.1f} s mean latency (p95 {r['latency_p95_s']:.1f} s), "
                     f"{r['input_tokens_mean']:.0f} in-tok, n={int(r['n_pos'])} pos")
    fig.text(0.01, 0.01, "Slow path cost/latency - " + " | ".join(lines), fontsize=7.5, color="#444", wrap=True)
    fig.suptitle("E8: VLM verification (slow path) vs fast path, Charades-STA", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(path, dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="E8 slow-path VLM verification")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--limit", type=int, default=None, help="N positives (and N//3 negatives)")
    ap.add_argument("--list-models", action="store_true"); ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--seed", type=int, default=8); ap.add_argument("--budget", type=float, default=BUDGET_USD)
    a = ap.parse_args()
    out = C.results_dir(EXP)
    if a.list_models:
        ids = Api(load_key()).models(); print("models:", ids); print(f"{MODEL!r} listed: {MODEL in ids}"); return

    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df))
    items_all = sample_queries(df, QE, list(vids), a.seed)
    items = limit_items(items_all, a.limit)
    print(f"[e8] sampled {sum(i['kind'] == 'pos' for i in items_all)} pos / {sum(i['kind'] == 'neg' for i in items_all)} neg "
          f"(using {sum(i['kind'] == 'pos' for i in items)} / {sum(i['kind'] == 'neg' for i in items)})", flush=True)
    plans_all = prepare(items_all, vids)
    keys = {(q["qid"], q["kind"]) for q in items}
    plans = {k: v for k, v in plans_all.items() if (v["q"]["qid"], v["q"]["kind"]) in keys}
    tasks = interleave(items, CONDITIONS)

    if a.dry_run:
        dry_run(plans_all, plans, tasks, out); return
    spent = None
    if not a.aggregate_only:
        if not (KEY_FILE.exists() or os.environ.get("DEEPSEEK_API_KEY")):
            print(f"[e8] no API key (env DEEPSEEK_API_KEY or {KEY_FILE}); nothing to run (use --dry-run or --aggregate-only)"); return
        api = Api(load_key())
        spent, stopped = execute(tasks, plans, out, api, a.budget)
    pq = build_per_query(out, plans)
    if pq.empty:
        print("[e8] no completed calls; nothing to aggregate"); return
    pq = pq.sort_values(["cond", "kind", "qid"]).reset_index(drop=True)
    pq.to_csv(out / "per_query.csv", index=False, encoding="utf-8")
    sm = aggregate(out, pq, spent)
    sm.to_csv(out / "summary.csv", index=False, encoding="utf-8")
    (C.RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    make_figure(sm, C.RESULTS_DIR / "figures" / "e8_fast_vs_slow.png")
    runutil.write_meta(out, limit=a.limit, model=MODEL, seed=a.seed, budget_usd=a.budget)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(sm[["condition", "system", "n_pos", "hit1", "hit1_lo", "hit1_hi", "r1_iou05_span", "r1_iou05_wide"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
