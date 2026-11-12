"""Annotation, query-embedding and cached frame-embedding loaders."""
from __future__ import annotations
import os, re
import numpy as np, pandas as pd
from . import config as C
from .selection import l2n

def load_annotations() -> pd.DataFrame:
    """One row per query: qid (row order of the parquet), vid, caption, gs, ge."""
    df = pd.read_parquet(C.ANNOTATIONS).reset_index(drop=True)
    df["qid"] = np.arange(len(df))
    df["vid"] = df["video"].str.replace(".mp4", "", regex=False)
    df["gs"] = df["timestamp"].map(lambda x: float(x[0]))
    df["ge"] = df["timestamp"].map(lambda x: float(x[1]))
    return df[["qid", "vid", "caption", "gs", "ge"]]

def _encode_with_team(texts: list[str]) -> np.ndarray:
    """Encode text with the team's loaded CLIP model, applying text_projection exactly ONCE.

    KNOWN ISSUE in the team's CLIPEncoder.encode_text (observed with transformers 4.57): get_text_features already returns the
    projected 512-d embedding, but _as_feature_tensor re-applies text_projection because the text width (512) equals
    in_features -> double projection, text embeddings are ~uncorrelated with image space (cos ~0.1 with the correct vector,
    max image-text cosine ~0.05 instead of ~0.3). Set SSEVAL_TEAM_TEXT=1 to use the team's encode_text unchanged.
    """
    C.setup_team()
    import torch
    from app.pipeline.clip_encoder import CLIPEncoder
    enc = CLIPEncoder.get(); enc.load()
    if os.environ.get("SSEVAL_TEAM_TEXT") == "1":
        return np.stack([enc.encode_text(t) for t in texts]).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            inp = enc.processor(text=texts[i:i + 64], return_tensors="pt", padding=True, truncation=True)
            raw = enc.model.get_text_features(**{k: v.to(enc.device) for k, v in inp.items()})
            if not torch.is_tensor(raw):                      # newer transformers: pooled, pre-projection output
                raw = enc.model.text_projection(raw.pooler_output)
            out.append(raw.cpu().numpy().astype(np.float32))
    return l2n(np.concatenate(out))

def load_query_emb(df: pd.DataFrame) -> np.ndarray:
    """Query text embeddings aligned to df rows. Uses the cache if its captions match df exactly, else re-encodes."""
    if C.QUERY_EMB_CACHE.exists():
        z = np.load(C.QUERY_EMB_CACHE, allow_pickle=True)
        if len(z["emb"]) == len(df) and (z["caption"] == df["caption"].values).all():
            return z["emb"].astype(np.float32)
    uniq = sorted(df["caption"].unique())
    m = dict(zip(uniq, _encode_with_team(uniq)))
    emb = np.stack([m[c] for c in df["caption"]]).astype(np.float32)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(C.QUERY_EMB_CACHE, emb=emb, caption=df["caption"].values)
    return emb

def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode arbitrary new text with the team's CLIPEncoder; cached by string in EXTRA_TEXT_CACHE."""
    cache = {}
    if C.EXTRA_TEXT_CACHE.exists():
        z = np.load(C.EXTRA_TEXT_CACHE, allow_pickle=True)
        cache = dict(zip(z["text"].tolist(), z["emb"]))
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        for t, e in zip(missing, _encode_with_team(missing)):
            cache[t] = e
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(C.EXTRA_TEXT_CACHE, text=np.array(list(cache.keys()), dtype=object),
                 emb=np.stack(list(cache.values())).astype(np.float32))
    return np.stack([cache[t] for t in texts]).astype(np.float32)

def list_available_videos(df: pd.DataFrame | None = None) -> list[str]:
    """Sorted video ids that have a complete cached embedding file AND annotations."""
    have = {p.name[:-4] for p in C.EMB_DIR.glob("*.npz") if not p.name.endswith(".tmp.npz")}
    if df is not None:
        have &= set(df["vid"].unique())
    return sorted(have)

def load_video(vid: str):
    """-> (emb [N,512] float32 L2-normalised, ts [N] float64, duration float)."""
    d = np.load(C.EMB_DIR / f"{vid}.npz")
    return d["emb"].astype(np.float32), d["ts"].astype(np.float64), float(d["duration"])

def load_videos(vids: list[str]) -> dict:
    out = {}
    for v in vids:
        try:
            emb, ts, dur = load_video(v)
        except Exception as e:           # unreadable/partial file: skip, never crash a long run
            print(f"[data] skip unreadable {v}: {e}", flush=True); continue
        if len(emb) == 0:
            continue
        out[v] = (emb, ts, dur)
    return out

def select_videos(df: pd.DataFrame, limit: int | None = None) -> list[str]:
    """Deterministic subset: the first `limit` available videos in sorted order (all if None)."""
    vids = list_available_videos(df)
    return vids[:limit] if limit else vids

# --- caption text utilities (E3) ----------------------------------------------------------------
STOP = set("""a an the and or of to in on at with from by for as is are was were be been being it its this that these those
his her their them they he she someone person people there then than into onto up down out off over under it's while
another other one two some any each both very just also not no so up""".split())
def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    out = set()
    for w in words:
        if w in STOP:
            continue
        if w.endswith("ing") and len(w) > 5: w = w[:-3]
        elif w.endswith("ed") and len(w) > 4: w = w[:-2]
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3: w = w[:-1]
        if w and w not in STOP:
            out.add(w)
    return out
