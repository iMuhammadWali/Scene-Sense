"""Tests for E8: response parser, candidate builder, frame-time picker, pricing, key scrubbing."""
import importlib.util, datetime
from pathlib import Path
import numpy as np, pytest
spec = importlib.util.spec_from_file_location("e8_run", Path(__file__).resolve().parents[1] / "experiments" / "e8_slowpath" / "run.py")
e8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e8)
from sseval.selection import from_indices

@pytest.mark.parametrize("text,exp", [
    ('{"choice": "B", "confidence": 0.8}', ("B", 0.8, "ok")),
    ('```json\n{"choice": "none", "confidence": 0.9}\n```', ("NONE", 0.9, "ok")),
    ('Sure! {"choice":"C","confidence":1.7} hope that helps', ("C", 1.0, "ok")),
    ("{'choice': 'a', 'confidence': 0.5}", ("A", 0.5, "ok")),
    ('choice: D, confidence: 0.3', ("D", 0.3, "ok")),
    ('{"choice": "A"}', ("A", None, "ok")),
    ('{"choice": "Clip B", "confidence": 0.2}', ("B", 0.2, "ok")),
])
def test_parse_ok(text, exp):
    ch, cf, st = e8.parse_response(text)
    assert (ch, st) == (exp[0], exp[2])
    assert (np.isnan(cf) if exp[1] is None else cf == pytest.approx(exp[1]))

@pytest.mark.parametrize("text", [None, "", "I think it is the second one", '{"choice": "Z"}', '{"answer": "A"}', '{"choice": "E"}'])
def test_parse_invalid(text):
    n = 4 if text == '{"choice": "E"}' else 5          # E is out of range when only 4 candidates
    assert e8.parse_response(text, n)[2] == "invalid"

def _sel(ts, dur=30.0):
    ts = np.asarray(ts, float)
    return from_indices(np.eye(len(ts), 4, dtype=np.float32) if len(ts) <= 4 else np.zeros((len(ts), 4), np.float32), ts, dur, np.arange(len(ts)))

def _hits(sel, sims, k=20):
    order = np.argsort(-sims, kind="stable")[:k]
    return [dict(rank=r + 1, faiss_index=int(i), frame_id=int(i), timestamp=float(sel.ts[i]), similarity=float(sims[i])) for r, i in enumerate(order)]

def test_candidates_groups_ranked_and_distinct():
    ts = np.arange(0, 30, 0.2); sel = _sel(ts)
    sims = np.full(len(ts), 0.1)
    for centre, h in ((5.0, .30), (12.0, .28), (20.0, .26), (26.0, .24), (1.0, .22), (16.0, .21)):
        i = int(round(centre / .2)); sims[i] = h; sims[i - 1] = h - .01; sims[i + 1] = h - .01   # 3-frame group
    c = e8.build_candidates(sel, _hits(sel, sims), sims, 30.0)
    assert len(c) == 5 and [x["label"] for x in c] == list("ABCDE")
    assert [round(x["t"]) for x in c[:4]] == [5, 12, 20, 26]                        # ranked by best hit, t = best-hit time
    assert c[0]["span_e"] - c[0]["span_s"] >= 0.4 and c[0]["span_s"] <= 4.8 and c[0]["span_e"] >= 5.2
    assert all(abs(a["t"] - b["t"]) > 2 for i, a in enumerate(c) for b in c[:i])
    assert c[0]["wide_s"] == pytest.approx(1.0) and c[0]["wide_e"] == pytest.approx(9.0)

def test_candidates_pad_and_clip():
    ts = np.arange(0, 12, 0.2); sel = _sel(ts, 12.0)
    sims = np.linspace(0.3, 0.1, len(ts))            # monotone: top-20 hits form ONE group at the start -> must pad
    c = e8.build_candidates(sel, _hits(sel, sims), sims, 12.0)
    assert c[0]["src"] == "group" and all(x["src"] == "pad" for x in c[1:])
    assert len(c) == 4                                # only 4 peaks >= 3 s apart exist in 12 s (0, 3.x, 6.x, 9.x)
    ts_ = [x["t"] for x in c]
    assert all(abs(a - b) >= 3 - 1e-9 for i, a in enumerate(ts_) for b in ts_[:i])
    assert c[0]["wide_s"] == 0.0 and c[-1]["wide_e"] <= 12.0

def test_clip_times():
    assert e8.clip_times(12.4, 30, 3) == [7.4, 12.4, 17.4]
    assert e8.clip_times(12.4, 30, 1) == [12.4]
    a = e8.clip_times(1.0, 30, 3); assert a[0] == 0.0 and a == sorted(a) and len(a) == 3     # window clipped at 0
    b = e8.clip_times(29.0, 29.5, 3); assert max(b) <= 29.4
    assert len(e8.clip_times(5.0, 30, 20)) <= e8.MAX_FRAMES_REAL                              # whole 1 FPS window <= real cap

def test_prompt_structure():
    c = [dict(label=l, t=float(i)) for i, l in enumerate("ABCDE")]
    p = e8.prompt_parts("person opens door", c, 3, 30.0)
    assert sum(x["type"] == "img" for x in p) <= 15
    assert any(x["type"] == "text" and x["text"].startswith("Clip A (t=0.0 s):") for x in p)
    assert "NONE" in p[-1]["text"] and "person opens door" in p[-1]["text"]

def test_pricing_and_scrub():
    peak = datetime.datetime(2026, 11, 13, 7, 0, tzinfo=datetime.timezone.utc)      # Friday
    assert e8.price_tier(peak) == "peak"
    assert e8.price_tier(datetime.datetime(2026, 11, 14, 7, 0, tzinfo=datetime.timezone.utc)) == "off"   # Saturday
    assert e8.price_tier(datetime.datetime(2026, 11, 13, 12, 0, tzinfo=datetime.timezone.utc)) == "off"
    assert e8.cost_usd(1_000_000, 1_000_000, "peak") == pytest.approx(1.5)
    assert "sk-SECRET" not in e8.scrub("fail with sk-SECRET and Bearer sk-SECRET", "sk-SECRET")
