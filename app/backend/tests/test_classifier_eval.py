"""BART classifier evaluation: evidence retrieval (SIMPLE) vs NL answer (COMPLEX)."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.pipeline.classifier import (
    HYPOTHESIS_TEMPLATE,
    LABEL_COMPLEX,
    LABEL_SIMPLE,
    QueryClassifier,
)
from app.utils.model_cache import model_cached

# Core eval set from the SceneSense classifier brief
EVAL_CASES: list[tuple[str, str]] = [
    # SIMPLE — visual evidence only
    ("Show me the person.", "SIMPLE"),
    ("Find the red car.", "SIMPLE"),
    ("Show me the frame containing the dog.", "SIMPLE"),
    ("Point out the bicycle.", "SIMPLE"),
    ("Find the man wearing a blue shirt.", "SIMPLE"),
    ("Show me the vehicle.", "SIMPLE"),
    ("Locate the person.", "SIMPLE"),
    ("Show me where the car appears.", "SIMPLE"),
    ("Find the frame where the person jumps.", "SIMPLE"),
    ("Show me the red vehicle.", "SIMPLE"),
    # COMPLEX — natural-language answer
    ("What is happening in the video?", "COMPLEX"),
    ("What is the person doing?", "COMPLEX"),
    ("What happens after the car appears?", "COMPLEX"),
    ("What happens before the person enters?", "COMPLEX"),
    ("How many vehicles passed in the video?", "COMPLEX"),
    ("How many vehicles appear throughout the video?", "COMPLEX"),
    ("How many times does the person jump?", "COMPLEX"),
    ("What color is the person's shirt?", "COMPLEX"),
    ("What is the person holding?", "COMPLEX"),
    ("How does the scene change?", "COMPLEX"),
]

CONTRASTIVE_PAIRS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("Show me the person.", "SIMPLE"), ("What is the person doing?", "COMPLEX")),
    (("Show me the red car.", "SIMPLE"), ("What color is the car?", "COMPLEX")),
    (
        ("Find the frame where the person jumps.", "SIMPLE"),
        ("How many times does the person jump?", "COMPLEX"),
    ),
    (("Show me the cars.", "SIMPLE"), ("How many cars are there?", "COMPLEX")),
    (
        ("Show me the frame where the vehicle appears.", "SIMPLE"),
        ("What happens after the vehicle appears?", "COMPLEX"),
    ),
    (
        ("Locate the person wearing blue.", "SIMPLE"),
        ("What color is the person's shirt?", "COMPLEX"),
    ),
]

SCENESENSE_QUERIES: list[tuple[str, str]] = [
    ("what is colour of the clothe the person is wearing", "COMPLEX"),
    ("what is happening in this video", "COMPLEX"),
    ("how many vehicles are there in the video", "COMPLEX"),
    ("show me the vehicle", "SIMPLE"),
    ("show me the person wearing blue", "SIMPLE"),
    ("what is the person doing", "COMPLEX"),
]


def _require_bart() -> QueryClassifier:
    if not model_cached(get_settings().classifier_model):
        pytest.skip("BART classifier weights not cached")
    QueryClassifier._instance = None
    return QueryClassifier.get()


def test_label_definitions_are_output_modality():
    assert "evidence" in LABEL_SIMPLE.lower()
    assert "natural-language" in LABEL_COMPLEX.lower()
    assert "This example is" in HYPOTHESIS_TEMPLATE


@pytest.mark.parametrize("question,expected", EVAL_CASES, ids=[c[0] for c in EVAL_CASES])
def test_eval_dataset(question: str, expected: str):
    clf = _require_bart()
    result, info = clf.classify(question)
    assert result == expected, (
        f"{question!r}: expected {expected}, got {result} "
        f"(simple={info['simple_confidence']}, complex={info['complex_confidence']})"
    )


@pytest.mark.parametrize(
    "simple_q,complex_q",
    [(a[0], b[0]) for a, b in CONTRASTIVE_PAIRS],
    ids=[f"pair_{i+1}" for i in range(len(CONTRASTIVE_PAIRS))],
)
def test_contrastive_pairs(simple_q: str, complex_q: str):
    clf = _require_bart()
    s_pred, s_info = clf.classify(simple_q)
    c_pred, c_info = clf.classify(complex_q)
    assert s_pred == "SIMPLE", (
        f"{simple_q!r} → {s_pred} "
        f"(simple={s_info['simple_confidence']}, complex={s_info['complex_confidence']})"
    )
    assert c_pred == "COMPLEX", (
        f"{complex_q!r} → {c_pred} "
        f"(simple={c_info['simple_confidence']}, complex={c_info['complex_confidence']})"
    )


@pytest.mark.parametrize(
    "question,expected",
    SCENESENSE_QUERIES,
    ids=[c[0] for c in SCENESENSE_QUERIES],
)
def test_scenesense_live_queries(question: str, expected: str):
    clf = _require_bart()
    result, info = clf.classify(question)
    assert result == expected, (
        f"{question!r}: expected {expected}, got {result} "
        f"(simple={info['simple_confidence']}, complex={info['complex_confidence']})"
    )


def test_eval_accuracy_report(capsys):
    """Print full report; assert overall accuracy on the 20-case set."""
    clf = _require_bart()
    rows = []
    correct = 0
    for q, expected in EVAL_CASES:
        pred, info = clf.classify(q)
        ok = pred == expected
        if ok:
            correct += 1
        rows.append(
            {
                "q": q,
                "expected": expected,
                "pred": pred,
                "ok": ok,
                "simple": info["simple_confidence"],
                "complex": info["complex_confidence"],
            }
        )

    accuracy = correct / len(EVAL_CASES)
    print("\n=== CLASSIFIER EVAL (20 cases) ===")
    for r in rows:
        mark = "OK" if r["ok"] else "FAIL"
        print(
            f"[{mark}] {r['expected']:7} → {r['pred']:7}  "
            f"simple={r['simple']:.4f} complex={r['complex']:.4f}  | {r['q']}"
        )
    print(f"Accuracy: {correct}/{len(EVAL_CASES)} = {accuracy:.1%}")

    print("\n=== CONTRASTIVE PAIRS ===")
    pair_ok = 0
    for i, (a, b) in enumerate(CONTRASTIVE_PAIRS, 1):
        pa, ia = clf.classify(a[0])
        pb, ib = clf.classify(b[0])
        ok = pa == a[1] and pb == b[1]
        if ok:
            pair_ok += 1
        print(
            f"Pair {i}: {'OK' if ok else 'FAIL'}\n"
            f"  {a[0]!r} → {pa} (s={ia['simple_confidence']:.4f} c={ia['complex_confidence']:.4f})\n"
            f"  {b[0]!r} → {pb} (s={ib['simple_confidence']:.4f} c={ib['complex_confidence']:.4f})"
        )
    print(f"Contrastive: {pair_ok}/{len(CONTRASTIVE_PAIRS)}")

    print("\n=== SCENESENSE QUERIES ===")
    for q, expected in SCENESENSE_QUERIES:
        pred, info = clf.classify(q)
        mark = "OK" if pred == expected else "FAIL"
        print(
            f"[{mark}] {expected:7} → {pred:7}  "
            f"simple={info['simple_confidence']:.4f} complex={info['complex_confidence']:.4f}  | {q}"
        )

    assert accuracy == 1.0, f"Eval accuracy {accuracy:.1%} < 100%"
    assert pair_ok == len(CONTRASTIVE_PAIRS)
