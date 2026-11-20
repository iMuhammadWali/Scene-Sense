"""Query-time retrieval filtering (separate from pruning threshold)."""

from __future__ import annotations

from typing import Any


def filter_by_similarity(
    hits: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep FAISS hits with similarity >= retrieval threshold.

    If none pass, fall back to the best match — but mark that clearly so
    debug UI never pretends the fallback passed the threshold.
    """
    candidate_count = len(hits)
    best_similarity = (
        max(float(h.get("similarity", 0.0)) for h in hits) if hits else None
    )
    passed = [h for h in hits if float(h.get("similarity", 0.0)) >= threshold]
    passed_count = len(passed)
    fallback_used = False

    if passed:
        filtered = [
            {
                **h,
                "passed_threshold": True,
                "fallback": False,
            }
            for h in passed
        ]
    elif hits:
        # Safety: keep best FAISS hit so downstream still has evidence
        best = dict(hits[0])
        best["passed_threshold"] = False
        best["fallback"] = True
        filtered = [best]
        fallback_used = True
    else:
        filtered = []

    info = {
        "stage": "similarity_filtering",
        "status": "completed",
        "input_count": candidate_count,
        "output_count": len(filtered),
        "threshold": threshold,
        "candidate_count": candidate_count,
        "passed_count": passed_count,
        "fallback_used": fallback_used,
        "best_similarity": (
            round(float(best_similarity), 4) if best_similarity is not None else None
        ),
        "fallback_reason": (
            "No candidates met the similarity threshold; best FAISS match selected"
            if fallback_used
            else None
        ),
        "note": (
            "Retrieval threshold is independent of pruning threshold 0.95. "
            "Fallback results did NOT pass the threshold."
            if fallback_used
            else "Retrieval threshold is independent of pruning threshold 0.95"
        ),
    }
    return filtered, info
