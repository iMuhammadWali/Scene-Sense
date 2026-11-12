"""Sweep grids (E1) and operating points (E1b). A config = (method, params dict)."""
from __future__ import annotations

def cfg_name(method: str, params: dict) -> str:
    return method + ("[" + ",".join(f"{k}={v}" for k, v in params.items()) + "]" if params else "")

TAUS = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98]

def e1_grid() -> dict[str, list[tuple[str, dict]]]:
    """Shard name -> list of (method, params). One shard per method family so runs resume by family."""
    g: dict[str, list] = {}
    g["refs"] = [("full_5fps", {})]
    g["uniform"] = [("uniform", {"fraction": f}) for f in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]]
    g["cos_last_drop"] = [("cos_last_drop", {"tau": t}) for t in TAUS]
    g["cos_consec_drop"] = [("cos_consec_drop", {"tau": t}) for t in TAUS]
    g["streammem_merge"] = [("streammem_merge", {"delta": t}) for t in TAUS]
    # tol 0.1 = value hard-coded in the released MaxInfo notebooks; 1.0 = maxvolpy default (both beyond the planned grid)
    TOLS = [0.1, 0.15, 0.3, 0.45, 1.0]
    g["maxinfo_fast"] = [("maxinfo_fast", {"R": R, "tol": t}) for R in [4, 8, 12, 16] for t in TOLS]
    g["maxinfo_slow"] = [("maxinfo_slow", {"R": R, "tol": t, "cap_frac": c}) for R in [4, 8, 12, 16] for t in TOLS
                         for c in [0.05, 0.10, 0.20]]
    g["maxinfo_chunk"] = [("maxinfo_chunk", {"chunk_s": c, "R": R, "tol": t}) for c in [2, 4, 8] for R in [1, 2, 3]
                          for t in [0.3, 0.6, 1.0]]
    g["hybrid"] = [("hybrid", {"delta": d, "chunk_s": c, "R": R, "tol": t}) for d in [0.90, 0.95] for c in [4, 8]
                   for R in [1, 2] for t in [0.3, 0.6]]
    return g

# E1b: "best-guess" operating points (fixed a priori, not tuned on the test set)
def e1b_methods() -> list[tuple[str, dict]]:
    return [
        ("full_5fps", {}),
        ("cos_last_drop", {"tau": 0.95}),          # current system
        ("cos_last_drop", {"tau": 0.90}),
        ("uniform", {"match": "cos_last_drop[tau=0.95]"}),   # budget-matched to the system default, per stream
        ("uniform", {"match": "cos_last_drop[tau=0.9]"}),
        ("uniform", {"fraction": 0.10}),
        ("streammem_merge", {"delta": 0.95}),
        ("maxinfo_chunk", {"chunk_s": 8, "R": 2, "tol": 0.6}),
        ("maxinfo_slow", {"R": 8, "tol": 0.1, "cap_frac": 0.10}),
        ("hybrid", {"delta": 0.95, "chunk_s": 8, "R": 2, "tol": 0.6}),
    ]

FAMILY_OF = {"full_5fps": "full", "uniform": "uniform", "cos_last_drop": "cos", "cos_consec_drop": "cos",
             "streammem_merge": "streammem", "maxinfo_fast": "maxinfo", "maxinfo_slow": "maxinfo",
             "maxinfo_chunk": "maxinfo", "hybrid": "hybrid", "random_moment": "random"}
