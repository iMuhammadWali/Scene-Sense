"""Frame-selection methods: pure functions (emb, ts, duration, **params) -> Selection."""
import time
from .uniform import full, uniform
from .cosine import cos_last_drop, cos_consec_drop
from .streammem import streammem_merge
from .maxinfo import maxinfo_fast, maxinfo_slow, maxinfo_chunk
from .hybrid import hybrid

METHODS = {
    "full_5fps": full, "uniform": uniform, "cos_last_drop": cos_last_drop, "cos_consec_drop": cos_consec_drop,
    "streammem_merge": streammem_merge, "maxinfo_fast": maxinfo_fast, "maxinfo_slow": maxinfo_slow,
    "maxinfo_chunk": maxinfo_chunk, "hybrid": hybrid,
}

def select(method: str, emb, ts, duration, **params):
    """Run a method and record its wall time (Selection.select_s)."""
    t0 = time.perf_counter()
    sel = METHODS[method](emb, ts, duration, **params)
    sel.select_s = time.perf_counter() - t0
    return sel
