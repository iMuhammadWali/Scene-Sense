"""Cosine-similarity drop methods."""
import numpy as np
from ..selection import from_indices
from .. import config as C

def cos_last_drop(emb, ts, duration, tau=0.95):
    """THE TEAM'S pruning: keep a frame if its cosine vs the last KEPT frame is <= tau. Calls app.pipeline.pruning.prune_frames."""
    C.setup_team()
    from app.pipeline.frame_extraction import CandidateFrame
    from app.pipeline.pruning import prune_frames
    cands = [CandidateFrame(frame_id=i, timestamp=float(ts[i]), source_frame_index=i) for i in range(len(emb))]
    kept, _, _ = prune_frames(cands, emb, float(tau))
    return from_indices(emb, ts, duration, [c.frame_id for c in kept])

def cos_consec_drop(emb, ts, duration, tau=0.95):
    """Variant: compare each frame with the immediately previous frame (kept or not); drop if cosine > tau."""
    sims = np.einsum("ij,ij->i", emb[1:], emb[:-1])
    keep = np.r_[True, sims <= tau]
    return from_indices(emb, ts, duration, np.flatnonzero(keep))
