"""Composite score from 6 metrics with configurable weights.

Weight defaults match the paper (Section 3.5):
  QQ 20%, SD 20%, SLR 25%, EDT 15%, DC 10%, MC_Verified 10%.

SLR scale: 5 = no solution leaking (best), 1 = heavy leaking (worst).
compute_slr returns 1-5 where higher = better, consistent with all other metrics.
No inversion is needed; slr_invert=False by default.
"""
import os
from typing import Dict, Optional

from ..utils.io import load_yaml
from .metric_qq import compute_qq
from .metric_sd import compute_sd
from .metric_slr import compute_slr
from .metric_edt import compute_edt
from .metric_dc import compute_dc
from .metric_mc_verified import compute_mc_verified
from .metric_mc_judged import compute_mc_judged

# Paper-specified default weights (study lock)
DEFAULT_WEIGHTS = {
    "QQ": 0.20,
    "SD": 0.20,
    "SLR": 0.25,   # applied to inverted SLR
    "EDT": 0.15,
    "DC": 0.10,
    "MC_Verified": 0.10,
}


def _norm_1_5_to_0_1(x: float) -> float:
    """Map 1-5 scale to 0-1."""
    if 1 <= x <= 5:
        return (x - 1) / 4.0
    return max(0.0, min(1.0, x))


def compute_composite(
    dialogue: Dict,
    weights: Optional[Dict[str, float]] = None,
    mc_judged_gate: float = 0.7,
    slr_invert: bool = False,
) -> float:
    """Compute weighted composite OTL score.

    Args:
        dialogue: dialogue dict with 'turns' list.
        weights: metric weights (defaults to DEFAULT_WEIGHTS if None or empty).
        mc_judged_gate: MC-Judged only included if inter-judge agreement >= this.
        slr_invert: if True, SLR is inverted (1 - SLR_norm) before weighting.

    Returns:
        Composite score in [0, 1].
    """
    w = weights if weights else DEFAULT_WEIGHTS
    # Ensure weights sum to 1 (normalize if not)
    w_sum = sum(w.values())
    if abs(w_sum - 1.0) > 1e-6:
        w = {k: v / w_sum for k, v in w.items()}

    qq = _norm_1_5_to_0_1(compute_qq(dialogue))
    sd = _norm_1_5_to_0_1(compute_sd(dialogue))
    slr_raw = _norm_1_5_to_0_1(compute_slr(dialogue))
    slr = (1.0 - slr_raw) if slr_invert else slr_raw
    edt = _norm_1_5_to_0_1(compute_edt(dialogue))
    dc = _norm_1_5_to_0_1(compute_dc(dialogue))
    mc_v = compute_mc_verified(dialogue)

    score = (
        w.get("QQ", 0.20) * qq
        + w.get("SD", 0.20) * sd
        + w.get("SLR", 0.25) * slr
        + w.get("EDT", 0.15) * edt
        + w.get("DC", 0.10) * dc
        + w.get("MC_Verified", 0.10) * mc_v
    )
    return float(min(1.0, max(0.0, score)))


def compute_per_dimension(
    dialogue: Dict,
    slr_invert: bool = False,
) -> Dict[str, float]:
    """Return all 6 normalized metric scores (pre-weighting) for a dialogue."""
    qq = _norm_1_5_to_0_1(compute_qq(dialogue))
    sd = _norm_1_5_to_0_1(compute_sd(dialogue))
    slr_raw = _norm_1_5_to_0_1(compute_slr(dialogue))
    slr = (1.0 - slr_raw) if slr_invert else slr_raw
    edt = _norm_1_5_to_0_1(compute_edt(dialogue))
    dc = _norm_1_5_to_0_1(compute_dc(dialogue))
    mc_v = compute_mc_verified(dialogue)
    return {
        "QQ": round(qq, 4),
        "SD": round(sd, 4),
        "SLR": round(slr, 4),
        "EDT": round(edt, 4),
        "DC": round(dc, 4),
        "MC_Verified": round(mc_v, 4),
    }
