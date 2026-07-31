"""MC-Judged: geometry/conceptual items by judge ensemble."""
from typing import Dict


def compute_mc_judged(dialogue: Dict, judge_fn=None) -> float:
    """Compute MC-Judged. Enters composite only if inter-model r>=0.7."""
    if judge_fn:
        return judge_fn(dialogue) or 0.5
    return 0.5
