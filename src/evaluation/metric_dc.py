"""DC: Dialogue Coherence - embedding continuity + judge logic."""
from typing import Dict


def compute_dc(dialogue: Dict, judge_fn=None) -> float:
    """Compute DC."""
    turns = dialogue.get("turns", [])
    if len(turns) < 2:
        return 3.0
    prev = ""
    coh = 0
    for t in turns:
        c = t.get("content", "")
        if prev and c and (prev[-20:] in c or c[:20] in prev or len(c) > 5):
            coh += 1
        prev = c
    return 3.0 + (coh / max(1, len(turns))) if turns else 0.0
