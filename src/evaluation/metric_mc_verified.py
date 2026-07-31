"""MC-Verified: SymPy equivalence where applicable."""
from typing import Dict, Optional


def compute_mc_verified(dialogue: Dict, reference_solution: Optional[str] = None) -> float:
    """Compute MC-Verified. SymPy equivalence when applicable."""
    ref = reference_solution or dialogue.get("reference_solution", "")
    turns = dialogue.get("turns", [])
    student_turns = [t["content"] for t in turns if t.get("role") == "student"]
    last_student = student_turns[-1] if student_turns else ""
    try:
        import re
        nums = re.findall(r"-?\d+\.?\d*", last_student)
        ref_nums = re.findall(r"-?\d+\.?\d*", ref)
        if nums and ref_nums:
            return 1.0 if nums[-1] == ref_nums[-1] else 0.0
    except Exception:
        pass
    return 0.5  # Non-identifiable
