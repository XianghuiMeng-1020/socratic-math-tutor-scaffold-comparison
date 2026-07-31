"""QQ: Question Quality - rule-first + judge fallback."""
import re
from typing import Dict, List, Optional


def _rule_based_qq(text: str) -> Optional[float]:
    """Rule-based QQ score 1-5."""
    text_lower = text.lower()
    if len(text) < 10:
        return 1.0
    if any(x in text_lower for x in ["the answer is", "equals", "= ", "therefore the"]):
        return 1.0
    if "?" in text:
        return 4.0
    if any(x in text_lower for x in ["can you", "what if", "how would", "why do you"]):
        return 4.0
    return 3.0


def compute_qq(dialogue: Dict, judge_fn=None) -> float:
    """Compute QQ for dialogue. Uses rule-first, judge fallback."""
    turns = dialogue.get("turns", [])
    tutor_turns = [t["content"] for t in turns if t.get("role") == "tutor"]
    if not tutor_turns:
        return 0.0
    scores = []
    for t in tutor_turns:
        rule = _rule_based_qq(t)
        if rule is not None:
            scores.append(rule)
        elif judge_fn:
            scores.append(judge_fn(t, dialogue) or 3.0)
        else:
            scores.append(3.0)
    return sum(scores) / len(scores) if scores else 0.0
