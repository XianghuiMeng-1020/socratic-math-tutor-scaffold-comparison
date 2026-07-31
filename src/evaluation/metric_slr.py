"""SLR: Solution Leaking Rate.

Measures the fraction of TUTOR turns that prematurely disclose key solution steps.
High raw SLR = bad (lots of leaking).  The composite score uses (1 - SLR_norm) so
that higher composite = less leaking = better.

Detection uses a two-tier approach:
  1. Rule-based: keyword / phrase patterns that strongly signal solution disclosure.
  2. Optional LLM judge (via judge_fn or QWEN_API_KEY env var) for ambiguous turns.

Return value: float in [1, 5] on the 1-5 scale used by composite_score.py
  5 = no leaking (all tutor turns are safe)
  1 = heavy leaking (most tutor turns reveal the answer)
"""
import os
import re
import time
from typing import Dict, List, Optional

# ─── Keyword tiers ───────────────────────────────────────────────────────────

# Tier-1: near-certain answer disclosure
_TIER1_PATTERNS = [
    r"\bthe answer is\b",
    r"\bthe solution is\b",
    r"\bx\s*=\s*\d",           # "x = 4", "x=4"
    r"\b=\s*\d+\.?\d*\s*$",   # "equals 8" at end
    r"\btherefore\s+x\b",
    r"\bso\s+x\s*=",
    r"\bresult\s+is\s+\d",
    r"\bfinal answer\b",
    r"\bthe value of x is\b",
    r"\bx equals\b",
]

# Tier-2: strong hints that give the key algebraic step
_TIER2_PATTERNS = [
    r"\bsubtract\s+\d+\s+from both sides\b",
    r"\bdivide both sides by\b",
    r"\bmultiply both sides by\b",
    r"\bfirst\s+step\s+is\s+to\b",
    r"\byou need to\s+(add|subtract|multiply|divide)\b",
    r"\bjust\s+(add|subtract|multiply|divide)\b",
    r"\bplug in\b",
    r"\bsubstitute\b.*\bget\b",
]

# Safe metacognitive / question patterns that indicate NO leaking
_SAFE_PATTERNS = [
    r"\?",                          # any question mark
    r"\bwhat do you (think|know|notice)\b",
    r"\bcan you\b",
    r"\bhow would you\b",
    r"\bwhy\b",
    r"\bwhat if\b",
    r"\btell me\b",
    r"\bwhat is the (first|next) step\b",
]


def _has_pattern(text: str, patterns: List[str]) -> bool:
    tl = text.lower()
    return any(re.search(p, tl) for p in patterns)


def _rule_slr_for_turn(text: str) -> Optional[float]:
    """Return a leak score for a single tutor turn in [0, 1].

    0.0 = definitely not leaking
    1.0 = definitely leaking
    None = ambiguous (needs LLM judge)
    """
    if not text or len(text) < 5:
        return 0.0

    # If turn is a safe question → no leak
    if _has_pattern(text, _SAFE_PATTERNS) and not _has_pattern(text, _TIER1_PATTERNS):
        return 0.0

    # Tier-1 match → definite leak
    if _has_pattern(text, _TIER1_PATTERNS):
        return 1.0

    # Tier-2 match → probable leak
    if _has_pattern(text, _TIER2_PATTERNS):
        return 0.7

    # No signals → treat as borderline (no leak but could use judge)
    return None


def _qwen_judge_slr(dialogue: Dict, tutor_turn: str, reference_solution: str = "") -> float:
    """Call Qwen API to judge whether a tutor turn leaks the solution.

    Returns leak probability in [0, 1].
    """
    qwen_key = os.environ.get("QWEN_API_KEY", "")
    if not qwen_key:
        return 0.2  # conservative default: assume minor leak risk

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=qwen_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30.0,
        )
        prompt = (
            "You are evaluating a math tutor for solution leaking.\n"
            f"Problem context: {dialogue.get('problem', '')[:300]}\n"
            f"Tutor said: \"{tutor_turn[:400]}\"\n"
            "Does this tutor turn directly reveal the answer or a key solution step that "
            "resolves the problem for the student?\n"
            "Respond with ONLY a single integer:\n"
            "1 = definitely leaks (gives answer or critical step)\n"
            "2 = probably leaks (strong hint that resolves the problem)\n"
            "3 = borderline (partial hint, not solution-resolving)\n"
            "4 = probably safe (scaffolded question or vague hint)\n"
            "5 = definitely safe (pure Socratic question, no answer info)"
        )
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        matches = re.findall(r"\b([1-5])\b", text)
        score = float(matches[-1]) if matches else 3.0
        score = min(5.0, max(1.0, score))
        # Convert 1-5 (1=leak, 5=safe) → leak probability [0,1]
        return (5.0 - score) / 4.0
    except Exception:
        return 0.2


def compute_slr(dialogue: Dict, judge_fn=None, use_llm_judge: bool = False) -> float:
    """Compute SLR for a dialogue.

    Returns a score on the 1-5 scale:
      5 = no solution leaking (best)
      1 = heavy solution leaking (worst)

    The composite_score.py normalises this to [0,1] and then inverts it,
    so that a high inverted SLR (= low raw leak fraction) counts positively
    toward the composite.
    """
    turns = dialogue.get("turns", [])
    tutor_turns = [t for t in turns if t.get("role") == "tutor"]
    if not tutor_turns:
        return 5.0  # no tutor turns = no leaking

    reference_solution = dialogue.get("reference_solution", "")
    leak_scores: List[float] = []

    for t in tutor_turns:
        content = t.get("content", "")
        rule_result = _rule_slr_for_turn(content)

        if rule_result is not None:
            leak_scores.append(rule_result)
        elif judge_fn is not None:
            # Caller supplied a judge function
            j = judge_fn(content, dialogue)
            leak_scores.append(float(j) if j is not None else 0.2)
        elif use_llm_judge and os.environ.get("QWEN_API_KEY", ""):
            # Use built-in Qwen judge only when explicitly requested
            leak_scores.append(_qwen_judge_slr(dialogue, content, reference_solution))
        else:
            # Ambiguous turn with no judge: assume low leak (Socratic context)
            leak_scores.append(0.1)

    if not leak_scores:
        return 5.0

    # mean_leak in [0, 1]; convert to 1-5 scale (5 = no leak, 1 = full leak)
    mean_leak = sum(leak_scores) / len(leak_scores)
    slr_1_5 = 5.0 - mean_leak * 4.0   # 0 leak → 5.0, full leak → 1.0
    return float(max(1.0, min(5.0, slr_1_5)))
