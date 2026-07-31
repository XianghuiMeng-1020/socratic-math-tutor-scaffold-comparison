"""SD: Scaffolding Depth.

Measures how graduated and learner-adaptive the tutor's hints are across turns.
Scale 1-5:
  1 = no scaffolding (direct instruction, statements only)
  3 = some graduated hints with moderate specificity
  5 = highly adaptive Socratic scaffolding with metacognitive prompts

Detection uses two-tier approach:
  1. Rule-based heuristic: vocabulary and structural cues in tutor turns.
  2. Optional LLM judge (via judge_fn or QWEN_API_KEY) for full rubric scoring.
"""
import os
import re
from typing import Dict, List, Optional


# ─── Pattern sets for rule-based heuristic ───────────────────────────────────

# High-SD markers (metacognitive, open-ended, adaptive)
_HIGH_SD = [
    r"why do you think",
    r"what makes you confident",
    r"can you explain (your|the) reasoning",
    r"what if",
    r"how would you (approach|solve|think about)",
    r"what do you notice",
    r"what patterns do you see",
    r"does that remind you of",
    r"what would happen if",
    r"what strategy could you use",
    r"metacognitive",
    r"let's think about (why|what|how)",
    r"can you walk me through",
]

# Mid-SD markers (graduated hints, present but not deeply adaptive)
_MID_SD = [
    r"\bhint\b",
    r"\bstep\b",
    r"let's (start|begin|try)",
    r"what is the (first|next) step",
    r"can you try",
    r"think about",
    r"consider",
    r"remember that",
    r"recall",
    r"\?",              # any question is at least mid
]

# Low-SD markers (direct instruction, no scaffolding)
_LOW_SD = [
    r"^(you need to|you should|you must|just|simply)\b",
    r"^(subtract|add|multiply|divide|plug in|substitute)\b",
    r"the (answer|solution|result) is",
    r"that's (correct|wrong|right|incorrect)\.",
]


def _rule_sd_for_turn(text: str) -> float:
    """Return a 1-5 SD score for a single tutor turn via rule heuristic."""
    tl = text.strip().lower()
    if not tl:
        return 1.0

    if any(re.search(p, tl) for p in _HIGH_SD):
        return 5.0

    if any(re.search(p, tl) for p in _LOW_SD):
        return 1.0

    if any(re.search(p, tl) for p in _MID_SD):
        return 3.0

    # Default: a non-empty turn with no strong signal = below mid
    return 2.0


def _build_judge_prompt(dialogue: Dict, tutor_turns_text: str) -> str:
    problem = dialogue.get("problem", "")[:300]
    profile = dialogue.get("profile", "unknown")
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "prompts", "judge_sd.txt"
    )
    prompt_path = os.path.normpath(prompt_path)
    if os.path.isfile(prompt_path):
        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()
        return template.replace("{problem}", problem).replace(
            "{profile}", profile
        ).replace("{dialogue}", tutor_turns_text[:2000])

    # Inline fallback
    return (
        f"You are evaluating scaffolding depth in a math tutoring dialogue.\n"
        f"Problem: {problem}\nLearner profile: {profile}\n"
        f"Tutor turns:\n{tutor_turns_text[:1500]}\n\n"
        "Rate the overall Scaffolding Depth on a 1-5 scale:\n"
        "1 = no scaffolding (direct instruction, no questions)\n"
        "2 = minimal scaffolding (few questions, mostly direct)\n"
        "3 = moderate graduated hints, some questions\n"
        "4 = good scaffolding, mostly adaptive questions\n"
        "5 = highly adaptive Socratic scaffolding with metacognitive prompts\n"
        "Respond with ONLY a single integer from 1 to 5."
    )


def _qwen_judge_sd(dialogue: Dict) -> float:
    """Call Qwen API for SD scoring. Returns 1-5 float."""
    qwen_key = os.environ.get("QWEN_API_KEY", "")
    if not qwen_key:
        return 3.0

    turns = dialogue.get("turns", [])
    tutor_turns_text = "\n".join(
        f"T{i+1}: {t['content']}"
        for i, t in enumerate(turns)
        if t.get("role") == "tutor"
    )

    prompt = _build_judge_prompt(dialogue, tutor_turns_text)
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=qwen_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30.0,
        )
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Extract the last standalone integer 1-5 in the response
        matches = re.findall(r"\b([1-5])\b", text)
        val = float(matches[-1]) if matches else 3.0
        return min(5.0, max(1.0, val))
    except Exception:
        return 3.0


def compute_sd(dialogue: Dict, judge_fn=None, use_llm_judge: bool = False) -> float:
    """Compute SD for a dialogue. Returns 1-5 float.

    Priority order:
      1. If judge_fn supplied by caller, use it.
      2. If use_llm_judge=True AND QWEN_API_KEY is set, call Qwen for full rubric.
      3. Rule-based heuristic (default - fast, consistent, no API needed).
    """
    turns = dialogue.get("turns", [])
    tutor_turns = [t for t in turns if t.get("role") == "tutor"]
    if not tutor_turns:
        return 1.0

    # Caller-supplied judge takes priority
    if judge_fn is not None:
        scores = [judge_fn(t["content"], dialogue) or 3.0 for t in tutor_turns]
        return float(sum(scores) / len(scores))

    # Qwen API judge: only when explicitly requested
    if use_llm_judge and os.environ.get("QWEN_API_KEY", ""):
        return _qwen_judge_sd(dialogue)

    # Rule-based default: average per-turn scores
    scores = [_rule_sd_for_turn(t.get("content", "")) for t in tutor_turns]
    return float(sum(scores) / len(scores))
