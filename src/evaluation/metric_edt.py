"""EDT: Error Diagnosis Targeting.

Measures how accurately the tutor identifies and responds to the specific
misconceptions or errors the student exhibits, rather than giving generic hints.

Scale 1-5:
  1 = no error targeting (tutor ignores student errors / gives generic responses)
  3 = partial targeting (tutor acknowledges error but response is not specifically targeted)
  5 = precise error targeting (tutor identifies the exact misconception and crafts
      a response that directly addresses it with a targeted hint or question)

Detection uses two-tier approach:
  1. Rule-based: identify student error turns, then check if the following tutor
     turn contains targeted follow-up language.
  2. LLM judge (via judge_fn or QWEN_API_KEY) for richer evaluation.
"""
import os
import re
from typing import Dict, List, Optional, Tuple


# ─── Student error signals ────────────────────────────────────────────────────

# Patterns indicating student expressed confusion or made an error
_STUDENT_ERROR_SIGNALS = [
    r"\bi (don't|do not|dont) (know|understand|get it)\b",
    r"\bi('m| am) (confused|lost|stuck|not sure)\b",
    r"\bi think(?: the answer)? is\s+\d",    # student giving a (potentially wrong) answer
    r"\bwait,?\s+(that's|is that) (wrong|right|correct)\b",
    r"\b(why|how) (does|do|is)\b.{0,50}\?",
    r"\b(not sure|unsure)\b",
    r"\bi made a mistake\b",
    r"\bcan you (explain|help|clarify)\b",
]

# Patterns indicating tutor targeted a specific error
_TUTOR_TARGETED_PATTERNS = [
    r"\byou (said|wrote|got)\b",          # referring to student's specific answer
    r"\bwhere (did|does) .{0,30} go wrong\b",
    r"\blet's look at .{0,30} step\b",
    r"\bthat's (because|since)\b",         # explaining the specific mistake
    r"\bthe issue is\b",
    r"\byou confused\b",
    r"\byou might have\b",
    r"\byour (answer|approach|calculation|step)\b",
    r"\bspecifically\b",
    r"\bin (this|your) case\b",
    r"\bthat (error|mistake|misconception)\b",
    r"\bactually\b.{0,50}(try|let's|check|look)",
    r"\bcareful with\b",
    r"\bwatch out for\b",
]

# Generic (non-targeted) tutor responses
_GENERIC_TUTOR_PATTERNS = [
    r"^let('s| us) (try|start|begin|think)\b",
    r"^what is the (first|next) step\?$",
    r"^(good|great|excellent|nice)!\s*\w",
    r"^(yes|no|correct|incorrect|right|wrong)\.?\s*$",
    r"^(keep going|try again|almost)\b",
]


def _has(text: str, patterns: List[str]) -> bool:
    tl = text.lower().strip()
    return any(re.search(p, tl) for p in patterns)


def _student_has_error(student_text: str) -> bool:
    return _has(student_text, _STUDENT_ERROR_SIGNALS)


def _tutor_is_targeted(tutor_text: str) -> bool:
    if _has(tutor_text, _GENERIC_TUTOR_PATTERNS) and not _has(
        tutor_text, _TUTOR_TARGETED_PATTERNS
    ):
        return False
    return _has(tutor_text, _TUTOR_TARGETED_PATTERNS)


def _extract_error_response_pairs(dialogue: Dict) -> List[Tuple[str, str]]:
    """Return list of (student_error_turn, following_tutor_turn) pairs."""
    turns = dialogue.get("turns", [])
    pairs = []
    for i, turn in enumerate(turns):
        if turn.get("role") == "student" and _student_has_error(turn.get("content", "")):
            # Find the next tutor turn
            for j in range(i + 1, len(turns)):
                if turns[j].get("role") == "tutor":
                    pairs.append((turn["content"], turns[j]["content"]))
                    break
    return pairs


def _rule_edt(dialogue: Dict) -> float:
    """Rule-based EDT: 1-5 score based on error-response pair analysis."""
    pairs = _extract_error_response_pairs(dialogue)
    if not pairs:
        # No detectable student errors: assume adequate baseline
        return 3.0

    targeted_count = sum(1 for _, tutor in pairs if _tutor_is_targeted(tutor))
    ratio = targeted_count / len(pairs)

    if ratio >= 0.75:
        return 5.0
    elif ratio >= 0.5:
        return 4.0
    elif ratio >= 0.25:
        return 3.0
    elif ratio > 0.0:
        return 2.0
    else:
        return 1.0


def _qwen_judge_edt(dialogue: Dict) -> float:
    """Use Qwen to rate Error Diagnosis Targeting. Returns 1-5 float."""
    qwen_key = os.environ.get("QWEN_API_KEY", "")
    if not qwen_key:
        return _rule_edt(dialogue)

    pairs = _extract_error_response_pairs(dialogue)
    if not pairs:
        return 3.0

    # Build compact representation of error-response pairs
    pairs_text = "\n".join(
        f"Student error: \"{s[:200]}\"\nTutor response: \"{t[:200]}\""
        for i, (s, t) in enumerate(pairs[:5])  # cap at 5 pairs
    )

    prompt = (
        "You are evaluating Error Diagnosis Targeting in a math tutoring dialogue.\n"
        f"Problem: {dialogue.get('problem', '')[:200]}\n\n"
        "Below are pairs of (student error / confusion, tutor response):\n"
        f"{pairs_text}\n\n"
        "Rate the tutor's ability to identify and specifically address the student's "
        "exact misconception on a 1-5 scale:\n"
        "1 = tutor completely ignores student errors or gives generic unrelated responses\n"
        "2 = tutor minimally acknowledges error but response is not targeted\n"
        "3 = tutor partially addresses the error with some specific language\n"
        "4 = tutor clearly identifies the misconception and gives a targeted hint\n"
        "5 = tutor precisely diagnoses the error and crafts a perfectly targeted response\n"
        "Respond with ONLY a single integer from 1 to 5."
    )

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
        matches = re.findall(r"\b([1-5])\b", text)
        val = float(matches[-1]) if matches else 3.0
        return min(5.0, max(1.0, val))
    except Exception:
        return _rule_edt(dialogue)


def compute_edt(dialogue: Dict, judge_fn=None, use_llm_judge: bool = False) -> float:
    """Compute EDT for a dialogue. Returns 1-5 float.

    Priority:
      1. Caller-supplied judge_fn (called per error-response pair)
      2. Qwen API only if use_llm_judge=True AND QWEN_API_KEY is set
      3. Rule-based heuristic (default)
    """
    turns = dialogue.get("turns", [])
    if not any(t.get("role") == "tutor" for t in turns):
        return 1.0

    if judge_fn is not None:
        pairs = _extract_error_response_pairs(dialogue)
        if not pairs:
            return 3.0
        scores = []
        for s_turn, t_turn in pairs:
            result = judge_fn(t_turn, dialogue)
            scores.append(float(result) if result is not None else 3.0)
        return float(sum(scores) / len(scores))

    if use_llm_judge and os.environ.get("QWEN_API_KEY", ""):
        return _qwen_judge_edt(dialogue)

    return _rule_edt(dialogue)
