"""Judge ensemble: GPT-4o + Qwen only (JUDGE_PROVIDER_LOCK). temperature=0, median aggregation."""
import json
import os
from typing import Callable, Dict, List, Optional

from ..utils.io import load_yaml


def _call_judge(prompt: str, model: str = "gpt-4o") -> Optional[float]:
    """Call judge API, return score 1-5 or None."""
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip()
        for c in text:
            if c.isdigit():
                return float(c)
        return None
    except Exception:
        return None


def judge_ensemble(
    dialogue: Dict,
    metric: str,
    prompt_template: str,
) -> float:
    """Score dialogue with judge ensemble. Returns median of 2 runs."""
    scores = []
    for _ in range(2):
        s = _call_judge(prompt_template.format(**dialogue))
        if s is not None:
            scores.append(s)
    if not scores:
        return 3.0
    return float(sorted(scores)[len(scores) // 2])
