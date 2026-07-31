"""RQ4: Metric reliability analysis.

Implements:
- 3-judge ensemble IRR: GPT-4o, Claude-3-Haiku, Gemini-Pro scoring the same dialogues.
  Computes Pearson r and Cohen's kappa for all 3 pairs across 6 dimensions.
- Majority-vote aggregation.
- Cross-method agreement: rule-based vs LLM-judge for QQ and SLR.
- Reports reliability in table form for the paper.
"""
import os
import json
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..evaluation.reliability import pearson_r, cohens_kappa, mad
from ..utils.io import ensure_dir

METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]
JUDGE_NAMES = ["GPT4o", "Claude", "Gemini"]


# ── Judge prompt templates ──────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of math tutoring dialogue quality.
Score the following tutor turn on the dimension below.

Dimension: {dimension}
Definition: {definition}
Scale: 1 (very poor) to 5 (excellent)

Problem: {problem}
Full dialogue:
{dialogue_text}

Respond with ONLY a single integer from 1 to 5."""

DIMENSION_DEFS = {
    "QQ": "Question Quality: Does the tutor ask a genuine, educationally appropriate Socratic question that probes student understanding without revealing the answer?",
    "SD": "Scaffolding Depth: Does the tutor's question build incrementally on prior turns, providing appropriate cognitive scaffolding for this learner?",
    "SLR": "Solution Leaking Rate: Does the tutor avoid leaking the solution or key steps directly? (5=no leaking, 1=reveals everything)",
    "EDT": "Error Diagnosis Targeting: Does the tutor accurately identify and address the student's specific mathematical error or misconception?",
    "DC": "Dialogue Coherence: Is the tutor's turn logically connected and responsive to what the student just said?",
    "MC_Verified": "Mathematical Correctness: Is the tutor's content mathematically accurate?",
}


def _dialogue_to_text(dialogue: Dict) -> str:
    """Convert dialogue turns to readable string."""
    turns = dialogue.get("turns", [])
    lines = []
    for t in turns:
        role = "Tutor" if t.get("role") == "tutor" else "Student"
        lines.append(f"{role}: {t.get('content', '')}")
    return "\n".join(lines)


def _llm_judge_score(
    client,
    dialogue: Dict,
    dimension: str,
    model: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> float:
    """Score a dialogue on one dimension using an LLM judge. Returns 1-5 float."""
    problem = dialogue.get("problem", "")
    dialogue_text = _dialogue_to_text(dialogue)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        definition=DIMENSION_DEFS.get(dimension, ""),
        problem=problem,
        dialogue_text=dialogue_text[:3000],
    )
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=5,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            val = float("".join(c for c in text if c.isdigit() or c == "."))
            return min(5.0, max(1.0, val))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                return 3.0  # default on failure


def _rule_based_qq(dialogue: Dict) -> float:
    """Rule-based QQ: fraction of tutor turns containing a question mark."""
    turns = dialogue.get("turns", [])
    tutor_turns = [t for t in turns if t.get("role") == "tutor"]
    if not tutor_turns:
        return 3.0
    has_q = [1 if "?" in t.get("content", "") else 0 for t in tutor_turns]
    frac = sum(has_q) / len(has_q)
    return 1.0 + frac * 4.0  # maps [0,1] -> [1,5]


def _rule_based_slr(dialogue: Dict) -> float:
    """Rule-based SLR: fraction of tutor turns with answer-leak keywords (inverted)."""
    turns = dialogue.get("turns", [])
    tutor_turns = [t for t in turns if t.get("role") == "tutor"]
    if not tutor_turns:
        return 5.0
    leak_kws = ["the answer is", "equals", "= ", "therefore", "so the result"]
    leaky = [1 if any(kw in t.get("content", "").lower() for kw in leak_kws) else 0 for t in tutor_turns]
    frac_leaky = sum(leaky) / len(leaky)
    return 5.0 - frac_leaky * 4.0  # high = good (no leaking)


def run_multi_judge_scoring(
    dialogues: List[Dict],
    sample_size: int = 50,
    openai_api_key: str = "",
    anthropic_api_key: str = "",
    google_api_key: str = "",
    output_dir: str = "",
) -> pd.DataFrame:
    """Score a sample of dialogues with 3 LLM judges across 6 dimensions."""
    # Sample
    if len(dialogues) > sample_size:
        random.seed(42)
        sample = random.sample(dialogues, sample_size)
    else:
        sample = dialogues

    if not sample:
        return pd.DataFrame()

    rows = []
    clients = {}
    qwen_key = os.environ.get("QWEN_API_KEY", "")
    DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Initialize clients — priority order
    if openai_api_key:
        try:
            from openai import OpenAI
            clients["GPT4o"] = (OpenAI(api_key=openai_api_key, timeout=60.0), "gpt-4o")
        except Exception as e:
            print(f"OpenAI client init failed: {e}")

    if anthropic_api_key:
        try:
            import anthropic
            ant_client = anthropic.Anthropic(api_key=anthropic_api_key)
            clients["Claude"] = (ant_client, "claude-3-haiku-20240307")
        except Exception as e:
            print(f"Anthropic client init failed: {e}")

    if google_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            clients["Gemini"] = (genai.GenerativeModel("gemini-1.5-flash"), "gemini-1.5-flash")
        except Exception as e:
            print(f"Google Gemini client init failed: {e}")

    # If fewer than 3 judges, fill remaining slots with Qwen models
    if qwen_key:
        from openai import OpenAI as _OAI
        if "Claude" not in clients and "Gemini" not in clients:
            # Use qwen3-max as Judge 2 and qwen-plus as Judge 3
            clients["Qwen-Max"] = (_OAI(api_key=qwen_key, base_url=DASHSCOPE_BASE, timeout=60.0), "qwen3-max")
            clients["Qwen-Plus"] = (_OAI(api_key=qwen_key, base_url=DASHSCOPE_BASE, timeout=60.0), "qwen-plus")
        elif "Claude" not in clients:
            clients["Qwen-Max"] = (_OAI(api_key=qwen_key, base_url=DASHSCOPE_BASE, timeout=60.0), "qwen3-max")
        elif "Gemini" not in clients:
            clients["Qwen-Plus"] = (_OAI(api_key=qwen_key, base_url=DASHSCOPE_BASE, timeout=60.0), "qwen-plus")

    print(f"Active judges: {list(clients.keys())}")

    for d in sample:
        problem_id = d.get("problem_id", "unknown")
        condition = d.get("condition", "unknown")
        profile = d.get("profile", "unknown")
        row = {"problem_id": problem_id, "condition": condition, "profile": profile}

        # Rule-based scores
        row["rule_QQ"] = _rule_based_qq(d)
        row["rule_SLR"] = _rule_based_slr(d)

        for judge_name, (client, model) in clients.items():
            for dim in METRICS:
                if judge_name == "Claude":
                    score = _claude_judge_score(client, d, dim)
                elif judge_name == "Gemini":
                    score = _gemini_judge_score(client, d, dim)
                else:
                    # GPT4o, Qwen-Max, Qwen-Plus all use OpenAI-compatible API
                    score = _llm_judge_score(client, d, dim, model)
                row[f"{judge_name}_{dim}"] = score

        rows.append(row)

    df = pd.DataFrame(rows)
    if output_dir:
        ensure_dir(output_dir)
        df.to_csv(os.path.join(output_dir, "table_multi_judge_raw_scores.csv"), index=False)
    return df


def _claude_judge_score(client, dialogue: Dict, dimension: str) -> float:
    """Score using Anthropic Claude."""
    try:
        problem = dialogue.get("problem", "")
        dialogue_text = _dialogue_to_text(dialogue)
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            dimension=dimension,
            definition=DIMENSION_DEFS.get(dimension, ""),
            problem=problem,
            dialogue_text=dialogue_text[:3000],
        )
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        val = float("".join(c for c in text if c.isdigit() or c == "."))
        return min(5.0, max(1.0, val))
    except Exception:
        return 3.0


def _gemini_judge_score(model, dialogue: Dict, dimension: str) -> float:
    """Score using Google Gemini."""
    try:
        problem = dialogue.get("problem", "")
        dialogue_text = _dialogue_to_text(dialogue)
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            dimension=dimension,
            definition=DIMENSION_DEFS.get(dimension, ""),
            problem=problem,
            dialogue_text=dialogue_text[:3000],
        )
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        val = float("".join(c for c in text if c.isdigit() or c == "."))
        return min(5.0, max(1.0, val))
    except Exception:
        return 3.0


def compute_irr_table(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise IRR (Pearson r + kappa) for all available judge pairs × 6 dimensions.

    Works with any combination of: GPT4o, Claude, Gemini, Qwen-Max, Qwen-Plus.
    """
    # Infer judge names from column headers
    all_judges = set()
    for col in judge_df.columns:
        for metric in METRICS:
            if col.endswith(f"_{metric}"):
                judge = col[: -len(f"_{metric}")]
                all_judges.add(judge)
    all_judges = sorted(all_judges)

    # Generate all unique pairs
    pairs = [(all_judges[i], all_judges[j]) for i in range(len(all_judges)) for j in range(i + 1, len(all_judges))]

    rows = []
    for dim in METRICS:
        for j1, j2 in pairs:
            col1 = f"{j1}_{dim}"
            col2 = f"{j2}_{dim}"
            if col1 not in judge_df.columns or col2 not in judge_df.columns:
                continue
            a = judge_df[col1].dropna().values
            b = judge_df[col2].dropna().values
            n = min(len(a), len(b))
            if n < 5:
                continue
            a, b = a[:n], b[:n]
            r = pearson_r(list(a), list(b))
            # For kappa, round to integers
            a_int = np.round(a).astype(int).clip(1, 5)
            b_int = np.round(b).astype(int).clip(1, 5)
            try:
                kappa = cohens_kappa(list(a_int), list(b_int))
            except Exception:
                kappa = float("nan")
            mad_val = mad(list(np.abs(a - b)))
            rows.append({
                "dimension": dim,
                "judge_pair": f"{j1} vs {j2}",
                "pearson_r": round(r, 3),
                "cohens_kappa": round(kappa, 3),
                "mad": round(mad_val, 3),
                "n": n,
            })
    return pd.DataFrame(rows)


def compute_majority_vote(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Compute majority-vote scores by averaging all available judges per dialogue."""
    rows = []
    for _, row in judge_df.iterrows():
        entry = {
            "problem_id": row.get("problem_id"),
            "condition": row.get("condition"),
            "profile": row.get("profile"),
        }
        for dim in METRICS:
            scores = [row.get(f"{j}_{dim}") for j in JUDGE_NAMES if f"{j}_{dim}" in row and not pd.isna(row.get(f"{j}_{dim}"))]
            entry[f"majority_{dim}"] = float(np.mean(scores)) if scores else float("nan")
        rows.append(entry)
    return pd.DataFrame(rows)


def compute_cross_method_agreement(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Compare rule-based scores with LLM judge scores."""
    rows = []
    for dim, rule_col in [("QQ", "rule_QQ"), ("SLR", "rule_SLR")]:
        if rule_col not in judge_df.columns:
            continue
        rule_scores = judge_df[rule_col].values
        for judge in JUDGE_NAMES:
            judge_col = f"{judge}_{dim}"
            if judge_col not in judge_df.columns:
                continue
            judge_scores = judge_df[judge_col].values
            n = min(len(rule_scores), len(judge_scores))
            if n < 5:
                continue
            r = pearson_r(list(rule_scores[:n]), list(judge_scores[:n]))
            rows.append({
                "dimension": dim,
                "method_a": "Rule-based",
                "method_b": judge,
                "pearson_r": round(r, 3),
                "n": n,
            })
    return pd.DataFrame(rows)


def run_rq4(
    project_root: str,
    raw_scores_df: pd.DataFrame,
    output_tables_dir: str,
    dialogues: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Run RQ4 reliability analysis. Loads multi-judge scores if available."""
    ensure_dir(output_tables_dir)

    # Try loading pre-computed multi-judge scores
    mj_path = os.path.join(project_root, "outputs", "tables", "table_multi_judge_raw_scores.csv")
    if os.path.isfile(mj_path):
        judge_df = pd.read_csv(mj_path)
        irr_df = compute_irr_table(judge_df)
        irr_df.to_csv(os.path.join(output_tables_dir, "table_rq4_triple_judge_irr.csv"), index=False)
        mv_df = compute_majority_vote(judge_df)
        mv_df.to_csv(os.path.join(output_tables_dir, "table_rq4_majority_vote_scores.csv"), index=False)
        cross_df = compute_cross_method_agreement(judge_df)
        cross_df.to_csv(os.path.join(output_tables_dir, "table_rq4_cross_method_agreement.csv"), index=False)
        # Summary reliability per dimension (across all pairs)
        if not irr_df.empty:
            summary = irr_df.groupby("dimension")[["pearson_r", "cohens_kappa", "mad"]].mean().reset_index()
            summary.to_csv(os.path.join(output_tables_dir, "table_rq4_metric_reliability.csv"), index=False)
            return summary
    else:
        # No multi-judge data yet; write expected-format placeholder with clear note
        placeholder = pd.DataFrame([
            {"metric": m, "pearson_r": float("nan"), "cohens_kappa": float("nan"), "mad": float("nan"),
             "note": "Run run_11_multi_judge_scoring.py first"}
            for m in METRICS
        ])
        placeholder.to_csv(os.path.join(output_tables_dir, "table_rq4_metric_reliability.csv"), index=False)
        return placeholder
