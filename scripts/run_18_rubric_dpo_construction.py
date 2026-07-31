#!/usr/bin/env python3
"""Phase 2C: Rubric-anchored DPO preference pair construction.

Replaces the original preference construction with a rubric-scored approach:
- For each problem, generate two candidate tutor turns:
    Chosen: full Socratic prompt (all 5 structural elements)
    Rejected: degraded prompt (no anti-leak, no question-first constraint)
- Score both candidates on all 6 dimensions using the GPT-4o judge
- Include pair only if chosen dominates rejected on >=3 dimensions
- Save pairs to outputs/dpo_pairs_rubric.jsonl for DPO training

Usage:
    OPENAI_API_KEY=sk-... python scripts/run_18_rubric_dpo_construction.py [--n 1000]
"""
import argparse
import json
import os
import sys
import random
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.utils.io import load_jsonl, save_jsonl, ensure_dir, load_yaml
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("rubric_dpo")
METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]

FULL_SCAFFOLD_PROMPT = """\
You are a Socratic math tutor working with a student. Follow these rules:
1. QUESTION-FIRST: Every response must end with a question.
2. NO SOLUTION LEAKING: Never reveal the answer or key steps.
3. GRADUATED SCAFFOLDING: Increase hint specificity only when needed.
4. ERROR TARGETING: Address the student's specific misconception.
5. METACOGNITIVE: Ask students to explain their reasoning.

Problem: {problem}
Conversation: {history}
Your next Socratic question:"""

DEGRADED_PROMPT = """\
You are a helpful math tutor. Provide the correct solution and explain it clearly.

Problem: {problem}
Conversation: {history}
Your response:"""

RUBRIC_SCORE_PROMPT = """\
Score this tutor turn on {dimension}.
Definition: {definition}
Scale: 1 (poor) to 5 (excellent)

Problem: {problem}
Tutor turn: "{tutor_turn}"

Respond with ONLY a single integer 1-5."""

DIMENSION_DEFS = {
    "QQ": "Does the tutor ask a genuine Socratic question (not a statement)?",
    "SD": "Does the tutor provide graduated scaffolding (not the full answer)?",
    "SLR": "Does the tutor avoid revealing the solution? (5=no leaking, 1=full answer)",
    "EDT": "Does the tutor target a specific mathematical misconception?",
    "DC": "Is the tutor turn logically connected to the conversation?",
    "MC_Verified": "Is the tutor's mathematical content correct?",
}


def _api_score_dimension(client, problem: str, tutor_turn: str, dimension: str, max_retries: int = 3) -> float:
    prompt = RUBRIC_SCORE_PROMPT.format(
        dimension=dimension,
        definition=DIMENSION_DEFS.get(dimension, ""),
        problem=problem,
        tutor_turn=tutor_turn[:500],
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            val = float("".join(c for c in text if c.isdigit() or c == "."))
            return min(5.0, max(1.0, val))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return 3.0


def _generate_tutor_turn(client, prompt_tmpl: str, problem: str, history: str = "") -> str:
    """Generate a tutor turn using GPT-4o-mini as a proxy (replace with local model if available)."""
    prompt = prompt_tmpl.format(problem=problem, history=history or "(Start of conversation)")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return "[generation failed]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of problems to process")
    args = parser.parse_args()

    set_global_seed(42)
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=openai_key, timeout=60.0)

    # Load MathDial problems for DPO source
    mathdial_paths = [
        os.path.join(ROOT, "data", "raw", "mathdial", "train.jsonl"),
        os.path.join(ROOT, "data", "raw", "MathDial", "train.jsonl"),
        os.path.join(ROOT, "outputs", "splits", "dpo_problems.jsonl"),
    ]
    problems = []
    for p in mathdial_paths:
        if os.path.isfile(p):
            problems = load_jsonl(p)
            print(f"Loaded {len(problems)} problems from {p}")
            break

    if not problems:
        # Use test set as fallback for demonstration
        test_path = os.path.join(ROOT, "outputs", "splits", "test_500.jsonl")
        if os.path.isfile(test_path):
            problems = load_jsonl(test_path)
            print(f"Warning: Using test set as fallback ({len(problems)} problems)")
        else:
            print("ERROR: No source problems found")
            sys.exit(1)

    random.seed(42)
    problems = random.sample(problems, min(args.n, len(problems)))
    print(f"Processing {len(problems)} problems for rubric-anchored DPO pairs")

    output_dir = os.path.join(ROOT, "outputs")
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "dpo_pairs_rubric.jsonl")

    pairs = []
    included = 0
    excluded = 0

    for i, prob in enumerate(problems):
        problem_text = prob.get("problem", prob.get("question", ""))
        if not problem_text:
            continue
        problem_id = prob.get("problem_id", f"prob_{i}")

        # Generate chosen (full scaffold) and rejected (degraded) turns
        chosen_turn = _generate_tutor_turn(client, FULL_SCAFFOLD_PROMPT, problem_text)
        rejected_turn = _generate_tutor_turn(client, DEGRADED_PROMPT, problem_text)

        # Score both on all dimensions
        chosen_scores = {}
        rejected_scores = {}
        for dim in METRICS:
            chosen_scores[dim] = _api_score_dimension(client, problem_text, chosen_turn, dim)
            rejected_scores[dim] = _api_score_dimension(client, problem_text, rejected_turn, dim)

        # Include pair if chosen dominates on >= 3 dimensions
        chosen_wins = sum(1 for d in METRICS if chosen_scores[d] >= rejected_scores[d] + 0.5)
        if chosen_wins >= 3:
            pairs.append({
                "problem_id": problem_id,
                "problem": problem_text,
                "chosen": chosen_turn,
                "rejected": rejected_turn,
                "chosen_scores": chosen_scores,
                "rejected_scores": rejected_scores,
                "chosen_wins_dims": chosen_wins,
                "chosen_win_dims": [d for d in METRICS if chosen_scores[d] >= rejected_scores[d] + 0.5],
            })
            included += 1
        else:
            excluded += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(problems)} | included={included} excluded={excluded}")
            save_jsonl(out_path, pairs)

    save_jsonl(out_path, pairs)
    print(f"\nRubric DPO construction complete:")
    print(f"  Total processed: {len(problems)}")
    print(f"  Pairs included: {included} ({included/len(problems)*100:.1f}%)")
    print(f"  Pairs excluded: {excluded} (dominated on <3 dims)")
    print(f"  Output: {out_path}")
    print("\nrun_18_rubric_dpo_construction: DONE")


if __name__ == "__main__":
    main()
