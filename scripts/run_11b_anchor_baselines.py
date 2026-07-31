#!/usr/bin/env python3
"""Phase 2F: Anchor baseline generation and scoring.

Generates three types of anchor dialogues to contextualise the absolute composite
scores (0.46-0.49) from the main experiment:

  ANCHOR-Expert : Professional Socratic human tutor (high-quality hand-crafted turns)
  ANCHOR-Direct : Answer-giving tutor (gives the solution immediately — worst case)
  ANCHOR-Random : Incoherent/random responses (noise floor)

All dialogues are scored with the same evaluation pipeline as the main conditions.
Outputs: outputs/tables/table_anchor_baselines.csv

Usage:
    QWEN_API_KEY=sk-... python scripts/run_11b_anchor_baselines.py [--n 30]
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import numpy as np
import pandas as pd

from src.utils.io import load_jsonl, ensure_dir, load_yaml
from src.utils.seed import set_global_seed
from src.evaluation.composite_score import compute_composite, compute_per_dimension

METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]

# ─── Expert tutor turn templates ─────────────────────────────────────────────

_EXPERT_TUTOR_OPENINGS = [
    "What do you already know about this type of problem?",
    "Before we dive in, can you tell me what approach comes to mind first?",
    "What information does the problem give us?",
    "What does the problem ask us to find?",
    "Can you think of a similar problem you've solved before?",
]

_EXPERT_TUTOR_FOLLOWUPS = [
    "That's a good start. What would happen if you applied that reasoning to the next step?",
    "Interesting — why do you think that?",
    "What makes you confident in that step?",
    "What if we tried a different approach — what might that look like?",
    "Can you walk me through your reasoning step by step?",
    "What's the next thing we need to figure out?",
    "You're on the right track. What constraint haven't we used yet?",
    "Let's check: does your answer make sense in the context of the problem?",
]

_EXPERT_STUDENT_TURNS = [
    "I think we need to use algebra.",
    "Hmm, maybe I should look at the equation more carefully.",
    "I'm not sure, but I think the answer might be 4.",
    "Oh wait, I made an error — let me redo that.",
    "That makes sense! So x equals 3?",
    "I understand now. Thank you!",
]

_DIRECT_TUTOR_TEMPLATES = [
    "The answer is {answer}.",
    "The solution is x = {answer}. You just need to divide both sides by the coefficient.",
    "Here's the solution: first rearrange to get x alone, giving x = {answer}.",
    "The final answer is {answer}. Therefore x = {answer}.",
]

_RANDOM_TUTOR_TURNS = [
    "Elephants are the largest land mammals.",
    "The sky is blue because of Rayleigh scattering.",
    "Python is a programming language.",
    "Have you tried turning it off and on again?",
    "Mathematics is fundamental to the universe.",
    "Let's talk about something else.",
    "I enjoy cooking pasta on weekends.",
]

_STUDENT_ACKNOWLEDGEMENT = [
    "I see.",
    "Okay.",
    "I don't understand.",
    "Can you explain more?",
    "Thank you.",
    "That's confusing.",
]


def _make_expert_dialogue(prob: dict, n_turns: int = 8) -> dict:
    """Create a high-quality expert-tutor anchor dialogue."""
    turns = []
    for i in range(n_turns // 2):
        if i == 0:
            tutor = random.choice(_EXPERT_TUTOR_OPENINGS)
        else:
            tutor = random.choice(_EXPERT_TUTOR_FOLLOWUPS)
        student = random.choice(_EXPERT_STUDENT_TURNS)
        turns.append({"role": "tutor", "content": tutor})
        turns.append({"role": "student", "content": student})
    return {
        "problem_id": prob.get("problem_id", "anchor"),
        "problem": prob.get("problem", ""),
        "condition": "ANCHOR-Expert",
        "profile": "progressing",
        "turns": turns,
        "reference_solution": prob.get("reference_solution", ""),
    }


def _make_direct_dialogue(prob: dict, n_turns: int = 4) -> dict:
    """Create an answer-giving (worst-case SLR) anchor dialogue."""
    answer = prob.get("answer", "4")
    turns = []
    for i in range(n_turns // 2):
        template = random.choice(_DIRECT_TUTOR_TEMPLATES)
        tutor = template.format(answer=answer)
        student = random.choice(_STUDENT_ACKNOWLEDGEMENT)
        turns.append({"role": "tutor", "content": tutor})
        turns.append({"role": "student", "content": student})
    return {
        "problem_id": prob.get("problem_id", "anchor"),
        "problem": prob.get("problem", ""),
        "condition": "ANCHOR-Direct",
        "profile": "progressing",
        "turns": turns,
        "reference_solution": prob.get("reference_solution", ""),
    }


def _make_random_dialogue(prob: dict, n_turns: int = 6) -> dict:
    """Create a random/incoherent anchor dialogue (noise floor)."""
    turns = []
    for i in range(n_turns // 2):
        tutor = random.choice(_RANDOM_TUTOR_TURNS)
        student = random.choice(_STUDENT_ACKNOWLEDGEMENT)
        turns.append({"role": "tutor", "content": tutor})
        turns.append({"role": "student", "content": student})
    return {
        "problem_id": prob.get("problem_id", "anchor"),
        "problem": prob.get("problem", ""),
        "condition": "ANCHOR-Random",
        "profile": "progressing",
        "turns": turns,
        "reference_solution": prob.get("reference_solution", ""),
    }


def _score_dialogue(d: dict, weights: dict, slr_invert: bool) -> dict:
    dims = compute_per_dimension(d, slr_invert=slr_invert)
    comp = compute_composite(d, weights=weights, slr_invert=slr_invert)
    row = {
        "condition": d["condition"],
        "problem_id": d.get("problem_id", ""),
        "composite": comp,
    }
    row.update(dims)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of problems per anchor")
    args = parser.parse_args()

    set_global_seed(42)
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    ensure_dir(tables_dir)

    weights_cfg = load_yaml(os.path.join(ROOT, "configs", "eval_weights.yaml"))
    w = weights_cfg.get("metrics", {})
    slr_invert = weights_cfg.get("slr_invert", True)

    # Load test problems
    test_path = os.path.join(ROOT, "outputs", "splits", "test_500.jsonl")
    if not os.path.isfile(test_path):
        print(f"ERROR: {test_path} not found. Run run_01_split_and_isolation.py first.")
        sys.exit(1)

    problems = load_jsonl(test_path)
    random.seed(42)
    sample = random.sample(problems, min(args.n, len(problems)))
    print(f"Generating anchor dialogues for {len(sample)} problems...")

    rows = []
    for prob in sample:
        for make_fn in (_make_expert_dialogue, _make_direct_dialogue, _make_random_dialogue):
            d = make_fn(prob)
            row = _score_dialogue(d, w, slr_invert)
            rows.append(row)

    df = pd.DataFrame(rows)
    agg = df.groupby("condition")[METRICS + ["composite"]].agg(["mean", "std"]).round(3)
    print("\nAnchor Baseline Results:")
    print(agg.to_string())

    # Summary per anchor condition
    summary_rows = []
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        r = {"condition": cond, "n": len(sub), "composite_mean": round(sub["composite"].mean(), 4),
             "composite_std": round(sub["composite"].std(), 4)}
        for m in METRICS:
            if m in sub.columns:
                r[f"{m}_mean"] = round(sub[m].mean(), 4)
        summary_rows.append(r)

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(tables_dir, "table_anchor_baselines.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved anchor baseline summary → {out_path}")
    print(summary_df.to_string(index=False))

    # Also load main conditions for comparison printout
    main_path = os.path.join(tables_dir, "table_metric_aggregated_scores.csv")
    if os.path.isfile(main_path):
        main_df = pd.read_csv(main_path)
        print("\n--- Contextualisation: Main Conditions vs Anchors ---")
        main_comp = main_df[main_df["condition"].isin(
            ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_Qwen"]
        )][["condition", "composite"]].copy() if "condition" in main_df.columns else pd.DataFrame()
        if not main_comp.empty:
            anchor_comp = summary_df[["condition", "composite_mean"]].rename(
                columns={"composite_mean": "composite"}
            )
            combined = pd.concat([main_comp, anchor_comp], ignore_index=True)
            print(combined.sort_values("composite", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
