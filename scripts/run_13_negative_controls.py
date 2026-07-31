#!/usr/bin/env python3
"""Phase 3C: Expanded negative controls — 5 deliberate degradation modes.

Each mode generates 30 synthetic "bad" dialogues that deliberately violate
one of the 6 pedagogical dimensions. We then verify that the corresponding
metric detects the violation (significant drop relative to C1 baseline).

Degradation modes:
  NC1: no-question     — tutor never asks a question (all statements); tests QQ
  NC2: answer-leak     — tutor reveals full solution in turn 1; tests SLR
  NC3: off-topic       — tutor discusses unrelated math; tests DC
  NC4: repetitive      — tutor repeats the same question every turn; tests QQ, DC
  NC5: no-scaffold     — tutor provides correct but unsupported hints; tests SD, EDT

Usage:
    python scripts/run_13_negative_controls.py [--n 30]
"""
import argparse
import json
import os
import sys
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import load_jsonl, save_jsonl, load_yaml, ensure_dir
from src.evaluation.composite_score import compute_composite, compute_per_dimension
from src.utils.seed import set_global_seed

METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]
AFFECTED_DIMS = {
    "NC1_no-question": ["QQ"],
    "NC2_answer-leak": ["SLR"],
    "NC3_off-topic": ["DC", "EDT"],
    "NC4_repetitive": ["QQ", "DC"],
    "NC5_no-scaffold": ["SD", "EDT"],
}

SYNTHETIC_PROBLEM = "Solve for x: 2x + 3 = 11"
SYNTHETIC_REFERENCE = "x = 4"
SYNTHETIC_TURNS_BASE = [
    {"role": "tutor", "content": "What do you know about solving linear equations?"},
    {"role": "student", "content": "I think I need to move the 3 to the other side."},
    {"role": "tutor", "content": "That's right. What operation undoes addition?"},
    {"role": "student", "content": "Subtraction."},
    {"role": "tutor", "content": "Exactly! Now apply that to both sides. What do you get?"},
    {"role": "student", "content": "2x = 8. So x = 4!"},
]


def _make_nc1_no_question(n_turns: int = 5) -> list:
    """Tutor never asks a question — all declarative statements."""
    turns = []
    statements = [
        "Linear equations require isolating the variable.",
        "You need to subtract 3 from both sides.",
        "The equation becomes 2x equals 8.",
        "Dividing both sides by 2 gives x equals 4.",
        "The solution is x equals 4.",
    ]
    for i in range(n_turns):
        turns.append({"role": "tutor", "content": statements[i % len(statements)]})
        turns.append({"role": "student", "content": "OK, I see."})
    return turns


def _make_nc2_answer_leak() -> list:
    """Tutor reveals full solution immediately."""
    return [
        {"role": "tutor", "content": "The answer is x = 4. Subtract 3 from both sides to get 2x = 8, then divide by 2."},
        {"role": "student", "content": "Oh, so x = 4. Got it!"},
        {"role": "tutor", "content": "Yes, x = 4 is the answer."},
        {"role": "student", "content": "Thanks!"},
    ]


def _make_nc3_off_topic(n_turns: int = 5) -> list:
    """Tutor discusses unrelated mathematics."""
    off_topic = [
        "Let's talk about the Pythagorean theorem. Do you know a² + b² = c²?",
        "Integration by parts is an important calculus technique.",
        "Prime numbers have only two divisors: 1 and themselves.",
        "The Fibonacci sequence starts 1, 1, 2, 3, 5, 8...",
        "Matrix multiplication is not commutative in general.",
    ]
    turns = []
    for i in range(n_turns):
        turns.append({"role": "tutor", "content": off_topic[i % len(off_topic)]})
        turns.append({"role": "student", "content": "Interesting, but I'm confused about the original problem."})
    return turns


def _make_nc4_repetitive(n_turns: int = 5) -> list:
    """Tutor repeats the exact same question every turn."""
    turns = []
    same_q = "What is the first step in solving this equation?"
    for i in range(n_turns):
        turns.append({"role": "tutor", "content": same_q})
        turns.append({"role": "student", "content": f"I already answered that (turn {i+1})."})
    return turns


def _make_nc5_no_scaffold(n_turns: int = 5) -> list:
    """Tutor provides correct but unsupported, non-scaffolded hints."""
    turns = [
        {"role": "tutor", "content": "Subtract 3 from both sides."},
        {"role": "student", "content": "OK, 2x = 8."},
        {"role": "tutor", "content": "Divide by 2."},
        {"role": "student", "content": "x = 4."},
        {"role": "tutor", "content": "Correct. The answer is x = 4."},
        {"role": "student", "content": "Got it."},
    ]
    return turns


GENERATORS = {
    "NC1_no-question": _make_nc1_no_question,
    "NC2_answer-leak": _make_nc2_answer_leak,
    "NC3_off-topic": _make_nc3_off_topic,
    "NC4_repetitive": _make_nc4_repetitive,
    "NC5_no-scaffold": _make_nc5_no_scaffold,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Dialogues per degradation mode")
    args = parser.parse_args()

    set_global_seed(42)
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    dialogues_dir = os.path.join(ROOT, "outputs", "dialogues")
    ensure_dir(tables_dir)
    ensure_dir(dialogues_dir)

    weights_cfg = load_yaml(os.path.join(ROOT, "configs", "eval_weights.yaml"))
    w = weights_cfg.get("metrics", {})
    slr_invert = weights_cfg.get("slr_invert", True)

    # Load C1 baseline for comparison
    c1_path = os.path.join(dialogues_dir, "dialogues_C1_PE_Llama.jsonl")
    c1_scores = {}
    if os.path.isfile(c1_path):
        c1_diag = load_jsonl(c1_path)
        c1_dim_scores = [compute_per_dimension(d, slr_invert=slr_invert) for d in c1_diag]
        for dim in METRICS:
            c1_scores[dim] = [s[dim] for s in c1_dim_scores if dim in s]
    else:
        print("Warning: C1 baseline not found; comparison will use fixed reference values.")
        c1_scores = {dim: [0.7] * 30 for dim in METRICS}

    all_rows = []
    stat_rows = []

    for nc_name, generator in GENERATORS.items():
        dialogues = []
        dim_scores_list = {dim: [] for dim in METRICS}

        for i in range(args.n):
            pid = f"nc_{nc_name}_{i}"
            turns = generator()
            d = {
                "problem_id": pid,
                "condition": nc_name,
                "profile": "progressing",
                "turns": turns,
                "problem": SYNTHETIC_PROBLEM,
                "reference_solution": SYNTHETIC_REFERENCE,
                "metadata": {"negative_control": True, "control_type": nc_name},
            }
            dialogues.append(d)
            dims = compute_per_dimension(d, slr_invert=slr_invert)
            comp = compute_composite(d, weights=w, slr_invert=slr_invert)
            row = {"condition": nc_name, "problem_id": pid, "composite": comp}
            row.update(dims)
            all_rows.append(row)
            for dim in METRICS:
                dim_scores_list[dim].append(dims.get(dim, 0.0))

        save_jsonl(os.path.join(dialogues_dir, f"dialogues_{nc_name}.jsonl"), dialogues)
        print(f"\n{nc_name}: {args.n} dialogues generated")

        # Test: does each affected dim show significant drop vs C1?
        affected = AFFECTED_DIMS.get(nc_name, [])
        for dim in METRICS:
            nc_vals = np.array(dim_scores_list[dim])
            c1_vals = np.array(c1_scores.get(dim, [0.7] * 10))
            nc_mean = float(nc_vals.mean())
            c1_mean = float(c1_vals.mean()) if len(c1_vals) > 0 else 0.7

            t_stat, p_val = stats.ttest_ind(nc_vals, c1_vals) if len(c1_vals) >= 2 else (float("nan"), 1.0)
            should_drop = dim in affected
            actually_dropped = nc_mean < c1_mean - 0.05
            gate_pass = (not should_drop) or actually_dropped

            stat_rows.append({
                "control": nc_name,
                "dimension": dim,
                "nc_mean": round(nc_mean, 3),
                "c1_baseline_mean": round(c1_mean, 3),
                "drop": round(c1_mean - nc_mean, 3),
                "t_stat": round(float(t_stat), 3) if not np.isnan(t_stat) else float("nan"),
                "p_val": round(float(p_val), 4) if not np.isnan(p_val) else float("nan"),
                "targeted_dimension": should_drop,
                "drop_detected": actually_dropped,
                "gate_pass": gate_pass,
            })
            status = "✓ PASS" if gate_pass else "✗ FAIL"
            if should_drop:
                print(f"  {dim}: NC={nc_mean:.3f} vs C1={c1_mean:.3f}  [{status}]")

    # Save results
    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(os.path.join(tables_dir, "table_negative_control_raw.csv"), index=False)

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(os.path.join(tables_dir, "table_negative_control_results.csv"), index=False)

    # Pivot for paper: 5 modes x 6 dims
    pivot = stat_df.pivot(index="control", columns="dimension", values="drop").round(3)
    pivot.to_csv(os.path.join(tables_dir, "table_negative_control_matrix.csv"))
    print("\nNegative Control Matrix (drop from C1 baseline):")
    print(pivot.to_string())

    # Count passes
    targeted = stat_df[stat_df["targeted_dimension"]]
    pass_rate = targeted["gate_pass"].mean() if not targeted.empty else float("nan")
    print(f"\nGate pass rate (targeted dimensions): {pass_rate:.0%}")
    print("\nrun_13_negative_controls: DONE")


if __name__ == "__main__":
    main()
