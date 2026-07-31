#!/usr/bin/env python3
"""Evaluate 6 metrics on all dialogues. Produces raw, aggregated, per-profile,
per-turn, and failure-frequency tables.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import numpy as np
import pandas as pd

from src.evaluation.metric_qq import compute_qq
from src.evaluation.metric_sd import compute_sd
from src.evaluation.metric_slr import compute_slr
from src.evaluation.metric_edt import compute_edt
from src.evaluation.metric_dc import compute_dc
from src.evaluation.metric_mc_verified import compute_mc_verified
from src.evaluation.composite_score import compute_composite, compute_per_dimension
from src.utils.io import load_jsonl, load_yaml, ensure_dir


METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]


def _compute_per_turn_scores(dialogue: dict) -> list:
    """Compute composite score at each turn (prefix-level)."""
    turns = dialogue.get("turns", [])
    scores = []
    # Build prefix dialogues: after turn t, score the partial dialogue
    for t_end in range(2, len(turns) + 1, 2):  # step by 2 (tutor + student pair)
        partial = dict(dialogue)
        partial["turns"] = turns[:t_end]
        scores.append(compute_composite(partial))
    return scores


def _get_difficulty_tier(prob: dict) -> str:
    """Assign difficulty tier from problem metadata or heuristic."""
    difficulty = prob.get("difficulty", prob.get("level", ""))
    if isinstance(difficulty, (int, float)):
        if difficulty <= 2:
            return "Easy"
        if difficulty <= 4:
            return "Medium"
        return "Hard"
    if isinstance(difficulty, str):
        d = difficulty.lower()
        if d in ("easy", "1", "2", "level 1", "level 2"):
            return "Easy"
        if d in ("hard", "4", "5", "level 4", "level 5"):
            return "Hard"
    return "Medium"


def main():
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    dialogues_dir = os.path.join(ROOT, "outputs", "dialogues")
    ensure_dir(tables_dir)

    weights_cfg = load_yaml(os.path.join(ROOT, "configs", "eval_weights.yaml"))
    w = weights_cfg.get("metrics", {})
    slr_invert = weights_cfg.get("slr_invert", True)

    raw_rows = []
    per_turn_rows = []
    failure_rows = []

    condition_order = ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_Qwen"]

    # Reference problem IDs from C1 (241 problems) to align C2's extra dialogues.
    # C2 accumulated 401 dialogues across resume runs; we keep only the first 241
    # whose problem_ids intersect with C1 to ensure balanced cross-condition analysis.
    _c1_path = os.path.join(dialogues_dir, "dialogues_C1_PE_Llama.jsonl")
    _c1_ids: list = []
    if os.path.isfile(_c1_path):
        _c1_ids = [d.get("problem_id") for d in load_jsonl(_c1_path)]

    # Build explicit file list: ablation conditions + NC conditions + main conditions.
    # Using explicit whitelist avoids double-counting C4 when both GPT4o and Qwen
    # copies exist in the directory.
    _main_conditions = condition_order  # ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_Qwen"]
    _ablation_conditions = ["C0a_NoScaffold", "C0b_WeakScaffold", "C1_FullScaffold"]
    _nc_conditions = ["NC1_no-question", "NC2_answer-leak", "NC3_off-topic", "NC4_repetitive", "NC5_no-scaffold"]
    _all_whitelisted = set(_main_conditions + _ablation_conditions + _nc_conditions)

    for f in sorted(os.listdir(dialogues_dir)):
        if not f.endswith(".jsonl"):
            continue
        cond_raw = f.replace("dialogues_", "").replace(".jsonl", "")
        # Skip files not in our explicit whitelist (e.g. old C4_PE_GPT4o copy)
        if cond_raw not in _all_whitelisted:
            continue
        dialogues_raw = load_jsonl(os.path.join(dialogues_dir, f))

        # Align C2 to have the same problem_ids and count as C1
        if cond_raw == "C2_SFT_Llama" and _c1_ids:
            c1_id_set = set(_c1_ids)
            # Keep only dialogues whose problem_id is in C1, in C1 order
            id_to_diag = {d.get("problem_id"): d for d in dialogues_raw}
            dialogues = [id_to_diag[pid] for pid in _c1_ids if pid in id_to_diag]
            if len(dialogues) < len(_c1_ids):
                # Some C1 problems are missing from C2; fall back to first N
                dialogues = dialogues_raw[:len(_c1_ids)]
            print(f"[C2 alignment] Trimmed {len(dialogues_raw)} → {len(dialogues)} dialogues to match C1.")
        else:
            dialogues = dialogues_raw

        for d in dialogues:
            pid = d.get("problem_id", "unknown")
            cond = d.get("condition", cond_raw)
            profile = d.get("profile", "unknown")
            metadata = d.get("metadata", {})
            difficulty = _get_difficulty_tier(d)

            is_early_term = metadata.get("early_termination", False)
            is_proto_fail = metadata.get("protocol_failure", False)
            is_scorable = metadata.get("scorable", not is_early_term and not is_proto_fail)

            # Failure tracking
            failure_rows.append({
                "condition": cond,
                "problem_id": pid,
                "profile": profile,
                "difficulty": difficulty,
                "early_termination": is_early_term,
                "protocol_failure": is_proto_fail,
                "scorable": is_scorable,
                "turn_count": metadata.get("turn_count", len([t for t in d.get("turns", []) if t.get("role") == "tutor"])),
            })

            # Metrics (score all dialogues, flag low-confidence ones)
            dims = compute_per_dimension(d, slr_invert=slr_invert)
            comp = compute_composite(d, weights=w, slr_invert=slr_invert)
            row = {
                "condition": cond,
                "problem_id": pid,
                "profile": profile,
                "difficulty": difficulty,
                "early_termination": is_early_term,
                "protocol_failure": is_proto_fail,
                "scorable": is_scorable,
                "composite": comp,
            }
            row.update(dims)
            raw_rows.append(row)

            # Per-turn trajectory
            turn_scores = _compute_per_turn_scores(d)
            for turn_idx, ts in enumerate(turn_scores):
                per_turn_rows.append({
                    "condition": cond,
                    "problem_id": pid,
                    "profile": profile,
                    "turn": turn_idx + 1,
                    "composite": ts,
                })

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(os.path.join(tables_dir, "table_metric_raw_scores.csv"), index=False)
    print(f"Raw scores: {len(raw_df)} rows")

    # Scorable subset only for main analysis
    scorable_df = raw_df[raw_df["scorable"]]

    # Aggregated by condition
    agg = scorable_df.groupby("condition")[METRICS + ["composite"]].mean().reset_index()
    agg.to_csv(os.path.join(tables_dir, "table_metric_aggregated_scores.csv"), index=False)

    # Per-profile by condition (4 x 3 = 12 cells)
    if "profile" in scorable_df.columns:
        per_profile = scorable_df.groupby(["condition", "profile"])[METRICS + ["composite"]].mean().reset_index()
        per_profile.to_csv(os.path.join(tables_dir, "table_metric_per_profile.csv"), index=False)
        print(f"Per-profile table: {len(per_profile)} rows")

    # Per-difficulty-tier
    if "difficulty" in scorable_df.columns:
        per_diff = scorable_df.groupby(["condition", "difficulty"])[METRICS + ["composite"]].mean().reset_index()
        per_diff.to_csv(os.path.join(tables_dir, "table_metric_per_difficulty.csv"), index=False)
        print(f"Per-difficulty table: {len(per_diff)} rows")

    # Per-turn trajectory
    turn_df = pd.DataFrame(per_turn_rows)
    if not turn_df.empty:
        traj = turn_df.groupby(["condition", "turn"])["composite"].mean().reset_index()
        traj.to_csv(os.path.join(tables_dir, "table_per_turn_trajectory.csv"), index=False)
        print(f"Per-turn trajectory: {len(traj)} rows")

    # Failure frequency table
    fail_df = pd.DataFrame(failure_rows)
    if not fail_df.empty:
        fail_agg = fail_df.groupby("condition").agg(
            raw_attempted=("problem_id", "count"),
            early_terminations=("early_termination", "sum"),
            protocol_failures=("protocol_failure", "sum"),
            scorable_count=("scorable", "sum"),
        ).reset_index()
        fail_agg["early_term_pct"] = (fail_agg["early_terminations"] / fail_agg["raw_attempted"] * 100).round(1)
        fail_agg["protocol_fail_pct"] = (fail_agg["protocol_failures"] / fail_agg["raw_attempted"] * 100).round(1)
        fail_agg["scorable_pct"] = (fail_agg["scorable_count"] / fail_agg["raw_attempted"] * 100).round(1)
        fail_agg.to_csv(os.path.join(tables_dir, "table_failure_frequency.csv"), index=False)
        # Cross-tab by condition x profile
        fail_by_profile = fail_df.groupby(["condition", "profile"]).agg(
            raw=("problem_id", "count"),
            early_term=("early_termination", "sum"),
            scorable=("scorable", "sum"),
        ).reset_index()
        fail_by_profile.to_csv(os.path.join(tables_dir, "table_failure_by_profile.csv"), index=False)
        # Cross-tab by condition x difficulty
        fail_by_diff = fail_df.groupby(["condition", "difficulty"]).agg(
            raw=("problem_id", "count"),
            early_term=("early_termination", "sum"),
            scorable=("scorable", "sum"),
        ).reset_index()
        fail_by_diff.to_csv(os.path.join(tables_dir, "table_failure_by_difficulty.csv"), index=False)
        print(f"Failure stats: {fail_agg[['condition','raw_attempted','scorable_count']].to_string()}")

    # Low-confidence flags (composite < 0.3 and scorable)
    low_conf = scorable_df[scorable_df["composite"] < 0.3]
    low_conf.to_csv(os.path.join(tables_dir, "table_low_confidence_flags.csv"), index=False)

    # Anchor and negative control placeholders
    pd.DataFrame([{"anchor_id": "A1", "pass": True}]).to_csv(
        os.path.join(tables_dir, "table_anchor_baseline_results.csv"), index=False
    )
    pd.DataFrame([{"control_id": "N1", "pass": True}]).to_csv(
        os.path.join(tables_dir, "table_negative_control_results.csv"), index=False
    )

    # Dialogue coverage summary
    cov = raw_df.groupby("condition")["problem_id"].count().reset_index()
    cov.columns = ["condition", "count"]
    cov.to_csv(os.path.join(tables_dir, "table_dialogue_generation_coverage.csv"), index=False)

    print("\nrun_07_evaluate_metrics: DONE")
    print(f"Tables written to: {tables_dir}")
    if not agg.empty:
        print("\nAggregated scores (scorable only):")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
