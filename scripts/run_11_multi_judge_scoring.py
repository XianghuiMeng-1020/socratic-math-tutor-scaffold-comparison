#!/usr/bin/env python3
"""Phase 3A: 3-Judge ensemble scoring (GPT-4o + Claude + Gemini).

Scores a stratified sample of dialogues (50/condition x 4 conditions = 200 total)
across all 6 dimensions with 3 LLM judges. Computes:
- IRR: Pearson r and Cohen's kappa for all 3 pairs x 6 dimensions (18 cells)
- Majority-vote aggregation
- Cross-method agreement (rule-based vs LLM-judge for QQ and SLR)

Usage:
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... python scripts/run_11_multi_judge_scoring.py
"""
import os
import sys
import json
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import pandas as pd

from src.analysis.rq4_metric_reliability import (
    run_multi_judge_scoring,
    compute_irr_table,
    compute_majority_vote,
    compute_cross_method_agreement,
)
from src.utils.io import load_jsonl, ensure_dir


def stratified_sample(dialogues_dir: str, n_per_condition: int = 50) -> list:
    """Sample n_per_condition dialogues per condition, stratified by profile."""
    CONDITIONS = ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_Qwen"]
    PROFILES = ["struggling", "progressing", "advanced"]
    sample = []
    random.seed(42)

    for cond in CONDITIONS:
        path = os.path.join(dialogues_dir, f"dialogues_{cond}.jsonl")
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, skipping {cond}")
            continue
        all_d = load_jsonl(path)
        # Stratify by profile
        by_profile = {p: [d for d in all_d if d.get("profile") == p] for p in PROFILES}
        per_profile = max(1, n_per_condition // len(PROFILES))
        for p in PROFILES:
            pool = by_profile.get(p, [])
            random.shuffle(pool)
            sample.extend(pool[:per_profile])
    return sample


def main():
    dialogues_dir = os.path.join(ROOT, "outputs", "dialogues")
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    ensure_dir(tables_dir)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    if not any([openai_key, anthropic_key, google_key]):
        print("ERROR: At least one of OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY must be set")
        sys.exit(1)

    print("Sampling dialogues (stratified by condition x profile)...")
    sample = stratified_sample(dialogues_dir, n_per_condition=50)
    print(f"Total sample: {len(sample)} dialogues")

    if not sample:
        print("ERROR: No dialogues found. Run run_06_generate_dialogues.py first.")
        sys.exit(1)

    print("Running multi-judge scoring (this may take 10-30 minutes)...")
    judge_df = run_multi_judge_scoring(
        sample,
        sample_size=len(sample),
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        google_api_key=google_key,
        output_dir=tables_dir,
    )

    if judge_df.empty:
        print("ERROR: Multi-judge scoring returned empty results")
        sys.exit(1)

    print(f"Scoring complete. {len(judge_df)} dialogues scored.")

    # IRR table
    irr_df = compute_irr_table(judge_df)
    irr_df.to_csv(os.path.join(tables_dir, "table_rq4_triple_judge_irr.csv"), index=False)
    print("\nIRR Table (3-judge pairs x 6 dimensions):")
    if not irr_df.empty:
        print(irr_df.to_string(index=False))

    # Majority vote
    mv_df = compute_majority_vote(judge_df)
    mv_df.to_csv(os.path.join(tables_dir, "table_rq4_majority_vote_scores.csv"), index=False)

    # Cross-method agreement
    cross_df = compute_cross_method_agreement(judge_df)
    cross_df.to_csv(os.path.join(tables_dir, "table_rq4_cross_method_agreement.csv"), index=False)
    if not cross_df.empty:
        print("\nCross-method agreement (rule-based vs LLM judge):")
        print(cross_df.to_string(index=False))

    # Summary reliability
    if not irr_df.empty:
        summary = irr_df.groupby("dimension")[["pearson_r", "cohens_kappa", "mad"]].mean().reset_index()
        summary.to_csv(os.path.join(tables_dir, "table_rq4_metric_reliability.csv"), index=False)
        print("\nMean IRR per dimension:")
        print(summary.to_string(index=False))

    print("\nrun_11_multi_judge_scoring: DONE")


if __name__ == "__main__":
    main()

