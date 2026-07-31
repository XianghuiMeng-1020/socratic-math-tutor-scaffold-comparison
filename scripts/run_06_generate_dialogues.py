#!/usr/bin/env python3
"""Generate dialogues for all 4 conditions using the corrected generation pipeline.

The revised pipeline:
- Loads base Llama 3.1-8B once per condition (persistent across problems)
- C1: base model + socratic prompt (no adapter)
- C2: SFT LoRA adapter merged onto base
- C3: DPO LoRA adapter merged onto base
- C4: GPT-4o via API with exponential-backoff retry
- Student simulator: same base Llama 3.1-8B (shared across all conditions)
- Tracks failure counts (early termination, protocol failure) per condition

Usage:
    python scripts/run_06_generate_dialogues.py [--dry-run] [--resume] [--sample=N]
    python scripts/run_06_generate_dialogues.py --conditions=C1_PE_Llama,C4_PE_GPT4o

Environment:
    OPENAI_API_KEY    Required for C4_PE_GPT4o (and API fallback if Llama unavailable)
    HF_TOKEN          Required to load gated Llama 3.1-8B weights
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.generation.run_all_conditions import run_all_conditions
from src.utils.io import ensure_dir


def main():
    dry_run = "--dry-run" in sys.argv
    resume = "--resume" in sys.argv
    sample_size = 0
    conditions = None

    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])
        elif arg.startswith("--conditions="):
            conditions = arg.split("=")[1].split(",")

    cfg_path = os.path.join(ROOT, "configs", "models.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(ROOT, "configs", "paths.yaml")

    ensure_dir(os.path.join(ROOT, "outputs", "dialogues"))
    ensure_dir(os.path.join(ROOT, "outputs", "tables"))

    print(f"Starting dialogue generation:")
    print(f"  dry_run={dry_run}, resume={resume}, sample_size={sample_size}")
    if conditions:
        print(f"  conditions={conditions}")
    if dry_run:
        print("  [DRY RUN: 3 problems per condition]")

    total, failure_stats = run_all_conditions(
        ROOT, cfg_path,
        dry_run=dry_run,
        sample_size=sample_size,
        resume=resume,
        conditions=conditions,
    )

    print(f"\nrun_06_generate_dialogues: {total} dialogues total")
    print("\nFailure statistics:")
    for cond, stats in failure_stats.items():
        if isinstance(stats, dict) and not stats.get("skipped"):
            raw = stats.get("raw_attempted", 0)
            scorable = stats.get("scorable", 0)
            early = stats.get("early_terminations", 0)
            print(f"  {cond}: {scorable}/{raw} scorable "
                  f"({early} early-term, {stats.get('protocol_failures',0)} proto-fail)")


if __name__ == "__main__":
    main()
