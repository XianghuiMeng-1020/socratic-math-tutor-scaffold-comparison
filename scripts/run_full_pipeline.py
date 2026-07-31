#!/usr/bin/env python3
"""Run full pipeline: 00 -> 10 in sequence."""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

SCRIPTS = [
    "run_00_data_audit.py",
    "run_01_split_and_isolation.py",
    "run_02_train_sft.py",
    "run_03_build_dpo_pairs.py",
    "run_04_train_dpo.py",
    "run_05_train_simulator.py",
    "run_06_generate_dialogues.py",
    "run_07_evaluate_metrics.py",
    "run_08_run_analyses.py",
    "run_09_build_reports.py",
    "run_10_all_gates.py",
]

def main():
    dry = "--dry-run" in sys.argv
    for name in SCRIPTS:
        path = os.path.join(ROOT, "scripts", name)
        cmd = [sys.executable, path]
        if dry and ("train" in name or "generate" in name):
            cmd.append("--dry-run")
        print(f"Running {name}...")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"FAILED: {name}")
            sys.exit(r.returncode)
    print("run_full_pipeline: DONE")

if __name__ == "__main__":
    main()
