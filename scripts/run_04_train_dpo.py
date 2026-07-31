#!/usr/bin/env python3
"""Train DPO (C3) from SFT checkpoint."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.training.dpo_train import train_dpo

def main():
    dry = "--dry-run" in sys.argv
    cfg = os.path.join(ROOT, "configs", "paths.yaml")
    train_dpo(ROOT, cfg, dry_run=dry)
    print("run_04_train_dpo: DONE")

if __name__ == "__main__":
    main()
