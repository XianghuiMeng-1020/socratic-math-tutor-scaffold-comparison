#!/usr/bin/env python3
"""Train SFT LoRA (C2)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.training.sft_lora import train_sft_lora

def main():
    dry = "--dry-run" in sys.argv
    cfg = os.path.join(ROOT, "configs", "paths.yaml")
    train_sft_lora(ROOT, cfg, dry_run=dry)
    print("run_02_train_sft: DONE")

if __name__ == "__main__":
    main()
