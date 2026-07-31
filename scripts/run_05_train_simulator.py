#!/usr/bin/env python3
"""Build simulator dataset and train simulator."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.data.isolate_splits import load_test_ids
from src.simulation.build_simulator_dataset import build_simulator_dataset
from src.simulation.train_simulator import train_simulator

def main():
    dry = "--dry-run" in sys.argv
    splits_dir = os.path.join(ROOT, "outputs", "splits")
    mathdial_path = os.path.join(ROOT, "data", "mathdial-main", "mathdial-main", "data")
    test_ids = load_test_ids(splits_dir)
    sim_path = os.path.join(splits_dir, "simulator_train.jsonl")
    build_simulator_dataset(ROOT, mathdial_path, test_ids, sim_path)
    train_simulator(ROOT, os.path.join(ROOT, "configs", "paths.yaml"), dry_run=dry)
    print("run_05_train_simulator: DONE")

if __name__ == "__main__":
    main()
