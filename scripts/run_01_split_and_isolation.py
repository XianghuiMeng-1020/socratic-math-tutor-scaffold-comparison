#!/usr/bin/env python3
"""Run split and isolation, produce test_500 and audit."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import pandas as pd

from src.data.discover import discover_datasets
from src.data.isolate_splits import isolate_splits

def main():
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    splits_dir = os.path.join(ROOT, "outputs", "splits")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    inv = discover_datasets(ROOT, tables_dir)
    isolate_splits(ROOT, inv, tables_dir, splits_dir)
    print("run_01_split_and_isolation: DONE")

if __name__ == "__main__":
    main()
