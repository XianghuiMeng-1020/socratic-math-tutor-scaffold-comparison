#!/usr/bin/env python3
"""Run data discovery and produce inventory + schema mapping."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.data.discover import discover_datasets
from src.data.schema_map import build_schema_mapping
from src.utils.io import load_yaml

def main():
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    inv = discover_datasets(ROOT, tables_dir)
    build_schema_mapping(inv, tables_dir)
    print("run_00_data_audit: DONE")

if __name__ == "__main__":
    main()
