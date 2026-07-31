#!/usr/bin/env python3
"""Run all gates and report STUDY_EXECUTION_READY."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.utils.io import load_yaml
from src.utils.checks import check_gate, check_file_exists

def main():
    gates_path = os.path.join(ROOT, "configs", "gates.yaml")
    gates_cfg = load_yaml(gates_path)
    gates = gates_cfg.get("gates", {})
    failed = []
    for gid, gcfg in gates.items():
        ok, msg = check_gate(gid, gcfg, ROOT)
        if not ok:
            failed.append((gid, msg))
        print(msg)
    print("")
    if failed:
        print("STUDY_EXECUTION_READY = FALSE")
        print("Failed gates:", [f[0] for f in failed])
        for gid, msg in failed:
            print(f"  {gid}: {msg}")
    else:
        print("STUDY_EXECUTION_READY = TRUE")

if __name__ == "__main__":
    main()
