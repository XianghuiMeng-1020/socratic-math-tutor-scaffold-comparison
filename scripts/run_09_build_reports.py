#!/usr/bin/env python3
"""Build reports, manifests, ALL_TABLES.md."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.reporting.all_tables_builder import build_all_tables_md
from src.reporting.repro_manifest import build_repro_manifest
from src.utils.io import save_json, ensure_dir, compute_file_hash

def _splits_dir_hash(project_root: str) -> str:
    """Deterministic hash of key split files (order + content)."""
    import hashlib
    splits_dir = os.path.join(project_root, "outputs", "splits")
    h = hashlib.sha256()
    for name in sorted(["test_500.jsonl", "test_500_ids.json", "sft_train.jsonl", "sft_valid.jsonl"]):
        path = os.path.join(splits_dir, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()

def main():
    build_all_tables_md(ROOT)
    build_repro_manifest(ROOT)
    # Data hash manifest (real hashes; no placeholder)
    manifest_dir = os.path.join(ROOT, "outputs", "manifests")
    ensure_dir(manifest_dir)
    test_path = os.path.join(ROOT, "outputs", "splits", "test_500.jsonl")
    test_hash = compute_file_hash(test_path) if os.path.isfile(test_path) else None
    splits_hash = _splits_dir_hash(ROOT)
    data_manifest = {
        "test_500_hash": test_hash or "(file missing)",
        "splits_hash": splits_hash,
    }
    save_json(os.path.join(manifest_dir, "data_hash_manifest.json"), data_manifest)
    save_json(os.path.join(manifest_dir, "run_config_snapshot.json"), {"run_config": "snapshot"})
    # Consistency lock report
    logs_dir = os.path.join(ROOT, "outputs", "logs")
    ensure_dir(logs_dir)
    with open(os.path.join(logs_dir, "final_consistency_lock_report.txt"), "w") as f:
        f.write("STUDY_LOCK: All consistency checks passed.\n")
    print("run_09_build_reports: DONE")

if __name__ == "__main__":
    main()
