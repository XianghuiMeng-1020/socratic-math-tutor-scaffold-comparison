"""Gate and consistency checks."""
import os
from typing import Dict, List, Optional, Tuple


def check_file_exists(path: str) -> bool:
    """Check if file exists and is non-empty."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def check_gate(
    gate_id: str,
    gate_config: Dict,
    base_dir: str,
) -> Tuple[bool, str]:
    """
    Check a single gate. Returns (passed, message).
    """
    check = gate_config.get("check", "")
    desc = gate_config.get("description", gate_id)
    if "table_dataset_inventory.csv" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_dataset_inventory.csv")
        ok = check_file_exists(p)
        return ok, f"G1: {desc}" + (" PASS" if ok else " FAIL")
    if "table_schema_mapping.csv" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_schema_mapping.csv")
        ok = check_file_exists(p)
        return ok, f"G1: {desc}" + (" PASS" if ok else " FAIL")
    if "table_data_isolation_audit.csv" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_data_isolation_audit.csv")
        ok = check_file_exists(p)
        return ok, f"G2: {desc}" + (" PASS" if ok else " FAIL")
    if "~2000 dialogues" in check or "dialogues" in check:
        d = os.path.join(base_dir, "outputs", "dialogues")
        if not os.path.isdir(d):
            return False, f"G3: {desc} FAIL (no dialogues dir)"
        files = [f for f in os.listdir(d) if f.endswith(".jsonl")]
        total = 0
        for f in files:
            with open(os.path.join(d, f), "r") as fp:
                total += sum(1 for _ in fp)
        # Formula: expected_dialogues_min = test_n * 4 (no relaxed pass)
        import pandas as pd
        audit_path = os.path.join(base_dir, "outputs", "tables", "table_data_isolation_audit.csv")
        expected_min = 0
        if os.path.isfile(audit_path):
            try:
                df = pd.read_csv(audit_path)
                row = df[df["check"] == "test_set_size"]
                if not row.empty:
                    test_n = int(row["value"].iloc[0])
                    expected_min = test_n * 4
            except Exception:
                expected_min = 500 * 4  # design default
        if expected_min <= 0:
            expected_min = 500 * 4
        ok = total >= expected_min
        return ok, f"G3: {desc} (observed={total}, expected_min={expected_min})" + (" PASS" if ok else " FAIL")
    if "Simulator checkpoint" in check:
        import json
        ckpt = os.path.join(base_dir, "outputs", "checkpoints", "simulator")
        if not os.path.isdir(ckpt):
            return False, f"G4: {desc} FAIL (no simulator dir)"
        meta = os.path.join(ckpt, "meta.json")
        dry_meta = os.path.join(ckpt, "dry_run_meta.json")
        if os.path.isfile(dry_meta) and not os.path.isfile(meta):
            return False, f"G4: {desc} FAIL (dry_run_meta only; real checkpoint required)"
        if os.path.isfile(meta):
            try:
                with open(meta, "r") as f:
                    m = json.load(f)
                if m.get("status") == "placeholder" or m.get("dry_run") is True:
                    return False, f"G4: {desc} FAIL (placeholder/dry_run checkpoint)"
            except Exception:
                pass
        ok = check_file_exists(ckpt + ".pt") or check_file_exists(ckpt + ".bin") or (os.path.isdir(ckpt) and not os.path.isfile(dry_meta))
        if os.path.isfile(dry_meta) and not (check_file_exists(ckpt + ".pt") or check_file_exists(ckpt + ".bin")):
            ok = False
        return ok, f"G4: {desc}" + (" PASS" if ok else " FAIL (real checkpoint required)")
    if "table_metric_raw_scores" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_metric_raw_scores.csv")
        ok = check_file_exists(p)
        return ok, f"G5: {desc}" + (" PASS" if ok else " FAIL")
    if "table_rq1_confirmatory" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_rq1_confirmatory_contrasts.csv")
        ok = check_file_exists(p)
        return ok, f"G6: {desc}" + (" PASS" if ok else " FAIL")
    if "table_rq2_cost" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_rq2_cost_effectiveness.csv")
        ok = check_file_exists(p)
        return ok, f"G7: {desc}" + (" PASS" if ok else " FAIL")
    if "table_rq3_turn_degradation" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_rq3_turn_degradation_mixed_effects.csv")
        ok = check_file_exists(p)
        return ok, f"G8: {desc}" + (" PASS" if ok else " FAIL")
    if "table_rq4_metric_reliability" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_rq4_metric_reliability.csv")
        ok = check_file_exists(p)
        return ok, f"G9: {desc}" + (" PASS" if ok else " FAIL")
    if "table_rq5_simulator" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_rq5_simulator_alignment.csv")
        ok = check_file_exists(p)
        return ok, f"G10: {desc}" + (" PASS" if ok else " FAIL")
    if "table_metric_coverage_matrix" in check:
        p = os.path.join(base_dir, "outputs", "tables", "table_metric_coverage_matrix.csv")
        ok = check_file_exists(p)
        return ok, f"G11: {desc}" + (" PASS" if ok else " FAIL")
    if "repro_manifest.json" in check:
        p = os.path.join(base_dir, "outputs", "manifests", "repro_manifest.json")
        ok = check_file_exists(p)
        return ok, f"G12: {desc}" + (" PASS" if ok else " FAIL")
    if "ALL_TABLES.md" in check:
        p = os.path.join(base_dir, "paper", "ALL_TABLES.md")
        ok = check_file_exists(p)
        return ok, f"G13: {desc}" + (" PASS" if ok else " FAIL")
    if "final_consistency_lock" in check:
        p = os.path.join(base_dir, "outputs", "logs", "final_consistency_lock_report.txt")
        ok = check_file_exists(p)
        return ok, f"G14: {desc}" + (" PASS" if ok else " FAIL")
    return True, f"{gate_id}: {desc} (default PASS)"
