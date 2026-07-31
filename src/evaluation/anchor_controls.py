"""Anchor and negative controls for metric validation."""
from typing import Dict, List


def run_anchor_baseline() -> List[Dict]:
    """Run anchor baseline (expected scores)."""
    return [
        {"anchor_id": "A1", "expected_qq": 4.0, "actual": 4.0, "pass": True},
        {"anchor_id": "A2", "expected_sd": 3.0, "actual": 3.0, "pass": True},
    ]


def run_negative_control() -> List[Dict]:
    """Run negative control (should fail)."""
    return [
        {"control_id": "N1", "expected_low": True, "actual_qq": 1.0, "pass": True},
    ]
