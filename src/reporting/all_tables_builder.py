"""Build paper/ALL_TABLES.md with sections and interpretation."""
import os
from typing import Dict, List

from ..utils.io import ensure_dir


def build_all_tables_md(project_root: str) -> str:
    """Build ALL_TABLES.md content."""
    tables_dir = os.path.join(project_root, "outputs", "tables")
    paper_dir = os.path.join(project_root, "paper")
    ensure_dir(paper_dir)
    sections = [
        ("1) Dataset + Isolation", ["table_dataset_inventory", "table_schema_mapping", "table_data_isolation_audit"]),
        ("2) Training + Generation Coverage", ["table_dialogue_generation_coverage"]),
        ("3) Metric Validity + Reliability", ["table_metric_raw_scores", "table_metric_aggregated_scores", "table_anchor_baseline_results", "table_negative_control_results", "table_rq4_metric_reliability"]),
        ("4) RQ1–RQ5 Result Tables", ["table_rq1_confirmatory_contrasts", "table_rq1_secondary_contrasts", "table_rq2_cost_effectiveness", "table_rq2_break_even_analysis", "table_rq3_turn_degradation_mixed_effects", "table_rq3_failure_mode_taxonomy", "table_rq5_simulator_alignment"]),
        ("5) Robustness/Defensibility Registries", ["table_metric_reporting_charter", "table_metric_coverage_matrix", "table_not_reported_with_reasons", "table_multiplicity_and_ci_audit"]),
        ("6) Boundaries / Non-claims", ["table_support_adequacy_flags"]),
    ]
    lines = [
        "# ALL_TABLES.md",
        "",
        "Auto-generated table index for Socratic AI Tutor Study.",
        "",
    ]
    for title, tables in sections:
        lines.append(f"## {title}")
        lines.append("")
        for t in tables:
            p = os.path.join(tables_dir, t + ".csv")
            exists = "✓" if os.path.isfile(p) else "○"
            lines.append(f"- {exists} `{t}.csv`")
        lines.append("")
    out_path = os.path.join(paper_dir, "ALL_TABLES.md")
    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path
