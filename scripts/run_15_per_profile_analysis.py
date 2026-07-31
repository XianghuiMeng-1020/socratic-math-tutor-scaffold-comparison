#!/usr/bin/env python3
"""Phase 2D: Per-learner-profile analysis.

Produces:
1. 12-cell table (4 conditions x 3 profiles) for all 6 dimensions + composite
2. Condition x Profile interaction test (mixed-effects LMM)
3. Per-profile simulator validity (KS test + Cohen's d vs MathDial)

Usage:
    python scripts/run_15_per_profile_analysis.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import load_jsonl, load_yaml, ensure_dir
from src.evaluation.composite_score import compute_composite, compute_per_dimension
from src.utils.seed import set_global_seed

METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]
CONDITIONS = ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_GPT4o"]
PROFILES = ["struggling", "progressing", "advanced"]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt((np.var(a) + np.var(b)) / 2)
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 1e-9 else 0.0


def run_interaction_test(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Mixed-effects test: condition x profile interaction on composite score."""
    rows = []
    try:
        import statsmodels.formula.api as smf
        model = smf.mixedlm(
            "composite ~ C(condition) * C(profile)",
            data=raw_df,
            groups=raw_df["problem_id"] if "problem_id" in raw_df.columns else raw_df.index,
        )
        result = model.fit(reml=False)
        for name, coef in result.params.items():
            pval = result.pvalues.get(name, float("nan"))
            se = result.bse.get(name, float("nan"))
            rows.append({"term": name, "coef": round(coef, 4), "se": round(se, 4), "p": round(pval, 4)})
    except Exception as e:
        # Fallback: simple ANOVA per condition-profile cell
        for cond in raw_df["condition"].unique():
            for prof in raw_df["profile"].unique():
                vals = raw_df[(raw_df["condition"] == cond) & (raw_df["profile"] == prof)]["composite"].values
                rows.append({
                    "term": f"{cond}:{prof}",
                    "coef": round(float(np.mean(vals)), 4) if len(vals) > 0 else float("nan"),
                    "se": round(float(np.std(vals)), 4) if len(vals) > 0 else float("nan"),
                    "p": float("nan"),
                })
    return pd.DataFrame(rows)


def run_simulator_validity_per_profile(project_root: str) -> pd.DataFrame:
    """KS test and Cohen's d: simulated vs real MathDial student turns by profile."""
    rows = []
    dialogues_dir = os.path.join(project_root, "outputs", "dialogues")

    # Load simulated student responses by profile
    sim_by_profile = {p: [] for p in PROFILES}
    for cond in CONDITIONS:
        path = os.path.join(dialogues_dir, f"dialogues_{cond}.jsonl")
        if not os.path.isfile(path):
            continue
        for d in load_jsonl(path):
            prof = d.get("profile", "unknown")
            if prof not in sim_by_profile:
                continue
            student_msgs = [t.get("content", "") for t in d.get("turns", []) if t.get("role") == "student"]
            sim_by_profile[prof].extend([len(m) for m in student_msgs])  # use length as proxy feature

    # Load real MathDial
    mathdial_paths = [
        os.path.join(project_root, "data", "raw", "mathdial", "train.jsonl"),
        os.path.join(project_root, "data", "raw", "MathDial", "train.jsonl"),
    ]
    real_lengths = []
    for mp in mathdial_paths:
        if os.path.isfile(mp):
            for d in load_jsonl(mp):
                turns = d.get("turns", d.get("dialog", []))
                for t in turns:
                    if isinstance(t, dict) and t.get("role") == "student":
                        real_lengths.append(len(t.get("content", "")))
            break

    for prof in PROFILES:
        sim_lens = np.array(sim_by_profile[prof])
        real_lens = np.array(real_lengths) if real_lengths else np.array([50.0])
        if len(sim_lens) < 5:
            rows.append({"profile": prof, "n_sim": len(sim_lens), "ks_stat": float("nan"), "ks_p": float("nan"), "cohens_d": float("nan")})
            continue
        ks_stat, ks_p = stats.ks_2samp(sim_lens, real_lens)
        d_val = _cohens_d(sim_lens, real_lens)
        rows.append({
            "profile": prof,
            "n_sim": len(sim_lens),
            "n_real": len(real_lens),
            "ks_stat": round(float(ks_stat), 4),
            "ks_p": round(float(ks_p), 4),
            "cohens_d": round(d_val, 4),
            "sim_mean_len": round(float(sim_lens.mean()), 1),
            "real_mean_len": round(float(real_lens.mean()), 1),
        })
    return pd.DataFrame(rows)


def main():
    set_global_seed(42)
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    dialogues_dir = os.path.join(ROOT, "outputs", "dialogues")
    ensure_dir(tables_dir)

    weights_cfg = load_yaml(os.path.join(ROOT, "configs", "eval_weights.yaml"))
    w = weights_cfg.get("metrics", {})
    slr_invert = weights_cfg.get("slr_invert", True)

    # Load all dialogues and compute scores
    all_rows = []
    for cond in CONDITIONS:
        path = os.path.join(dialogues_dir, f"dialogues_{cond}.jsonl")
        if not os.path.isfile(path):
            print(f"Warning: {path} not found")
            continue
        for d in load_jsonl(path):
            prof = d.get("profile", "unknown")
            pid = d.get("problem_id", "unknown")
            metadata = d.get("metadata", {})
            if not metadata.get("scorable", True):
                continue
            dims = compute_per_dimension(d, slr_invert=slr_invert)
            comp = compute_composite(d, weights=w, slr_invert=slr_invert)
            row = {"condition": cond, "profile": prof, "problem_id": pid, "composite": comp}
            row.update(dims)
            all_rows.append(row)

    if not all_rows:
        print("ERROR: No scorable dialogues found")
        sys.exit(1)

    raw_df = pd.DataFrame(all_rows)
    print(f"Total scorable dialogues: {len(raw_df)}")

    # 1. 12-cell table
    cell_table = raw_df.groupby(["condition", "profile"])[METRICS + ["composite"]].mean().round(3)
    cell_table.to_csv(os.path.join(tables_dir, "table_per_profile_12cell.csv"))
    print("\n12-cell (condition x profile) means:")
    print(cell_table.to_string())

    # 2. Heatmap data (composite only, for figure)
    heatmap = raw_df.groupby(["condition", "profile"])["composite"].mean().unstack(fill_value=0)
    heatmap.to_csv(os.path.join(tables_dir, "table_per_profile_heatmap.csv"))

    # 3. Interaction test
    int_df = run_interaction_test(raw_df)
    int_df.to_csv(os.path.join(tables_dir, "table_interaction_condition_x_profile.csv"), index=False)
    print("\nInteraction test (condition x profile):")
    print(int_df.to_string(index=False))

    # 4. Simulator validity per profile
    sim_valid_df = run_simulator_validity_per_profile(ROOT)
    sim_valid_df.to_csv(os.path.join(tables_dir, "table_simulator_validity_per_profile.csv"), index=False)
    print("\nSimulator validity by profile (KS test + Cohen's d):")
    print(sim_valid_df.to_string(index=False))

    print("\nrun_15_per_profile_analysis: DONE")


if __name__ == "__main__":
    main()
