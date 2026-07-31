#!/usr/bin/env python3
"""Phase 4A & 4B: Power analysis and formal scaffold-ceiling test.

Power analysis:
- Retrospective: given observed effect sizes and N, what was our detection power?
- Prospective: how many dialogues needed to detect a 2pp composite difference?

Scaffold-ceiling formal test:
- Variance decomposition: what fraction of composite variance is explained by
  training method vs scaffold strength (from ablation data)?
- Formal hypothesis: scaffold strength explains > X% of variance.

Usage:
    python scripts/run_16_power_analysis.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import load_yaml, ensure_dir


def retrospective_power(effect_d: float, n: int, alpha: float = 0.05) -> float:
    """Estimate power for Welch t-test given Cohen's d and sample size."""
    try:
        from statsmodels.stats.power import TTestIndPower
        analysis = TTestIndPower()
        return float(analysis.power(effect_size=abs(effect_d), nobs1=n, alpha=alpha, ratio=1.0))
    except ImportError:
        # Manual approximation using scipy
        ncp = abs(effect_d) * np.sqrt(n / 2)
        df = 2 * n - 2
        crit = stats.t.ppf(1 - alpha / 2, df)
        power = 1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
        return float(max(0.0, min(1.0, power)))


def prospective_power(min_detectable_d: float = 0.3, alpha: float = 0.05, target_power: float = 0.80) -> int:
    """Find minimum N per condition to achieve target power."""
    try:
        from statsmodels.stats.power import TTestIndPower
        analysis = TTestIndPower()
        n = analysis.solve_power(effect_size=min_detectable_d, alpha=alpha, power=target_power, ratio=1.0)
        return int(np.ceil(n))
    except ImportError:
        # Binary search approximation
        for n in range(5, 10000, 5):
            if retrospective_power(min_detectable_d, n, alpha) >= target_power:
                return n
        return 10000


def variance_decomposition(raw_df: pd.DataFrame, ablation_df: pd.DataFrame) -> pd.DataFrame:
    """Decompose composite variance into scaffold-strength vs training-method components."""
    rows = []

    # Total variance (all conditions including ablation)
    all_composites = raw_df["composite"].values
    total_var = float(np.var(all_composites))

    # Between-condition variance
    cond_means = raw_df.groupby("condition")["composite"].mean()
    between_cond_var = float(np.var(cond_means.values)) if len(cond_means) > 1 else 0.0

    # Scaffold strength variance (from ablation: C0a, C0b, C1)
    scaffold_var = 0.0
    if ablation_df is not None and not ablation_df.empty and "composite" in ablation_df.columns:
        scaffold_means = ablation_df.groupby("condition")["composite"].mean()
        scaffold_var = float(np.var(scaffold_means.values)) if len(scaffold_means) > 1 else 0.0

    # Training method variance (C1 vs C2 vs C3; same scaffold, different training)
    training_df = raw_df[raw_df["condition"].isin(["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama"])]
    training_means = training_df.groupby("condition")["composite"].mean()
    training_var = float(np.var(training_means.values)) if len(training_means) > 1 else 0.0

    rows = [
        {"source": "Total (between conditions)", "variance": round(total_var, 5),
         "pct_of_total": 100.0},
        {"source": "Scaffold strength (NoScaffold→FullScaffold)", "variance": round(scaffold_var, 5),
         "pct_of_total": round(scaffold_var / max(total_var, 1e-9) * 100, 1)},
        {"source": "Training method (PE→SFT→DPO, same scaffold)", "variance": round(training_var, 5),
         "pct_of_total": round(training_var / max(total_var, 1e-9) * 100, 1)},
        {"source": "Between-condition (all 4)", "variance": round(between_cond_var, 5),
         "pct_of_total": round(between_cond_var / max(total_var, 1e-9) * 100, 1)},
    ]
    return pd.DataFrame(rows)


def main():
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    ensure_dir(tables_dir)

    # Load aggregated scores
    raw_path = os.path.join(tables_dir, "table_metric_raw_scores.csv")
    if not os.path.isfile(raw_path):
        print("ERROR: table_metric_raw_scores.csv not found. Run run_07 first.")
        sys.exit(1)
    raw_df = pd.read_csv(raw_path)
    scorable_df = raw_df[raw_df.get("scorable", pd.Series([True] * len(raw_df)))] if "scorable" in raw_df.columns else raw_df

    # Load ablation data if available
    ablation_path = os.path.join(tables_dir, "table_scaffold_ablation_raw.csv")
    ablation_df = pd.read_csv(ablation_path) if os.path.isfile(ablation_path) else pd.DataFrame()

    # --- Retrospective power analysis ---
    retro_rows = []
    CONDITIONS = ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_GPT4o"]
    CONTRASTS = [
        ("C1_PE_Llama", "C4_PE_GPT4o"),
        ("C1_PE_Llama", "C2_SFT_Llama"),
        ("C2_SFT_Llama", "C3_DPO_Llama"),
    ]
    for c1, c2 in CONTRASTS:
        a = scorable_df[scorable_df["condition"] == c1]["composite"].values
        b = scorable_df[scorable_df["condition"] == c2]["composite"].values
        if len(a) < 2 or len(b) < 2:
            continue
        sp = np.sqrt((np.var(a) + np.var(b)) / 2)
        d = float((np.mean(a) - np.mean(b)) / sp) if sp > 1e-9 else 0.0
        n = min(len(a), len(b))
        power = retrospective_power(d, n)
        retro_rows.append({
            "contrast": f"{c1} vs {c2}",
            "n_per_condition": n,
            "observed_d": round(d, 3),
            "retrospective_power": round(power, 3),
            "adequately_powered (>0.8)": power >= 0.80,
        })

    retro_df = pd.DataFrame(retro_rows)
    retro_df.to_csv(os.path.join(tables_dir, "table_power_retrospective.csv"), index=False)
    print("Retrospective power analysis:")
    print(retro_df.to_string(index=False))

    # --- Prospective power analysis ---
    prosp_rows = []
    for min_d in [0.2, 0.3, 0.5, 0.8]:
        n_needed = prospective_power(min_detectable_d=min_d)
        prosp_rows.append({
            "min_detectable_d": min_d,
            "n_per_condition_needed": n_needed,
            "total_dialogues_needed": n_needed * 4,  # 4 conditions
        })
    prosp_df = pd.DataFrame(prosp_rows)
    prosp_df.to_csv(os.path.join(tables_dir, "table_power_prospective.csv"), index=False)
    print("\nProspective power (target 80%, alpha=0.05):")
    print(prosp_df.to_string(index=False))

    # --- Scaffold-ceiling variance decomposition ---
    var_df = variance_decomposition(scorable_df, ablation_df)
    var_df.to_csv(os.path.join(tables_dir, "table_scaffold_ceiling_variance.csv"), index=False)
    print("\nScaffold-ceiling variance decomposition:")
    print(var_df.to_string(index=False))

    print("\nrun_16_power_analysis: DONE")


if __name__ == "__main__":
    main()
