"""RQ1: Confirmatory 3 contrasts (C1 vs C4), (C1 vs C2), (C2 vs C3) with Holm-Bonferroni."""
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.io import load_yaml, ensure_dir
from .bootstrap_ci import bootstrap_ci


CONFIRMATORY = [
    ("C1_PE_Llama", "C4_PE_Qwen"),
    ("C1_PE_Llama", "C2_SFT_Llama"),
    ("C2_SFT_Llama", "C3_DPO_Llama"),
]
SECONDARY = [
    ("C1_PE_Llama", "C3_DPO_Llama"),
    ("C2_SFT_Llama", "C4_PE_Qwen"),
    ("C3_DPO_Llama", "C4_PE_Qwen"),
]


def _effect_size_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d."""
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a), np.std(b)
    sp = np.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / sp if sp > 0 else 0.0


def _welch_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Welch t-test. Returns (stat, pvalue)."""
    from scipy import stats
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def holm_bonferroni(pvalues: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni adjustment."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = []
    for i, idx in enumerate(order):
        adj = pvalues[idx] * (n - i)
        adjusted.append(adj <= alpha)
    return [adjusted[order.tolist().index(i)] for i in range(n)]


def run_rq1_contrasts(
    project_root: str,
    scores_df: pd.DataFrame,
    output_tables_dir: str,
) -> pd.DataFrame:
    """Run RQ1 contrasts, produce confirmatory and secondary tables."""
    ensure_dir(output_tables_dir)
    confirmatory_rows = []
    for c1, c2 in CONFIRMATORY:
        a = scores_df[scores_df["condition"] == c1]["composite"].values
        b = scores_df[scores_df["condition"] == c2]["composite"].values
        if len(a) < 2 or len(b) < 2:
            continue
        d = _effect_size_d(a, b)
        stat, p = _welch_t(a, b)
        lo, hi = bootstrap_ci(a, b, n_boot=1000)
        abs_gain_abs = abs(np.mean(b) - np.mean(a))  # absolute pp difference on [0,1] scale
        abs_gain_pct = (np.mean(a) - np.mean(b)) / max(np.mean(b), 1e-6) * 100
        # Pre-registered gates: p < 0.05 AND |d| >= 0.2 AND absolute gain >= 0.03 (3 pp)
        decision = p < 0.05 and abs(d) >= 0.2 and abs_gain_abs >= 0.03
        confirmatory_rows.append({
            "contrast": f"{c1} vs {c2}",
            "mean_a": round(np.mean(a), 6),
            "mean_b": round(np.mean(b), 6),
            "d": round(d, 4),
            "p": round(p, 6),
            "ci_lower": round(lo, 6),
            "ci_upper": round(hi, 6),
            "delta_pp": round(abs(np.mean(b) - np.mean(a)) * 100, 2),
            "abs_gain_pct": round(abs_gain_pct, 3),
            "pass_holm": p < 0.05,
            "pass_d": abs(d) >= 0.2,
            "pass_gain": abs_gain_abs >= 0.03,
            "confirmatory_claim": decision,
        })
    df = pd.DataFrame(confirmatory_rows)
    pvals = df["p"].tolist()
    df["holm_adjusted"] = holm_bonferroni(pvals)
    df.to_csv(os.path.join(output_tables_dir, "table_rq1_confirmatory_contrasts.csv"), index=False)
    # Secondary
    sec_rows = []
    for c1, c2 in SECONDARY:
        a = scores_df[scores_df["condition"] == c1]["composite"].values
        b = scores_df[scores_df["condition"] == c2]["composite"].values
        if len(a) < 2 or len(b) < 2:
            continue
        d = _effect_size_d(a, b)
        stat, p = _welch_t(a, b)
        sec_rows.append({"contrast": f"{c1} vs {c2}", "d": d, "p": p})
    pd.DataFrame(sec_rows).to_csv(os.path.join(output_tables_dir, "table_rq1_secondary_contrasts.csv"), index=False)
    return df
