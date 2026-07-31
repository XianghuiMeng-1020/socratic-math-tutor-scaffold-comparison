"""RQ1 planned contrasts: Wilcoxon signed-rank on paired composite differences."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CONTRASTS = [
    ("C1_PE_Llama", "C2_SFT_Llama", "C1_vs_C2"),
    ("C2_SFT_Llama", "C3_DPO_Llama", "C2_vs_C3"),
    ("C1_PE_Llama", "C4_PE_Qwen", "C1_vs_C4"),
]

def holm_adjust(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (n - i)))
        adj[idx] = running
    return adj.tolist()

def main(master_csv: str, out_csv: str):
    master = pd.read_csv(master_csv)
    master["pairing_key"] = master["problem_id"].astype(str) + "::" + master["profile"].astype(str)
    rows = []
    pvals = []
    for a, b, name in CONTRASTS:
        da = master[master.condition == a][["pairing_key", "composite"]]
        db = master[master.condition == b][["pairing_key", "composite"]]
        m = da.merge(db, on="pairing_key", suffixes=("_a", "_b"))
        diff = (m["composite_b"] - m["composite_a"]).values
        nz = diff[diff != 0]
        stat, p = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox") if len(nz) else (0.0, 1.0)
        rows.append({"contrast": name, "paired_N": len(diff), "mean_diff": float(diff.mean()),
                     "wilcoxon_stat": float(stat), "p_raw": float(p)})
        pvals.append(float(p))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adjust(pvals)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.master, args.out)
