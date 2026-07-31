#!/usr/bin/env python3
"""Phase 4D & 5C: Generate all extended figures for submission.

Produces:
  fig_per_dim_ci.pdf         - Per-dimension CI plots (6 panels, 4 conditions each)
  fig_per_turn_trajectory.pdf - Per-turn composite score trajectory
  fig_scaffold_curve.pdf     - Scaffold strength ablation curve
  fig_bootstrap_violin.pdf   - Bootstrap distribution violin/raincloud plots
  fig_heatmap_profile.pdf    - Condition x profile heatmap
  fig_cost_model.pdf         - Updated cost model
  fig_simulator_plausibility.pdf - Simulator plausibility distribution

Usage:
    python scripts/run_17_generate_figures_extended.py
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available; figures will not be generated")

from src.utils.io import ensure_dir


FIGURES_DIR = os.path.join(ROOT, "submission", "figures")
TABLES_DIR = os.path.join(ROOT, "outputs", "tables")

CONDITION_LABELS = {
    "C1_PE_Llama": "PE-Llama",
    "C2_SFT_Llama": "SFT-Llama",
    "C3_DPO_Llama": "DPO-Llama",
    "C4_PE_Qwen": "PE-Qwen",
}
CONDITION_COLORS = {
    "C1_PE_Llama": "#4C72B0",
    "C2_SFT_Llama": "#DD8452",
    "C3_DPO_Llama": "#55A868",
    "C4_PE_Qwen": "#C44E52",
}
METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]
METRIC_LABELS = {
    "QQ": "Question Quality",
    "SD": "Scaffolding Depth",
    "SLR": "Solution Leaking\n(inverted)",
    "EDT": "Error Diagnosis",
    "DC": "Dialogue Coherence",
    "MC_Verified": "Math Correctness",
}
CONDITIONS = list(CONDITION_LABELS.keys())


def _bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, ci: float = 95) -> tuple:
    """Bootstrap CI for the mean."""
    if len(arr) < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean())
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, ci + (100 - ci) / 2)
    return float(np.mean(arr)), float(lo), float(hi)


def fig_per_dim_ci():
    """6-panel CI plot: one panel per metric, 4 bars per panel."""
    raw_path = os.path.join(TABLES_DIR, "table_metric_raw_scores.csv")
    if not os.path.isfile(raw_path):
        print("Skipping fig_per_dim_ci: raw scores not found")
        return
    df = pd.read_csv(raw_path)
    if "scorable" in df.columns:
        df = df[df["scorable"]]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        xs, means, lowers, uppers = [], [], [], []
        for i, cond in enumerate(CONDITIONS):
            sub = df[df["condition"] == cond][metric].dropna().values
            if len(sub) == 0:
                continue
            mean, lo, hi = _bootstrap_ci(sub)
            xs.append(i)
            means.append(mean)
            lowers.append(mean - lo)
            uppers.append(hi - mean)
            color = CONDITION_COLORS.get(cond, "gray")
            ax.bar(i, mean, 0.6, color=color, alpha=0.85)
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", color="black", capsize=4, lw=1.5)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in CONDITIONS], fontsize=7, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score (0-1)", fontsize=8)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-Dimension Scores with 95% Bootstrap CI", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_per_dim_ci.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: fig_per_dim_ci.pdf")


def fig_per_turn_trajectory():
    """Per-turn composite score trajectory for 4 conditions."""
    traj_path = os.path.join(TABLES_DIR, "table_per_turn_trajectory.csv")
    if not os.path.isfile(traj_path):
        print("Skipping fig_per_turn_trajectory: trajectory data not found")
        return
    df = pd.read_csv(traj_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].groupby("turn")["composite"].agg(["mean", "std"]).reset_index()
        if sub.empty:
            continue
        label = CONDITION_LABELS.get(cond, cond)
        color = CONDITION_COLORS.get(cond, "gray")
        ax.plot(sub["turn"], sub["mean"], marker="o", label=label, color=color, lw=2)
        ax.fill_between(
            sub["turn"],
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Dialogue Turn", fontsize=11)
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Per-Turn Composite Score Trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_per_turn_trajectory.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: fig_per_turn_trajectory.pdf")


def fig_scaffold_curve():
    """Scaffold strength ablation curve (3 levels x 6 dims + composite)."""
    abl_path = os.path.join(TABLES_DIR, "table_scaffold_ablation_raw.csv")
    if not os.path.isfile(abl_path):
        print("Skipping fig_scaffold_curve: ablation data not found")
        return
    df = pd.read_csv(abl_path)
    if df.empty:
        return

    SCAFFOLD_ORDER = ["C0a_NoScaffold", "C0b_WeakScaffold", "C1_FullScaffold"]
    SCAFFOLD_LABELS = {"C0a_NoScaffold": "No Scaffold", "C0b_WeakScaffold": "Weak Scaffold", "C1_FullScaffold": "Full Scaffold"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(SCAFFOLD_ORDER))
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    for dim, color, ls in zip(
        METRICS + ["composite"],
        ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "black"],
        ["-", "--", "-.", ":", "-", "--", "-"],
    ):
        col = dim
        if col not in df.columns:
            continue
        means = [df[df["condition"] == c][col].mean() for c in SCAFFOLD_ORDER]
        lw = 2.5 if dim == "composite" else 1.2
        label = "Composite" if dim == "composite" else METRIC_LABELS.get(dim, dim)
        ax.plot(x, means, marker="o", color=color, ls=ls, lw=lw, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([SCAFFOLD_LABELS.get(c, c) for c in SCAFFOLD_ORDER], fontsize=10)
    ax.set_ylabel("Score (0-1)", fontsize=11)
    ax.set_title("Scaffold Strength Ablation: Score by Scaffold Level", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_scaffold_curve.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: fig_scaffold_curve.pdf")


def fig_bootstrap_violin():
    """Bootstrap violin/raincloud plots of composite score distribution per condition."""
    raw_path = os.path.join(TABLES_DIR, "table_metric_raw_scores.csv")
    if not os.path.isfile(raw_path):
        print("Skipping fig_bootstrap_violin: raw scores not found")
        return
    df = pd.read_csv(raw_path)
    if "scorable" in df.columns:
        df = df[df["scorable"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(CONDITIONS))
    for i, cond in enumerate(CONDITIONS):
        sub = df[df["condition"] == cond]["composite"].dropna().values
        if len(sub) == 0:
            continue
        color = CONDITION_COLORS.get(cond, "gray")
        # Violin
        vp = ax.violinplot([sub], positions=[i], widths=0.5, showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(2)
        # Bootstrap CI
        mean, lo, hi = _bootstrap_ci(sub)
        ax.scatter([i], [mean], color="black", zorder=5, s=30)
        ax.plot([i, i], [lo, hi], color="black", lw=2)

    ax.set_xticks(positions)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in CONDITIONS], fontsize=10)
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Composite Score Distribution by Condition\n(violin + 95% bootstrap CI)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_bootstrap_violin.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: fig_bootstrap_violin.pdf")


def fig_heatmap_profile():
    """Condition x learner profile heatmap of composite scores."""
    hm_path = os.path.join(TABLES_DIR, "table_per_profile_heatmap.csv")
    if not os.path.isfile(hm_path):
        print("Skipping fig_heatmap_profile: heatmap data not found")
        return
    heatmap = pd.read_csv(hm_path, index_col=0)
    if heatmap.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(heatmap.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Composite Score")
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_yticks(np.arange(heatmap.shape[0]))
    ax.set_xticklabels(heatmap.columns, fontsize=9)
    ax.set_yticklabels([CONDITION_LABELS.get(r, r) for r in heatmap.index], fontsize=9)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.6 else "black", fontsize=9)
    ax.set_title("Composite Score: Condition 脳 Learner Profile", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_heatmap_profile.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: fig_heatmap_profile.pdf")


def fig_cost_model():
    """Cost model: total deployment cost vs volume for all conditions (2026 pricing)."""
    cost_path = os.path.join(TABLES_DIR, "table_rq2_cost_over_volume.csv")
    if not os.path.isfile(cost_path):
        print("Skipping fig_cost_model: cost data not found")
        return
    df = pd.read_csv(cost_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        color = CONDITION_COLORS.get(cond, "gray")
        ax.plot(sub["volume"], sub["total_cost_usd"], marker="o", markersize=4,
                label=CONDITION_LABELS.get(cond, cond), color=color, lw=2)

    ax.set_xlabel("Deployment Volume (dialogues)", fontsize=11)
    ax.set_ylabel("Total Cost (USD)", fontsize=11)
    ax.set_title("Deployment Cost vs Volume (2026 Pricing)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_cost_model.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: fig_cost_model.pdf")


def fig_simulator_plausibility():
    """Simulator plausibility score distribution by profile."""
    plaus_path = os.path.join(TABLES_DIR, "table_simulator_plausibility_scores.csv")
    if not os.path.isfile(plaus_path):
        print("Skipping fig_simulator_plausibility: plausibility data not found")
        return
    df = pd.read_csv(plaus_path)
    if df.empty:
        return

    profiles = ["struggling", "progressing", "advanced"]
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for prof, color in zip(profiles, colors):
        sub = df[df["profile"] == prof]["plausibility_score"].dropna().values
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=np.arange(0.5, 6, 1), alpha=0.6, color=color, label=prof.capitalize(), edgecolor="black", lw=0.5)

    ax.axvline(x=3.5, color="red", ls="--", lw=1.5, label="Threshold (3.5)")
    ax.set_xlabel("Plausibility Score (1-5)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Simulator Turn Plausibility by Learner Profile", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_simulator_plausibility.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: fig_simulator_plausibility.pdf")


def main():
    if not HAS_MPL:
        print("matplotlib not available; cannot generate figures")
        sys.exit(1)

    ensure_dir(FIGURES_DIR)
    print(f"Generating figures to: {FIGURES_DIR}\n")

    fig_per_dim_ci()
    fig_per_turn_trajectory()
    fig_scaffold_curve()
    fig_bootstrap_violin()
    fig_heatmap_profile()
    fig_cost_model()
    fig_simulator_plausibility()

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("run_17_generate_figures_extended: DONE")


if __name__ == "__main__":
    main()

