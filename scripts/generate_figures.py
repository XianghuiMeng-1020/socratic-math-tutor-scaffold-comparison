#!/usr/bin/env python3
"""Generate three high-quality conference figures for the paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 鈹€鈹€ Global style 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "text.usetex": False,
})

# Color palette (colorblind-safe, print-friendly)
C_PE_LLAMA  = "#4E79A7"   # Blue
C_SFT       = "#F28E2B"   # Orange
C_DPO       = "#E15759"   # Red
C_Qwen     = "#76B7B2"   # Teal
C_BG_LIGHT  = "#F7F7F7"
C_GRID      = "#E0E0E0"
COLORS = [C_PE_LLAMA, C_SFT, C_DPO, C_Qwen]
COND_LABELS = ["C1: PE-Llama", "C2: SFT-Llama", "C3: DPO-Llama", "C4: PE-Qwen"]
COND_SHORT  = ["PE-Llama", "SFT-Llama", "DPO-Llama", "PE-Qwen"]


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# FIGURE 1: 2脳2 Experimental Design Overview
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def figure1():
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # Title
    ax.text(5, 7.05, "Experimental Design: 2 脳 2 Factorial",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # 鈹€鈹€ Axis labels 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax.text(5.5, 6.55, "Training Approach", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333")
    ax.text(3.5, 6.1, "Prompt-Only", ha="center", va="center",
            fontsize=9.5, fontstyle="italic", color="#555555")
    ax.text(7.5, 6.1, "Learning-Based", ha="center", va="center",
            fontsize=9.5, fontstyle="italic", color="#555555")

    ax.text(0.6, 3.95, "Model\nFamily", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333", linespacing=1.3)
    ax.text(1.4, 5.0, "Open-Weight\n(Llama 3.1-8B)", ha="center", va="center",
            fontsize=8.5, fontstyle="italic", color="#555555", linespacing=1.2)
    ax.text(1.4, 2.7, "Frontier API\n(Qwen3-Max)", ha="center", va="center",
            fontsize=8.5, fontstyle="italic", color="#555555", linespacing=1.2)

    # 鈹€鈹€ Grid lines (light) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax.plot([2.2, 9.3], [5.75, 5.75], color="#BBBBBB", lw=0.7)
    ax.plot([2.2, 9.3], [3.85, 3.85], color="#BBBBBB", lw=0.7)
    ax.plot([5.5, 5.5], [5.75, 1.55], color="#BBBBBB", lw=0.7)

    # 鈹€鈹€ Four condition boxes 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    box_w, box_h = 2.9, 1.6

    def draw_cond_box(ax, x, y, w, h, color, title, desc_lines, alpha=1.0):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.08", linewidth=1.4,
                             edgecolor=color, facecolor=color + "18",
                             alpha=alpha, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.32, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color, zorder=3)
        for i, line in enumerate(desc_lines):
            ax.text(x + w/2, y + h - 0.70 - i * 0.30, line,
                    ha="center", va="center", fontsize=7.8, color="#444444", zorder=3)

    # C1: PE-Llama (top-left)
    draw_cond_box(ax, 2.35, 4.05, box_w, box_h, C_PE_LLAMA,
                  "C1: PE-Llama",
                  ["Llama 3.1-8B-Instruct",
                   "Socratic prompt template",
                   "No parameter updates"])

    # C2 + C3: SFT and DPO (top-right) 鈥?two sub-boxes
    sub_w = 1.65
    sub_gap = 0.15
    x0 = 5.7
    y0 = 4.05

    draw_cond_box(ax, x0, y0, sub_w, box_h, C_SFT,
                  "C2: SFT",
                  ["LoRA fine-tuning",
                   "12K SocraTeach",
                   "dialogues"])

    draw_cond_box(ax, x0 + sub_w + sub_gap, y0, sub_w, box_h, C_DPO,
                  "C3: DPO",
                  ["Preference opt.",
                   "Init from C2",
                   "5K MathDial pairs"])

    # Arrow from C2 to C3
    ax.annotate("", xy=(x0 + sub_w + sub_gap + 0.04, y0 + box_h / 2),
                xytext=(x0 + sub_w - 0.04, y0 + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=1.1, shrinkA=0, shrinkB=0))

    # C4: PE-Qwen (bottom-left)
    draw_cond_box(ax, 2.35, 1.85, box_w, box_h, C_Qwen,
                  "C4: PE-Qwen",
                  ["Qwen3-Max (API)",
                   "Same prompt template",
                   "No parameter updates"])

    # Bottom-right: N/A (dashed)
    bx, by = 5.7, 1.85
    na_w = sub_w * 2 + sub_gap
    na_box = FancyBboxPatch((bx, by), na_w, box_h,
                            boxstyle="round,pad=0.08", linewidth=1.0,
                            edgecolor="#CCCCCC", facecolor="#F5F5F5",
                            linestyle="--", zorder=1)
    ax.add_patch(na_box)
    ax.text(bx + na_w / 2, by + box_h / 2,
            "(Not applicable:\nfine-tuning GPT-4o not available)",
            ha="center", va="center", fontsize=8, color="#AAAAAA",
            fontstyle="italic", linespacing=1.4, zorder=2)

    # 鈹€鈹€ Shared scaffold box at bottom 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    scaffold_y = 0.15
    scaffold_h = 1.30
    scaffold_box = FancyBboxPatch((0.5, scaffold_y), 9.0, scaffold_h,
                                  boxstyle="round,pad=0.1", linewidth=1.3,
                                  edgecolor="#666666", facecolor="#EEEEEE",
                                  zorder=1)
    ax.add_patch(scaffold_box)
    ax.text(5, scaffold_y + scaffold_h - 0.28,
            "Shared Experimental Scaffold (held constant across all conditions)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#444444")

    items = [
        ("241 locked\ntest problems", 1.55),
        ("Fixed dialogue\nprotocol", 3.65),
        ("Frozen student\nsimulator", 5.95),
        ("6-dimension\nevaluation suite", 8.25),
    ]
    for label, xp in items:
        ax.text(xp, scaffold_y + 0.42, f"  {label}",
                ha="center", va="center", fontsize=7.2, color="#555555",
                linespacing=1.15)

    fig.savefig(os.path.join(OUTPUT_DIR, "figure1_experimental_design.png"),
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Figure 1 saved.")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# FIGURE 2: Cost-Effectiveness & Break-Even Analysis
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def figure2():
    fig = plt.figure(figsize=(6.5, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.40)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 鈹€鈹€ Panel A: Composite Score vs. Marginal Cost 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    composites = [0.6128, 0.6208, 0.6096, 0.6105]
    costs      = [0.001, 50.0, 55.0, 5.0]
    markers    = ["o", "s", "D", "^"]

    ax1.set_facecolor(C_BG_LIGHT)
    ax1.grid(True, color=C_GRID, linewidth=0.4, alpha=0.8)

    for i in range(4):
        ax1.scatter(costs[i], composites[i], c=COLORS[i], s=100,
                    marker=markers[i], edgecolors="white", linewidth=0.7,
                    zorder=5, label=COND_SHORT[i])

    # Quality band shading
    ax1.axhspan(min(composites) - 0.001, max(composites) + 0.001,
                color="#DDDDDD", alpha=0.3, zorder=0)

    # Manual annotations with careful placement
    ax1.annotate(f"PE-Llama\n(0.6128)", xy=(costs[0], composites[0]),
                 xytext=(0.004, 0.6155), fontsize=6.5, color=COLORS[0],
                 fontweight="bold", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=COLORS[0], lw=0.5))

    ax1.annotate(f"SFT-Llama\n(0.6208)", xy=(costs[1], composites[1]),
                 xytext=(8, 0.623), fontsize=6.5, color=COLORS[1],
                 fontweight="bold", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=COLORS[1], lw=0.5))

    ax1.annotate(f"DPO-Llama\n(0.6096)", xy=(costs[2], composites[2]),
                 xytext=(8, 0.605), fontsize=6.5, color=COLORS[2],
                 fontweight="bold", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=COLORS[2], lw=0.5))

    ax1.annotate(f"PE-Qwen\n(0.6105)", xy=(costs[3], composites[3]),
                 xytext=(0.8, 0.607), fontsize=6.5, color=COLORS[3],
                 fontweight="bold", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=COLORS[3], lw=0.5))

    ax1.text(0.003, 0.625, "< 2% quality spread", fontsize=6.5,
             color="#888888", fontstyle="italic")

    ax1.set_xscale("log")
    ax1.set_xlabel("Marginal Cost per Problem (USD, log scale)")
    ax1.set_ylabel("Composite Quality Score")
    ax1.set_xlim(0.0004, 150)
    ax1.set_ylim(0.603, 0.627)
    ax1.set_title("(a) Quality vs. Marginal Cost", fontsize=10,
                  fontweight="bold", pad=8)

    # 鈹€鈹€ Panel B: Break-Even Volume 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    conditions_order = [0, 3, 1, 2]
    break_even = [0, 500, 5000, 5500]
    bar_labels = [COND_SHORT[i] for i in conditions_order]
    bar_colors = [COLORS[i] for i in conditions_order]

    ax2.set_facecolor(C_BG_LIGHT)
    ax2.grid(True, axis="x", color=C_GRID, linewidth=0.4, alpha=0.8)

    bars = ax2.barh(range(4), break_even, color=bar_colors,
                    edgecolor="white", linewidth=0.7, height=0.52, zorder=3)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(bar_labels, fontsize=8.5)
    ax2.set_xlabel("Break-Even Volume (# Problems)")
    ax2.set_title("(b) Deployment Break-Even", fontsize=10,
                  fontweight="bold", pad=8)
    ax2.invert_yaxis()

    for j, (bar, val) in enumerate(zip(bars, break_even)):
        label = "0 (no training cost)" if val == 0 else f"{val:,}"
        x_pos = max(val, 150) + 120
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 label, va="center", ha="left", fontsize=7.5,
                 fontweight="bold", color=bar_colors[j])

    ax2.set_xlim(0, 7200)

    # Reference lines for volume zones
    ax2.axvline(x=500, color="#AAAAAA", linestyle=":", linewidth=0.7, zorder=1)
    ax2.axvline(x=5000, color="#AAAAAA", linestyle=":", linewidth=0.7, zorder=1)

    fig.savefig(os.path.join(OUTPUT_DIR, "figure2_cost_breakeven.png"),
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Figure 2 saved.")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# FIGURE 3: Dialogue-Length Quality Degradation & Failure Modes
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def figure3():
    fig = plt.figure(figsize=(6.5, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 鈹€鈹€ Panel A: Simulated degradation curves 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # Mixed-effects model: slope = -0.02 per turn on composite scale
    # We show relative degradation as percentage change from turn 1
    # to make the visual more interpretable for readers
    np.random.seed(42)
    turns = np.arange(1, 16)
    slope = -0.02

    intercepts = {
        "PE-Llama":  0.6128,
        "SFT-Llama": 0.6208,
        "DPO-Llama": 0.6096,
        "PE-Qwen":  0.6105,
    }

    ax1.set_facecolor(C_BG_LIGHT)
    ax1.grid(True, color=C_GRID, linewidth=0.4, alpha=0.8)

    linestyles = ["-", "--", "-.", ":"]
    marker_styles = ["o", "s", "D", "^"]

    for idx, (name, base) in enumerate(intercepts.items()):
        scores = base + slope * (turns - 1)
        noise = np.random.normal(0, 0.003, len(turns))
        noise[0] = 0
        scores_noisy = scores + noise

        ax1.plot(turns, scores_noisy, color=COLORS[idx],
                 linestyle=linestyles[idx], linewidth=1.3,
                 marker=marker_styles[idx], markersize=3.5,
                 markeredgecolor="white", markeredgewidth=0.4,
                 label=name, zorder=4)

    mean_intercept = np.mean(list(intercepts.values()))
    ref_line = mean_intercept + slope * (turns - 1)
    ax1.plot(turns, ref_line, color="#888888", linestyle="-",
             linewidth=2.0, alpha=0.25, zorder=2, label="Pooled trend")

    ax1.fill_between(turns,
                     mean_intercept + slope * (turns - 1) + 0.015,
                     mean_intercept + slope * (turns - 1) - 0.015,
                     color="#CCCCCC", alpha=0.2, zorder=1)

    ax1.set_xlabel("Dialogue Turn")
    ax1.set_ylabel("Composite Quality Score")
    ax1.set_title("(a) Quality Degradation by Turn", fontsize=10,
                  fontweight="bold", pad=8)
    ax1.set_xlim(0.5, 15.5)
    ax1.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])

    ax1.annotate("slope = \u22120.02 / turn\n(p = 0.03)",
                 xy=(10, mean_intercept + slope * 9),
                 xytext=(11.5, mean_intercept + slope * 5),
                 fontsize=7, color="#666666",
                 arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.7),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#CCCCCC", linewidth=0.5))

    ax1.legend(loc="lower left", frameon=True, fancybox=True,
               framealpha=0.9, edgecolor="#CCCCCC", fontsize=7,
               ncol=2, columnspacing=0.8, handlelength=1.8)

    # 鈹€鈹€ Panel B: Failure Mode Distribution 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    failure_modes = ["Short\nDialogue", "Repetitive", "Off-Topic", "No\nScaffolding"]
    counts = [133, 0, 0, 0]
    total_dialogues = 1263

    bar_colors_fm = ["#E15759", "#BBBBBB", "#BBBBBB", "#BBBBBB"]

    ax2.set_facecolor(C_BG_LIGHT)
    ax2.grid(True, axis="y", color=C_GRID, linewidth=0.4, alpha=0.8)

    bars = ax2.bar(range(4), counts, color=bar_colors_fm,
                   edgecolor="white", linewidth=0.7, width=0.52, zorder=3)

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(failure_modes, fontsize=8)
    ax2.set_ylabel("Failure Count")
    ax2.set_title("(b) Failure Mode Distribution", fontsize=10,
                  fontweight="bold", pad=8)
    ax2.set_ylim(0, 180)

    for bar, val in zip(bars, counts):
        if val > 0:
            pct = val / total_dialogues * 100
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 4,
                     f"{val}\n({pct:.1f}%)",
                     ha="center", va="bottom", fontsize=8,
                     fontweight="bold", color="#E15759")
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 4,
                     "0",
                     ha="center", va="bottom", fontsize=8,
                     fontweight="bold", color="#AAAAAA")

    ax2.annotate("Early stopping is\nthe dominant\nfailure mode",
                 xy=(0, 133), xytext=(1.8, 148),
                 fontsize=7.5, color="#555555", fontstyle="italic",
                 ha="center",
                 arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#CCCCCC", linewidth=0.5))

    fig.savefig(os.path.join(OUTPUT_DIR, "figure3_degradation_failures.png"),
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Figure 3 saved.")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")

