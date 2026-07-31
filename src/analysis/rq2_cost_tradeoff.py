"""RQ2: Quality-cost trade-off, 3 weighting profiles, break-even analysis.

Updated to 2026 API pricing and corrected break-even:
- C1/C2/C3 (Llama): infrastructure cost only (GPU server amortized).
  Training costs (SFT, DPO) are one-time; inference is near-zero marginal.
- C4 (GPT-4o): per-token API billing; varies with dialogue length.
- Break-even: only C2 vs C1, C3 vs C2 (Llama-only) reported as primary.
  GPT-4o vs Llama: discussed separately (different cost structure).
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


# 2026 approximate pricing (USD)
COSTS_2026 = {
    # Training costs (one-time, amortized over expected deployment volume)
    "C1_PE_Llama": {
        "training_usd": 0.0,           # no training
        "inference_per_dialogue": 0.0,  # local GPU inference
        "note": "Base Llama 3.1-8B + prompt; ~$0 marginal inference on owned GPU",
    },
    "C2_SFT_Llama": {
        "training_usd": 45.0,          # ~3 epochs LoRA SFT on RTX 4080
        "inference_per_dialogue": 0.0,
        "note": "SFT LoRA adapter; one-time training cost only",
    },
    "C3_DPO_Llama": {
        "training_usd": 55.0,          # SFT (45) + DPO preference training (10)
        "inference_per_dialogue": 0.0,
        "note": "DPO from SFT checkpoint; one-time training cost",
    },
    "C4_PE_Qwen": {
        "training_usd": 0.0,
        "inference_per_dialogue": 0.045,  # ~2.8k tokens/dialogue x Qwen3-Max $0.016/1k (DashScope 2026)
        "note": "Qwen3-Max DashScope API; $0.016/1K tokens (2026 pricing)",
    },
}

DEPLOYMENT_VOLUMES = [100, 500, 1000, 5000, 10000, 50000]

PROFILES = {
    "risk_averse": {"QQ": 0.30, "SD": 0.30, "SLR": 0.20, "EDT": 0.10, "DC": 0.05, "MC_Verified": 0.05},
    "balanced": {"QQ": 0.20, "SD": 0.20, "SLR": 0.25, "EDT": 0.15, "DC": 0.10, "MC_Verified": 0.10},
    "learning_first": {"QQ": 0.15, "SD": 0.15, "SLR": 0.30, "EDT": 0.25, "DC": 0.10, "MC_Verified": 0.05},
}


def _total_cost(condition: str, volume: int) -> float:
    """Total deployment cost = training + volume * per_dialogue."""
    info = COSTS_2026.get(condition, {})
    return info.get("training_usd", 0.0) + volume * info.get("inference_per_dialogue", 0.0)


def _break_even_volume(cond_a: str, cond_b: str, quality_gap: float) -> Optional[int]:
    """Volume at which cond_a is cost-effective vs cond_b given quality_gap (in composite points).

    Only meaningful for Llama-only comparison (same inference cost).
    """
    train_a = COSTS_2026.get(cond_a, {}).get("training_usd", 0.0)
    train_b = COSTS_2026.get(cond_b, {}).get("training_usd", 0.0)
    delta_train = train_a - train_b
    delta_inf = (
        COSTS_2026.get(cond_a, {}).get("inference_per_dialogue", 0.0)
        - COSTS_2026.get(cond_b, {}).get("inference_per_dialogue", 0.0)
    )
    if delta_inf == 0 and quality_gap > 0 and delta_train > 0:
        # Break-even: training cost spread over volume; how many dialogues justify it?
        return int(np.ceil(delta_train / max(quality_gap * 0.01, 1e-6)))
    return None


def run_rq2_cost(
    project_root: str,
    scores_df: pd.DataFrame,
    output_tables_dir: str,
) -> pd.DataFrame:
    """Compute cost-effectiveness under 3 stakeholder profiles + break-even analysis."""
    ensure_dir(output_tables_dir)

    METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]
    rows = []
    for cond in scores_df["condition"].unique():
        sub = scores_df[scores_df["condition"] == cond]
        for profile_name, profile in PROFILES.items():
            comp = 0.0
            for m, w in profile.items():
                if m in sub.columns:
                    comp += float(sub[m].mean()) * w
                else:
                    comp += 0.5 * w
            rows.append({
                "condition": cond,
                "profile": profile_name,
                "quality_composite": round(comp, 4),
                "training_cost_usd": COSTS_2026.get(cond, {}).get("training_usd", 0),
                "inference_per_dialogue": COSTS_2026.get(cond, {}).get("inference_per_dialogue", 0),
                "note": COSTS_2026.get(cond, {}).get("note", ""),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_tables_dir, "table_rq2_cost_effectiveness.csv"), index=False)

    # Break-even: Llama conditions only
    cond_means = scores_df.groupby("condition")["composite"].mean()
    be_rows = []
    llama_pairs = [("C2_SFT_Llama", "C1_PE_Llama"), ("C3_DPO_Llama", "C2_SFT_Llama")]
    for c_hi, c_lo in llama_pairs:
        q_hi = float(cond_means.get(c_hi, 0.5))
        q_lo = float(cond_means.get(c_lo, 0.5))
        train_hi = COSTS_2026.get(c_hi, {}).get("training_usd", 0)
        train_lo = COSTS_2026.get(c_lo, {}).get("training_usd", 0)
        delta_train = train_hi - train_lo
        delta_quality = q_hi - q_lo
        if delta_quality > 0 and delta_train > 0:
            be_volume = int(np.ceil(delta_train / (delta_quality + 1e-9)))
        elif delta_quality <= 0:
            be_volume = None
        else:
            be_volume = 0
        be_rows.append({
            "comparison": f"{c_hi} vs {c_lo}",
            "delta_training_cost_usd": delta_train,
            "delta_composite": round(delta_quality, 4),
            "break_even_volume": be_volume,
            "note": "Llama-only (same ~$0 inference cost)" if be_volume else "Training cost not justified by quality gain",
        })

    # Qwen3-Max (C4) break-even: at what volume does Qwen API cost exceed Llama training cost?
    for c_hi in ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama"]:
        training = COSTS_2026.get(c_hi, {}).get("training_usd", 0)
        inf_qwen = COSTS_2026.get("C4_PE_Qwen", {}).get("inference_per_dialogue", 0.045)
        # Volume at which Qwen cumulative API cost equals Llama training cost
        qwen_be = int(np.ceil(training / inf_qwen)) if inf_qwen > 0 and training > 0 else None
        note_str = f"Qwen API cost surpasses {c_hi} training cost at >{qwen_be} dialogues" if qwen_be else "C1 zero training; Qwen API always more expensive"
        be_rows.append({
            "comparison": f"C4_PE_Qwen vs {c_hi}",
            "delta_training_cost_usd": -training,
            "delta_composite": round(float(cond_means.get("C4_PE_Qwen", 0.5) - cond_means.get(c_hi, 0.5)), 4),
            "break_even_volume": qwen_be,
            "note": note_str,
        })

    pd.DataFrame(be_rows).to_csv(
        os.path.join(output_tables_dir, "table_rq2_break_even_analysis.csv"), index=False
    )

    # Deployment cost over volumes
    vol_rows = []
    for cond in scores_df["condition"].unique():
        for v in DEPLOYMENT_VOLUMES:
            vol_rows.append({
                "condition": cond,
                "volume": v,
                "total_cost_usd": round(_total_cost(cond, v), 2),
            })
    pd.DataFrame(vol_rows).to_csv(
        os.path.join(output_tables_dir, "table_rq2_cost_over_volume.csv"), index=False
    )

    return df


# Make Optional importable
from typing import Optional
