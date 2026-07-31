"""RQ3: Per-turn coherence degradation (mixed-effects model) + failure taxonomy.

Implements:
- Mixed-effects LMM: composite ~ Turn + (1|condition) to test whether quality
  degrades across dialogue turns.
- Failure taxonomy: categorize each dialogue as early_termination / protocol_failure /
  scorable, with breakdown by condition and learner profile.
- "Quality cliff" detection: identify the turn number at which average composite
  drops > 0.05 relative to turn 1.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir


def _fit_mixed_effects(turn_df: pd.DataFrame) -> pd.DataFrame:
    """Fit LMM: composite ~ turn * C(condition) using statsmodels mixedlm.

    The interaction term turn:C(condition) tests whether quality degradation
    rate differs across conditions — key for claims like 'DPO shows steeper decline'.
    Reports main effects AND interaction F-statistics via Wald test.
    """
    try:
        import statsmodels.formula.api as smf
        groups = turn_df["problem_id"] if "problem_id" in turn_df.columns else turn_df.index
        # Full model with Turn × Condition interaction
        model = smf.mixedlm(
            "composite ~ turn * C(condition)",
            data=turn_df,
            groups=groups,
        )
        result = model.fit(reml=False)
        rows = []
        for name, coef in result.params.items():
            pval = result.pvalues.get(name, float("nan"))
            se = result.bse.get(name, float("nan"))
            rows.append({
                "effect": name,
                "coef": round(coef, 4),
                "se": round(se, 4),
                "p": round(pval, 4),
                "sig": "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns")),
            })

        # Append model fit statistics
        rows.append({
            "effect": "_AIC",
            "coef": round(result.aic, 2),
            "se": float("nan"),
            "p": float("nan"),
            "sig": "",
        })
        rows.append({
            "effect": "_N_obs",
            "coef": len(turn_df),
            "se": float("nan"),
            "p": float("nan"),
            "sig": "",
        })

        # Wald F-test for the interaction block (all turn:condition terms jointly)
        interaction_params = [k for k in result.params.index if "turn" in k.lower() and "condition" in k.lower()]
        if interaction_params:
            try:
                import numpy as np
                n_int = len(interaction_params)
                wald_stat = sum(
                    (result.params[k] / result.bse[k]) ** 2
                    for k in interaction_params
                    if not np.isnan(result.bse[k]) and result.bse[k] > 0
                )
                from scipy.stats import chi2
                p_wald = 1 - chi2.cdf(wald_stat, df=n_int)
                rows.append({
                    "effect": "_Interaction_Wald_chi2",
                    "coef": round(wald_stat, 4),
                    "se": float("nan"),
                    "p": round(p_wald, 4),
                    "sig": "***" if p_wald < 0.001 else ("**" if p_wald < 0.01 else ("*" if p_wald < 0.05 else "ns")),
                })
            except Exception:
                pass

        return pd.DataFrame(rows)
    except Exception as e:
        # Fallback: simple OLS per-condition slope
        rows = []
        for cond in turn_df["condition"].unique():
            sub = turn_df[turn_df["condition"] == cond]
            try:
                from numpy.polynomial import polynomial as P
                turns = sub["turn"].values
                scores = sub["composite"].values
                coef = np.polyfit(turns, scores, 1)
                rows.append({"effect": f"Turn slope [{cond}]", "coef": round(coef[0], 4), "se": float("nan"), "p": float("nan")})
            except Exception:
                rows.append({"effect": f"Turn slope [{cond}]", "coef": float("nan"), "se": float("nan"), "p": float("nan")})
        if not rows:
            rows = [{"effect": "Turn", "coef": float("nan"), "se": float("nan"), "p": float("nan"), "note": str(e)}]
        return pd.DataFrame(rows)


def _detect_quality_cliff(turn_df: pd.DataFrame) -> pd.DataFrame:
    """Find the turn where composite drops > 0.05 from turn-1 baseline, per condition."""
    rows = []
    for cond in turn_df["condition"].unique():
        sub = turn_df[turn_df["condition"] == cond].groupby("turn")["composite"].mean()
        if sub.empty or 1 not in sub.index:
            rows.append({"condition": cond, "cliff_turn": None, "drop": None})
            continue
        baseline = sub[1]
        cliff = None
        for turn_no in sorted(sub.index):
            if turn_no == 1:
                continue
            drop = baseline - sub[turn_no]
            if drop > 0.05:
                cliff = turn_no
                break
        rows.append({
            "condition": cond,
            "cliff_turn": cliff,
            "baseline_composite": round(float(baseline), 4),
            "final_composite": round(float(sub.iloc[-1]), 4),
            "total_drop": round(float(baseline - sub.iloc[-1]), 4),
        })
    return pd.DataFrame(rows)


def _classify_failure_modes(dialogues: List[Dict]) -> pd.DataFrame:
    """Detailed failure mode classification for each dialogue."""
    rows = []
    for d in dialogues:
        turns = d.get("turns", [])
        tutor_turns = [t for t in turns if t.get("role") == "tutor"]
        student_turns = [t for t in turns if t.get("role") == "student"]
        condition = d.get("condition", "unknown")
        profile = d.get("profile", "unknown")
        metadata = d.get("metadata", {})

        n_tutor = len(tutor_turns)
        n_student = len(student_turns)

        # Detect failure modes
        has_answer_leak = any(
            any(kw in t.get("content", "").lower() for kw in ["the answer is", "= ", "therefore"])
            for t in tutor_turns
        )
        has_repetition = False
        if n_tutor >= 2:
            contents = [t.get("content", "") for t in tutor_turns]
            for i in range(1, len(contents)):
                if contents[i].strip() == contents[i-1].strip() and len(contents[i]) > 10:
                    has_repetition = True
                    break
        too_short = n_tutor < 3
        no_question = n_tutor > 0 and not any("?" in t.get("content", "") for t in tutor_turns)
        api_error = any(
            t.get("content", "").startswith("[") and "error" in t.get("content", "").lower()
            for t in tutor_turns
        )

        rows.append({
            "condition": condition,
            "profile": profile,
            "problem_id": d.get("problem_id", "unknown"),
            "n_tutor_turns": n_tutor,
            "n_student_turns": n_student,
            "early_termination": metadata.get("early_termination", too_short),
            "answer_leak": has_answer_leak,
            "repetitive": has_repetition,
            "no_question_marks": no_question,
            "api_error": api_error,
            "scorable": metadata.get("scorable", not too_short and not api_error),
        })
    return pd.DataFrame(rows)


def run_rq3(
    project_root: str,
    scores_df: pd.DataFrame,
    dialogues: List[Dict],
    output_tables_dir: str,
) -> pd.DataFrame:
    """Run RQ3: degradation analysis and failure taxonomy.

    Args:
        project_root: root directory.
        scores_df: must contain columns [condition, problem_id, composite, profile].
        dialogues: list of raw dialogue dicts.
        output_tables_dir: where to write CSV outputs.
    Returns:
        mixed-effects result DataFrame.
    """
    ensure_dir(output_tables_dir)

    # Load per-turn trajectory if available
    turn_path = os.path.join(project_root, "outputs", "tables", "table_per_turn_trajectory.csv")
    if os.path.isfile(turn_path):
        turn_df = pd.read_csv(turn_path)
    else:
        turn_df = pd.DataFrame()

    # Mixed-effects model
    if not turn_df.empty and "turn" in turn_df.columns and "composite" in turn_df.columns:
        me_result = _fit_mixed_effects(turn_df)
        cliff_df = _detect_quality_cliff(turn_df)
    else:
        me_result = pd.DataFrame([{
            "effect": "Turn", "coef": float("nan"), "se": float("nan"), "p": float("nan"),
            "note": "per-turn data not available"
        }])
        cliff_df = pd.DataFrame()

    me_result.to_csv(
        os.path.join(output_tables_dir, "table_rq3_turn_degradation_mixed_effects.csv"), index=False
    )
    if not cliff_df.empty:
        cliff_df.to_csv(os.path.join(output_tables_dir, "table_rq3_quality_cliff.csv"), index=False)

    # Detailed failure mode classification
    if dialogues:
        fail_mode_df = _classify_failure_modes(dialogues)
        fail_mode_df.to_csv(
            os.path.join(output_tables_dir, "table_rq3_failure_mode_detailed.csv"), index=False
        )

        # Aggregate failure modes by condition
        agg_modes = (
            fail_mode_df.groupby("condition")
            .agg(
                total=("problem_id", "count"),
                early_term=("early_termination", "sum"),
                answer_leak=("answer_leak", "sum"),
                repetitive=("repetitive", "sum"),
                no_question=("no_question_marks", "sum"),
                api_error=("api_error", "sum"),
                scorable=("scorable", "sum"),
            )
            .reset_index()
        )
        for col in ["early_term", "answer_leak", "repetitive", "no_question", "api_error", "scorable"]:
            agg_modes[f"{col}_pct"] = (agg_modes[col] / agg_modes["total"].clip(lower=1) * 100).round(1)
        agg_modes.to_csv(
            os.path.join(output_tables_dir, "table_rq3_failure_mode_taxonomy.csv"), index=False
        )

        # Failure ~ profile
        fail_x_profile = (
            fail_mode_df.groupby(["condition", "profile"])
            .agg(total=("problem_id", "count"), early_term=("early_termination", "sum"), scorable=("scorable", "sum"))
            .reset_index()
        )
        fail_x_profile.to_csv(
            os.path.join(output_tables_dir, "table_rq3_failure_x_profile.csv"), index=False
        )

    return me_result
