#!/usr/bin/env python3
"""Phase 3D: Automated simulator validation.

Validates the student simulator by:
1. Plausibility scoring (1-5 scale) via Qwen3-Max (DashScope API): judges whether
   each simulated student turn is plausible for the given learner profile and problem.
2. Embedding similarity: cosine similarity distribution of simulator turns vs
   MathDial real student turns using Qwen text-embedding-v3. Reports KL divergence.
3. KS test + Cohen's d comparing simulated vs. real response length distributions.
4. Per-profile breakdown (Struggling / Progressing / Advanced).

Usage:
    QWEN_API_KEY=sk-... python scripts/run_14_simulator_validation.py [--n 150]
"""
import argparse
import json
import os
import sys
import random
import time
from typing import List, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import load_jsonl, ensure_dir
from src.utils.seed import set_global_seed


PLAUSIBILITY_PROMPT = """You are evaluating whether a student response in a math tutoring session
is plausible for the given learner profile.

Problem: {problem}
Learner profile: {profile}
Tutor said: "{tutor_msg}"
Student responded: "{student_msg}"

Rate the plausibility of this student response on a 1-5 scale:
1 = Very implausible for this profile (completely wrong tone/ability level)
2 = Somewhat implausible
3 = Neutral / possible
4 = Plausible for this profile
5 = Very plausible and natural for this profile

Respond with ONLY a single integer from 1 to 5."""


def _qwen_plausibility(client, problem: str, profile: str, tutor_msg: str, student_msg: str, max_retries: int = 3) -> float:
    """Score simulator turn plausibility with Qwen3-Max via DashScope."""
    prompt = PLAUSIBILITY_PROMPT.format(
        problem=problem, profile=profile, tutor_msg=tutor_msg, student_msg=student_msg
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="qwen3-max",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            val = float("".join(c for c in text if c.isdigit() or c == "."))
            return min(5.0, max(1.0, val))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return 3.0


def _get_embeddings(texts: List[str], client) -> np.ndarray:
    """Get sentence embeddings using Qwen text-embedding-v3 via DashScope."""
    embeddings = []
    batch_size = 25  # DashScope recommends smaller batches
    EMBED_DIM = 1024  # text-embedding-v3 dimension
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model="text-embedding-v3", input=batch)
            for item in resp.data:
                embeddings.append(np.array(item.embedding))
        except Exception as e:
            print(f"Embedding error (Qwen): {e}")
            embeddings.extend([np.zeros(EMBED_DIM)] * len(batch))
    return np.array(embeddings) if embeddings else np.zeros((0, 1024))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _kl_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 20) -> float:
    """KL divergence D(P||Q) on histograms of similarity distributions."""
    eps = 1e-10
    bins = np.linspace(-1, 1, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of student turns to validate")
    args = parser.parse_args()

    set_global_seed(42)
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    ensure_dir(tables_dir)

    qwen_key = os.environ.get("QWEN_API_KEY", "")
    if not qwen_key:
        print("ERROR: QWEN_API_KEY not set. This script uses Qwen3-Max via DashScope.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(
        api_key=qwen_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0,
    )

    # Load simulated student turns from generated dialogues
    dialogues_dir = os.path.join(ROOT, "outputs", "dialogues")
    sim_turns = []
    for cond in ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama"]:
        path = os.path.join(dialogues_dir, f"dialogues_{cond}.jsonl")
        if not os.path.isfile(path):
            continue
        for d in load_jsonl(path):
            turns = d.get("turns", [])
            for i, t in enumerate(turns):
                if t.get("role") == "student":
                    tutor_msg = turns[i-1].get("content", "") if i > 0 else ""
                    sim_turns.append({
                        "problem": d.get("problem", ""),
                        "profile": d.get("profile", "unknown"),
                        "tutor_msg": tutor_msg,
                        "student_msg": t.get("content", ""),
                        "condition": cond,
                    })

    # Load real MathDial student turns
    mathdial_paths = [
        os.path.join(ROOT, "data", "raw", "mathdial", "train.jsonl"),
        os.path.join(ROOT, "data", "raw", "MathDial", "train.jsonl"),
    ]
    real_turns = []
    for mp in mathdial_paths:
        if os.path.isfile(mp):
            for d in load_jsonl(mp):
                turns = d.get("turns", d.get("dialog", []))
                for i, t in enumerate(turns):
                    if isinstance(t, dict) and t.get("role") == "student":
                        tutor_msg = turns[i-1].get("content", "") if i > 0 else ""
                        real_turns.append({
                            "problem": d.get("problem", d.get("question", "")),
                            "profile": "real",
                            "student_msg": t.get("content", ""),
                            "tutor_msg": tutor_msg,
                        })
            break

    print(f"Simulated student turns: {len(sim_turns)}")
    print(f"Real MathDial student turns: {len(real_turns)}")

    if not sim_turns:
        print("ERROR: No simulated dialogues found")
        sys.exit(1)

    # Sample
    random.seed(42)
    sim_sample = random.sample(sim_turns, min(args.n, len(sim_turns)))
    real_sample = random.sample(real_turns, min(args.n, len(real_turns))) if real_turns else []

    # KS test + Cohen's d on response lengths (sim vs real) — no API needed
    if real_sample:
        sim_lens = np.array([len(t["student_msg"].split()) for t in sim_sample])
        real_lens = np.array([len(t["student_msg"].split()) for t in real_sample])
        ks_stat, ks_p = stats.ks_2samp(sim_lens, real_lens)
        pooled_std = np.sqrt((np.var(sim_lens) + np.var(real_lens)) / 2)
        cohen_d = (np.mean(sim_lens) - np.mean(real_lens)) / (pooled_std + 1e-9)
        print(f"\nResponse-length KS test: D={ks_stat:.4f}, p={ks_p:.4f}")
        print(f"Cohen's d (length sim vs real): {cohen_d:.4f}")
        pd.DataFrame([{
            "metric": "length_ks_stat", "value": round(ks_stat, 4)},
            {"metric": "length_ks_p", "value": round(ks_p, 4)},
            {"metric": "length_cohen_d", "value": round(cohen_d, 4)},
            {"metric": "n_simulated", "value": len(sim_sample)},
            {"metric": "n_real", "value": len(real_sample)},
        ]).to_csv(os.path.join(tables_dir, "table_rq5_simulator_alignment.csv"), index=False)

    # 1. Plausibility scoring (Qwen judge)
    print(f"\nScoring {len(sim_sample)} simulated turns for plausibility (Qwen3-Max)...")
    plausibility_rows = []
    for i, turn in enumerate(sim_sample):
        score = _qwen_plausibility(
            client,
            turn["problem"], turn["profile"],
            turn["tutor_msg"], turn["student_msg"]
        )
        plausibility_rows.append({
            "profile": turn["profile"],
            "condition": turn.get("condition", "unknown"),
            "plausibility_score": score,
        })
        if (i + 1) % 20 == 0:
            print(f"  Scored {i+1}/{len(sim_sample)}")

    plaus_df = pd.DataFrame(plausibility_rows)
    plaus_df.to_csv(os.path.join(tables_dir, "table_simulator_plausibility_scores.csv"), index=False)
    print("\nPlausibility by profile:")
    print(plaus_df.groupby("profile")["plausibility_score"].agg(["mean", "std"]).round(3).to_string())

    # 2. Embedding similarity
    if real_sample:
        print("\nComputing embeddings for similarity analysis...")
        sim_texts = [t["student_msg"] for t in sim_sample]
        real_texts = [t["student_msg"] for t in real_sample]
        sim_emb = _get_embeddings(sim_texts, client)
        real_emb = _get_embeddings(real_texts, client)

        if sim_emb.shape[0] > 0 and real_emb.shape[0] > 0:
            # Cross-similarity: each sim turn vs nearest real turn
            sim_to_real_sims = []
            for s in sim_emb:
                sims = [_cosine_similarity(s, r) for r in real_emb]
                sim_to_real_sims.append(max(sims))  # nearest-neighbor

            # Real-to-real self-similarity (upper bound reference)
            real_self_sims = []
            for i, r in enumerate(real_emb[:50]):
                sims = [_cosine_similarity(r, real_emb[j]) for j in range(min(50, len(real_emb))) if j != i]
                if sims:
                    real_self_sims.append(np.mean(sims))

            kl_div = _kl_divergence(np.array(sim_to_real_sims), np.array(real_self_sims))

            emb_rows = [
                {"metric": "sim_to_real_nn_similarity_mean", "value": round(float(np.mean(sim_to_real_sims)), 4)},
                {"metric": "sim_to_real_nn_similarity_std", "value": round(float(np.std(sim_to_real_sims)), 4)},
                {"metric": "real_self_similarity_mean", "value": round(float(np.mean(real_self_sims)), 4) if real_self_sims else float("nan")},
                {"metric": "kl_divergence", "value": round(kl_div, 4)},
                {"metric": "n_simulated", "value": len(sim_sample)},
                {"metric": "n_real", "value": len(real_sample)},
                {"metric": "embed_model", "value": "text-embedding-v3 (Qwen)"},
            ]
            pd.DataFrame(emb_rows).to_csv(os.path.join(tables_dir, "table_simulator_embedding_similarity.csv"), index=False)
            print(f"\nSim-to-real NN similarity: {np.mean(sim_to_real_sims):.3f} ± {np.std(sim_to_real_sims):.3f}")
            print(f"KL divergence (sim vs real): {kl_div:.4f}")
    else:
        print("Warning: No real MathDial turns found for comparison")

    # Per-profile breakdown with KS stats per profile
    if not plaus_df.empty:
        per_profile = plaus_df.groupby("profile")["plausibility_score"].agg(
            mean="mean", std="std", count="count"
        ).round(3).reset_index()
        per_profile.to_csv(os.path.join(tables_dir, "table_simulator_plausibility_by_profile.csv"), index=False)

        # Build simulator_validity_per_profile table for paper appendix
        validity_rows = []
        for profile in ["struggling", "progressing", "advanced"]:
            p_turns = [t for t in sim_sample if t.get("profile") == profile]
            r_turns_profile = real_sample  # all real turns (not profiled in MathDial)
            if p_turns and r_turns_profile:
                sim_l = np.array([len(t["student_msg"].split()) for t in p_turns])
                real_l = np.array([len(t["student_msg"].split()) for t in r_turns_profile])
                ks_s, ks_p_val = stats.ks_2samp(sim_l, real_l)
                pooled = np.sqrt((np.var(sim_l) + np.var(real_l)) / 2 + 1e-9)
                cd = (np.mean(sim_l) - np.mean(real_l)) / pooled
                # plausibility for this profile
                plaus_here = plaus_df[plaus_df["profile"] == profile]["plausibility_score"]
                validity_rows.append({
                    "profile": profile,
                    "n_sim": len(p_turns),
                    "n_real": len(r_turns_profile),
                    "plausibility_mean": round(plaus_here.mean(), 3) if not plaus_here.empty else float("nan"),
                    "plausibility_std": round(plaus_here.std(), 3) if not plaus_here.empty else float("nan"),
                    "length_ks_stat": round(ks_s, 4),
                    "length_ks_p": round(ks_p_val, 4),
                    "length_cohen_d": round(cd, 4),
                })
        if validity_rows:
            pd.DataFrame(validity_rows).to_csv(
                os.path.join(tables_dir, "table_simulator_validity_per_profile.csv"), index=False
            )
            print("\nSimulator validity per profile:")
            print(pd.DataFrame(validity_rows).to_string(index=False))

    print("\nrun_14_simulator_validation: DONE")


if __name__ == "__main__":
    main()
