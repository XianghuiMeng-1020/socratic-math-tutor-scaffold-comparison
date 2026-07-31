#!/usr/bin/env python3
"""Phase 2A: Scaffold strength ablation experiment.

Generates dialogues under 3 scaffold levels using the base Llama 3.1-8B model:
  C0a: NoScaffold   - minimal system prompt ("You are a math tutor. Help the student.")
  C0b: WeakScaffold - partial guidance ("Ask questions to guide the student")
  C1:  FullScaffold - complete Socratic prompt (same as main experiment C1 condition)

Evaluates 6-dimensional metrics and plots a scaffold-strength curve.
Designed to empirically validate the Scaffold-Ceiling hypothesis (Section 4.4).

Usage:
    python scripts/run_12_scaffold_ablation.py [--sample N] [--dry-run]
    N: number of problems from test_500.jsonl to use (default 50)
"""
import argparse
import json
import os
import sys
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

import pandas as pd
import numpy as np

from src.utils.io import load_jsonl, save_jsonl, load_yaml, ensure_dir
from src.evaluation.composite_score import compute_composite, compute_per_dimension
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("scaffold_ablation")
METRICS = ["QQ", "SD", "SLR", "EDT", "DC", "MC_Verified"]

SCAFFOLD_CONDITIONS = {
    "C0a_NoScaffold": "configs/prompts/no_scaffold_prompt.txt",
    "C0b_WeakScaffold": "configs/prompts/weak_scaffold_prompt.txt",
    "C1_FullScaffold": "configs/prompts/socratic_system_prompt.txt",
}

PROFILE_DESCRIPTIONS = {
    "struggling": "You struggle with math. You often make errors and need guidance.",
    "progressing": "You have basic math skills and can follow hints.",
    "advanced": "You are a strong math student who responds quickly to hints.",
}


def _route_profile(problem_id: str) -> str:
    h = hash(problem_id) % 100
    if h < 33:
        return "struggling"
    if h < 83:
        return "progressing"
    return "advanced"


def _load_prompt(path: str) -> str:
    full = os.path.join(ROOT, path)
    if os.path.isfile(full):
        with open(full, encoding="utf-8") as f:
            return f.read().strip()
    return "You are a math tutor."


def _generate_ablation_dialogue(
    model, tokenizer, sys_prompt: str, problem: str, profile: str,
    max_turns: int = 8, max_tokens: int = 200, dry_run: bool = False
) -> list:
    """Generate a dialogue under one scaffold condition."""
    import torch
    turns = []
    profile_desc = PROFILE_DESCRIPTIONS[profile]
    history_str = ""

    for turn_idx in range(max_turns):
        if dry_run:
            tutor_msg = f"[NoScaffold mock] What do you know about the problem?"
            student_msg = "I'm not sure."
            turns.extend([
                {"role": "tutor", "content": tutor_msg},
                {"role": "student", "content": student_msg},
            ])
            continue

        tutor_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Problem: {problem}\n\nConversation so far:\n{history_str}\nYour response:"},
        ]
        text = tokenizer.apply_chat_template(tutor_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        tutor_msg = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        turns.append({"role": "tutor", "content": tutor_msg})
        history_str += f"Tutor: {tutor_msg}\n"

        # Student simulator
        sim_prompt = (
            f"You are a {profile} math student. {profile_desc}\n"
            f"Problem: {problem}\nTutor said: {tutor_msg}\nConversation: {history_str}\n"
            f"Your response (1-2 sentences):"
        )
        sim_msgs = [{"role": "system", "content": "You are a math student."}, {"role": "user", "content": sim_prompt}]
        sim_text = tokenizer.apply_chat_template(sim_msgs, tokenize=False, add_generation_prompt=True)
        sim_inputs = tokenizer(sim_text, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            sim_inputs = {k: v.cuda() for k, v in sim_inputs.items()}
        with torch.no_grad():
            sim_out = model.generate(**sim_inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        student_msg = tokenizer.decode(sim_out[0][sim_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        turns.append({"role": "student", "content": student_msg})
        history_str += f"Student: {student_msg}\n"

        if any(x in student_msg.lower() for x in ["i got it", "i understand", "correct"]):
            break
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_global_seed(42)

    dialogues_dir = os.path.join(ROOT, "outputs", "dialogues")
    tables_dir = os.path.join(ROOT, "outputs", "tables")
    ensure_dir(dialogues_dir)
    ensure_dir(tables_dir)

    test_path = os.path.join(ROOT, "outputs", "splits", "test_500.jsonl")
    if not os.path.isfile(test_path):
        logger.error("test_500.jsonl not found. Run run_01_split_and_isolation.py first.")
        sys.exit(1)

    problems = load_jsonl(test_path)
    random.seed(42)
    sample_probs = random.sample(problems, min(args.sample, len(problems)))
    logger.info(f"Scaffold ablation: {len(sample_probs)} problems, 3 scaffold levels")

    weights_cfg = load_yaml(os.path.join(ROOT, "configs", "eval_weights.yaml"))
    w = weights_cfg.get("metrics", {})
    slr_invert = weights_cfg.get("slr_invert", True)

    # Load model once
    model_cfg = load_yaml(os.path.join(ROOT, "configs", "models.yaml"))
    base_model = model_cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    use_4bit = model_cfg.get("generation", {}).get("load_in_4bit", True)

    model = tokenizer = None
    if not args.dry_run:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            logger.info(f"Loading {base_model}...")
            quant_cfg = None
            if use_4bit and torch.cuda.is_available():
                try:
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    )
                except Exception:
                    quant_cfg = None
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=quant_cfg,
                torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and quant_cfg is None) else None,
                device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True,
            )
            model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            logger.error(
                "FATAL: Cannot run scaffold ablation without a loaded model. "
                "Dry-run fallback is DISABLED to prevent invalid homogeneous data. "
                "Ensure GPU + model weights are available, then re-run."
            )
            sys.exit(1)

    all_rows = []

    for cond_name, prompt_path in SCAFFOLD_CONDITIONS.items():
        sys_prompt = _load_prompt(prompt_path)
        logger.info(f"Running scaffold condition: {cond_name}")
        out_path = os.path.join(dialogues_dir, f"dialogues_{cond_name}.jsonl")
        dialogues = []

        for prob in sample_probs:
            pid = prob.get("problem_id", "unknown")
            problem_text = prob.get("problem", "")
            profile = _route_profile(pid)
            turns = _generate_ablation_dialogue(
                model, tokenizer, sys_prompt, problem_text, profile,
                dry_run=args.dry_run
            )
            d = {
                "problem_id": pid, "condition": cond_name, "profile": profile,
                "scaffold_level": cond_name.split("_")[0],
                "turns": turns, "problem": problem_text,
            }
            dialogues.append(d)
            dims = compute_per_dimension(d, slr_invert=slr_invert)
            comp = compute_composite(d, weights=w, slr_invert=slr_invert)
            row = {"condition": cond_name, "problem_id": pid, "profile": profile, "composite": comp}
            row.update(dims)
            all_rows.append(row)

        save_jsonl(out_path, dialogues)
        logger.info(f"{cond_name}: {len(dialogues)} dialogues -> {out_path}")

    # Summary table
    df = pd.DataFrame(all_rows)
    agg = df.groupby("condition")[METRICS + ["composite"]].agg(["mean", "std"]).round(3)
    print("\nScaffold Ablation Results:")
    print(agg.to_string())
    df.to_csv(os.path.join(tables_dir, "table_scaffold_ablation_raw.csv"), index=False)
    agg.to_csv(os.path.join(tables_dir, "table_scaffold_ablation_summary.csv"))

    logger.info("run_12_scaffold_ablation: DONE")


if __name__ == "__main__":
    main()
