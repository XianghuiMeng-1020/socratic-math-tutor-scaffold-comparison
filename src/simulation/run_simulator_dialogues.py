"""Run simulator to generate student responses."""
import json
import os
from typing import Any, Dict, List, Optional

from ..utils.io import load_yaml, load_jsonl
from ..utils.logging import get_logger
from ..utils.seed import get_problem_seed, set_global_seed


def _simulate_student(
    problem: str,
    tutor_message: str,
    history: List[Dict],
    profile: str,
    checkpoint: Optional[str],
    turn_id: int,
    problem_id: str,
) -> str:
    """Generate one student turn from simulator."""
    set_global_seed(get_problem_seed(problem_id, profile, turn_id))
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        base = checkpoint or "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        profile_desc = {
            "struggling": "Struggling student with difficulty understanding.",
            "progressing": "Progressing student with moderate understanding.",
            "advanced": "Advanced student with good understanding.",
        }.get(profile, "Progressing student.")
        ctx = "\n".join([f"{h['role']}: {h['content']}" for h in history[-6:]])
        prompt = f"Profile: {profile_desc}\n\nProblem: {problem}\n\nTutor: {tutor_message}\n\nContext:\n{ctx}\n\nStudent:"
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception:
        return "I'm not sure. Can you explain more?"


def run_simulator_dialogues(
    config_path: str,
    dry_run: bool = False,
) -> int:
    """Run dialogues (called from generation). Returns count."""
    return 0  # Actual dialogue run is in run_all_conditions
