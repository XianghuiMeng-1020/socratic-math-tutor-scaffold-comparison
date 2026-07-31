"""Build reproducibility manifests."""
import os
from datetime import datetime
from typing import Dict

from ..utils.io import save_json, ensure_dir, compute_hash


def build_repro_manifest(project_root: str) -> Dict:
    """Build repro_manifest.json."""
    manifest = {
        "study": "Socratic AI Tutor Four-Condition Comparison",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "project_root": project_root,
            "conditions": ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_GPT4o"],
            "test_size": 500,
            "max_turns": 10,
        },
        "packages": {},
        "model_ids": {
            "base": "meta-llama/Llama-3.1-8B-Instruct",
            "gpt4o": "gpt-4o",
        },
        "prompt_hashes": {},
        "seed_policy": "fixed_per_problem_profile_turn",
    }
    try:
        import transformers
        manifest["packages"]["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import torch
        manifest["packages"]["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import openai
        manifest["packages"]["openai"] = openai.__version__
    except Exception:
        pass
    prompt_dir = os.path.join(project_root, "configs", "prompts")
    for f in os.listdir(prompt_dir) if os.path.isdir(prompt_dir) else []:
        if f.endswith(".txt"):
            p = os.path.join(prompt_dir, f)
            with open(p, "r") as fh:
                manifest["prompt_hashes"][f] = compute_hash(fh.read())
    out_dir = os.path.join(project_root, "outputs", "manifests")
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, "repro_manifest.json"), manifest)
    return manifest
