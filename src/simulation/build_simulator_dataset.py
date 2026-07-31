"""Build simulator training dataset from MathDial student turns (held-out from test 500)."""
import json
import os
from typing import Any, Dict, List

from ..utils.io import load_jsonl, save_jsonl, ensure_dir
from ..utils.logging import get_logger


LEARNER_PROFILES = {
    "struggling": "Struggling: difficulty with concepts, frequent errors, needs more scaffolding.",
    "progressing": "Progressing: moderate understanding, occasional errors, learns with guidance.",
    "advanced": "Advanced: strong understanding, occasional slips, benefits from subtle hints.",
}


def _parse_mathdial_conversation(conv_str: str) -> List[Dict[str, str]]:
    """Parse MathDial conversation format (Teacher: ...|EOM|Student: ...)."""
    turns = []
    parts = conv_str.split("|EOM|")
    for p in parts:
        p = p.strip()
        if p.startswith("Teacher:"):
            turns.append({"role": "user", "content": p.replace("Teacher:", "").strip()})
        elif p.startswith("Student:"):
            turns.append({"role": "assistant", "content": p.replace("Student:", "").strip()})
    return turns


def build_simulator_dataset(
    project_root: str,
    mathdial_path: str,
    test_ids: set,
    output_path: str,
) -> int:
    """Build simulator training data from MathDial train, excluding test."""
    logger = get_logger("simulation.build")
    ensure_dir(os.path.dirname(output_path))
    train_path = os.path.join(mathdial_path, "train.jsonl")
    if not os.path.isfile(train_path):
        return 0
    examples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("qid", ""))
            if qid in test_ids:
                continue
            conv = obj.get("conversation", "")
            turns = _parse_mathdial_conversation(conv)
            profile = "progressing"  # default
            if "difficulty" in obj.get("student_profile", "").lower():
                profile = "struggling"
            elif "7th grade" in obj.get("student_profile", ""):
                profile = "progressing"
            for i, t in enumerate(turns):
                if t["role"] == "assistant":
                    prev = turns[:i]
                    ctx = "\n".join([f"{x['role']}: {x['content']}" for x in prev[-4:]])
                    examples.append({
                        "problem_id": qid,
                        "learner_profile": profile,
                        "tutor_message": prev[-1]["content"] if prev else "",
                        "conversation_history": ctx,
                        "student_response": t["content"],
                    })
    save_jsonl(output_path, examples)
    logger.info(f"Simulator dataset: {len(examples)} examples -> {output_path}")
    return len(examples)
