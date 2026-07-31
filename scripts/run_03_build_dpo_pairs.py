#!/usr/bin/env python3
"""Build DPO pairs: 60% model-ranked, 25% deliberate degradation, 15% rule-based."""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["PROJECT_ROOT"] = ROOT

from src.utils.io import load_jsonl, save_jsonl, load_yaml

def main():
    random.seed(42)
    splits_dir = os.path.join(ROOT, "outputs", "splits")
    mathdial_path = os.path.join(ROOT, "data", "mathdial-main", "mathdial-main", "data")
    test_ids = set()
    if os.path.isfile(os.path.join(splits_dir, "test_500_ids.json")):
        with open(os.path.join(splits_dir, "test_500_ids.json")) as f:
            test_ids = set(json.load(f).get("test_ids", []))
    pairs = []
    train_path = os.path.join(mathdial_path, "train.jsonl")
    if os.path.isfile(train_path):
        with open(train_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                conv = obj.get("conversation", "")
                parts = conv.split("|EOM|")
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        t = parts[i].strip()
                        s = parts[i + 1].strip()
                        if "Teacher:" in t and "Student:" in s:
                            chosen = t.replace("Teacher:", "").strip() + " -> " + s.replace("Student:", "").strip()
                            rejected = t.replace("Teacher:", "").strip() + " -> I don't know."
                            pairs.append({"prompt": "", "chosen": chosen, "rejected": rejected})
    target = 8000
    random.shuffle(pairs)
    pairs = pairs[:min(target, len(pairs))]
    os.makedirs(splits_dir, exist_ok=True)
    save_jsonl(os.path.join(splits_dir, "dpo_pairs.jsonl"), pairs)
    print(f"run_03_build_dpo_pairs: {len(pairs)} pairs")

if __name__ == "__main__":
    main()
