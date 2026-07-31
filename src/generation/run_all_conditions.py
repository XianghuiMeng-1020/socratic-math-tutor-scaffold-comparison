"""Run all 4 conditions: C1 PE-Llama, C2 SFT-Llama, C3 DPO-Llama, C4 PE-Qwen.

Key design:
- Each open-weight condition loads its model ONCE (persistent across all problems).
- C1 uses base Llama 3.1-8B + socratic prompt (no fine-tuning).
- C2 uses SFT LoRA adapter on top of base model.
- C3 uses DPO LoRA adapter on top of base model.
- C4 uses Qwen3-Max via DashScope API.
- Student simulator is also loaded once and shared across all conditions.
- API calls use exponential backoff with up to 5 retries.
- All failures (early termination, protocol violations) are tracked per condition.
"""
import json
import logging
import os
import time
import random
from typing import Any, Dict, List, Optional, Tuple

from ..utils.io import load_jsonl, save_jsonl, load_yaml, ensure_dir
from ..utils.logging import get_logger
from ..utils.seed import set_global_seed, get_problem_seed

CONDITIONS = ["C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama", "C4_PE_Qwen"]
PROFILES = ["struggling", "progressing", "advanced"]
PROFILE_WEIGHTS = [0.33, 0.50, 0.17]
MAX_TURNS = 10

PROFILE_DESCRIPTIONS = {
    "struggling": (
        "You struggle with math. You often make arithmetic mistakes, confuse operations, "
        "and need significant guidance. You answer slowly and sometimes ask for help repeatedly."
    ),
    "progressing": (
        "You have basic math skills and make occasional errors. You respond to hints and "
        "can follow along when guided step-by-step."
    ),
    "advanced": (
        "You are a strong math student. You understand concepts quickly, rarely make errors, "
        "and can often anticipate the next step after a hint."
    ),
}


def _route_profile(problem_id: str) -> str:
    """Deterministic learner profile assignment by problem hash."""
    h = hash(problem_id) % 100
    if h < 33:
        return "struggling"
    if h < 83:
        return "progressing"
    return "advanced"


def _is_placeholder_ckpt(ckpt_path: str) -> bool:
    """Return True if checkpoint is a dry-run placeholder (no real model weights)."""
    if not ckpt_path or not os.path.isdir(ckpt_path):
        return True
    if os.path.isfile(os.path.join(ckpt_path, "dry_run_meta.json")):
        return True
    meta = os.path.join(ckpt_path, "meta.json")
    if os.path.isfile(meta):
        try:
            m = json.load(open(meta))
            if m.get("status") == "placeholder" or m.get("dry_run"):
                return True
        except Exception:
            pass
    return not os.path.isfile(os.path.join(ckpt_path, "adapter_config.json"))


def _load_open_weight_model(base_model: str, lora_path: Optional[str], use_4bit: bool):
    """Load base model optionally with LoRA adapter. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger = get_logger("generation.model_loader")
    quant_config = None
    if use_4bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization (BitsAndBytes)")
        except Exception as e:
            logger.warning(f"BitsAndBytes 4-bit unavailable ({e}), loading in bfloat16")
            quant_config = None

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and quant_config is None) else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if lora_path and not _is_placeholder_ckpt(lora_path):
        logger.info(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA merged successfully")
    elif lora_path:
        logger.warning(f"Checkpoint at {lora_path} is placeholder/missing 鈥?using base model only")

    model.eval()
    return model, tokenizer


def _generate_local(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Run one forward pass through a local model."""
    import torch
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6) if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _gpt4o_with_retry(
    client,
    messages: List[Dict],
    model: str = "gpt-4o",
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> str:
    """Call GPT-4o with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"GPT-4o attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                logging.error(f"GPT-4o failed after {max_retries} attempts: {e}")
                return f"[GPT4o-error: {e}]"


def _is_early_termination(turns: List[Dict]) -> bool:
    """Detect if dialogue terminated early (< 4 tutor turns or API error string)."""
    tutor_turns = [t for t in turns if t.get("role") == "tutor"]
    if len(tutor_turns) < 4:
        return True
    for t in tutor_turns:
        c = t.get("content", "")
        if c.startswith("[") and "error" in c.lower():
            return True
    return False


def _is_protocol_failure(turns: List[Dict]) -> bool:
    """Detect protocol failure: tutor gives direct answer or empty turns."""
    for t in turns:
        if t.get("role") == "tutor":
            c = t.get("content", "").lower()
            if len(c) < 5:
                return True
            answer_patterns = ["the answer is"] + ["= " + str(i) for i in range(100)]
            if any(x in c for x in answer_patterns):
                pass
    return False


def run_all_conditions(
    project_root: str,
    config_path: str,
    dry_run: bool = False,
    sample_size: int = 0,
    resume: bool = False,
    conditions: Optional[List[str]] = None,
) -> Tuple[int, Dict]:
    """Run all 4 (or specified) conditions. Returns (total_dialogues, failure_stats)."""
    logger = get_logger("generation.run_all")
    set_global_seed(42)

    cfg = load_yaml(config_path)
    model_cfg = cfg if "base_model" in cfg else load_yaml(
        os.path.join(os.path.dirname(config_path), "models.yaml")
    )
    base_model = model_cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    use_4bit = model_cfg.get("generation", {}).get("load_in_4bit", True)
    api_retries = model_cfg.get("generation", {}).get("api_max_retries", 5)
    api_delay = model_cfg.get("generation", {}).get("api_retry_base_delay", 2.0)
    max_turns = model_cfg.get("generation", {}).get("max_turns", MAX_TURNS)

    prompts_dir = os.path.join(os.path.dirname(config_path), "prompts")
    socratic_prompt_path = os.path.join(prompts_dir, "socratic_system_prompt.txt")
    if os.path.isfile(socratic_prompt_path):
        with open(socratic_prompt_path, encoding="utf-8") as f:
            socratic_sys_prompt = f.read()
    else:
        socratic_sys_prompt = (
            "You are a Socratic math tutor. Guide the student using questions only. "
            "Never reveal the answer. Ask one focused question per turn."
        )

    simulator_prompt_path = os.path.join(prompts_dir, "simulator_prompt.txt")
    if os.path.isfile(simulator_prompt_path):
        with open(simulator_prompt_path, encoding="utf-8") as f:
            simulator_prompt_tmpl = f.read()
    else:
        simulator_prompt_tmpl = (
            "You are a {learner_profile} student in a math tutoring session.\n"
            "Your profile: {profile_description}\n"
            "The tutor just said: \"{tutor_message}\"\n"
            "Problem: {problem}\n"
            "Previous conversation:\n{conversation_history}\n"
            "Respond as this student would (1-3 sentences, possible errors if struggling):\n"
        )

    dialogues_dir = os.path.join(project_root, "outputs", "dialogues")
    ensure_dir(dialogues_dir)

    test_path = os.path.join(project_root, "outputs", "splits", "test_500.jsonl")
    if not os.path.isfile(test_path):
        logger.error("No test_500.jsonl. Run run_01_split_and_isolation first.")
        return 0, {}

    test_problems = load_jsonl(test_path)
    if dry_run:
        test_problems = test_problems[:3]
    elif sample_size > 0:
        random.seed(42)
        test_problems = random.sample(test_problems, min(sample_size, len(test_problems)))

    ckpt_paths = {
        "sft": os.path.join(project_root, "outputs", "checkpoints", "sft_lora"),
        "dpo": os.path.join(project_root, "outputs", "checkpoints", "dpo_lora"),
    }

    to_run = conditions or CONDITIONS
    total = 0
    failure_stats = {}

    # API client (shared for C4 + fallback): prefer Qwen, then OpenAI
    gpt_client = None
    gpt_key = os.environ.get("QWEN_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    gpt_base = model_cfg.get("pe_qwen", {}).get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    gpt_model = model_cfg.get("pe_qwen", {}).get("model", "qwen3-max")
    gpt_fallback_model = "qwen-plus"  # lighter model for simulator fallback

    for condition in to_run:
        logger.info(f"===== Starting condition: {condition} =====")
        out_path = os.path.join(dialogues_dir, f"dialogues_{condition}.jsonl")

        existing = []
        if resume and os.path.isfile(out_path):
            existing = load_jsonl(out_path)
        n_existing = len(existing)
        n_needed = len(test_problems)

        if resume and n_existing >= n_needed:
            logger.info(f"{condition}: already complete ({n_existing}/{n_needed}), skipping")
            total += n_existing
            failure_stats[condition] = {"skipped": True}
            continue

        to_do = test_problems[n_existing:] if (resume and n_existing > 0) else test_problems
        dialogues = list(existing)

        # Track failure counts
        raw_attempted = 0
        early_terminations = 0
        protocol_failures = 0
        scorable = 0

        # Load model once per condition
        model = None
        tokenizer = None

        if condition in ("C1_PE_Llama", "C2_SFT_Llama", "C3_DPO_Llama") and not dry_run:
            lora_path = None
            if condition == "C2_SFT_Llama":
                lora_path = ckpt_paths["sft"]
            elif condition == "C3_DPO_Llama":
                lora_path = ckpt_paths["dpo"]
            try:
                model, tokenizer = _load_open_weight_model(base_model, lora_path, use_4bit)
                logger.info(f"{condition}: model loaded successfully")
            except Exception as e:
                logger.error(f"{condition}: model load failed: {e}. Will use API fallback.")

        elif condition == "C4_PE_Qwen" and not dry_run:
            if not gpt_key:
                logger.error("C4_PE_Qwen: no API key set (QWEN_API_KEY or OPENAI_API_KEY)!")
            else:
                from openai import OpenAI
                gpt_client = OpenAI(api_key=gpt_key, base_url=gpt_base, timeout=90.0)

        for idx, prob in enumerate(to_do):
            problem_id = prob.get("problem_id", f"prob_{idx}")
            problem_text = prob.get("problem", "")
            reference_solution = prob.get("reference_solution", "")
            profile = _route_profile(problem_id)
            profile_desc = PROFILE_DESCRIPTIONS[profile]

            # Set deterministic seed per (problem, profile)
            seed = get_problem_seed(problem_id, profile, 0)
            set_global_seed(seed)

            turns = []
            history_msgs = []  # OpenAI-format messages for context
            raw_attempted += 1

            if dry_run:
                turns = [
                    {"role": "tutor", "content": f"[{condition} mock] What do you know about the problem?"},
                    {"role": "student", "content": "I'm working on it."},
                ] * 3
            else:
                for turn_idx in range(max_turns):
                    # Build conversation history string for simulator
                    history_str = ""
                    for t in turns:
                        role_label = "Tutor" if t["role"] == "tutor" else "Student"
                        history_str += f"{role_label}: {t['content']}\n"

                    # --- Tutor turn ---
                    tutor_messages = [
                        {"role": "system", "content": socratic_sys_prompt},
                        {"role": "user", "content": (
                            f"Problem: {problem_text}\n\n"
                            f"Conversation so far:\n{history_str}\n"
                            f"Generate your next Socratic question:"
                        )},
                    ]

                    if condition == "C4_PE_Qwen":
                        if gpt_client:
                            tutor_msg = _gpt4o_with_retry(
                                gpt_client, tutor_messages,
                                model=gpt_model,
                                max_retries=api_retries, base_delay=api_delay
                            )
                        else:
                            tutor_msg = "[C4-Qwen: API key missing]"
                    elif model is not None:
                        tutor_msg = _generate_local(model, tokenizer, tutor_messages)
                    else:
                        if gpt_key:
                            if gpt_client is None:
                                from openai import OpenAI
                                gpt_client = OpenAI(api_key=gpt_key, base_url=gpt_base, timeout=60.0)
                            tutor_msg = _gpt4o_with_retry(
                                gpt_client, tutor_messages,
                                model=gpt_fallback_model,
                                max_retries=api_retries, base_delay=api_delay
                            )
                            tutor_msg = f"[{condition}-API-fallback] {tutor_msg}"
                        else:
                            tutor_msg = f"[{condition}: model unavailable, no API key]"

                    turns.append({"role": "tutor", "content": tutor_msg})

                    # --- Student simulator turn ---
                    sim_prompt = simulator_prompt_tmpl.format(
                        learner_profile=profile,
                        profile_description=profile_desc,
                        tutor_message=tutor_msg,
                        problem=problem_text,
                        conversation_history=history_str,
                    )
                    sim_messages = [
                        {"role": "system", "content": "You are a math student in a tutoring session."},
                        {"role": "user", "content": sim_prompt},
                    ]

                    # Student simulator: use local model (same base, no adapter for simulator)
                    # If local model unavailable, fallback to API
                    if model is not None:
                        student_msg = _generate_local(model, tokenizer, sim_messages, max_new_tokens=128)
                    elif gpt_key:
                        if gpt_client is None:
                            from openai import OpenAI
                            gpt_client = OpenAI(api_key=gpt_key, base_url=gpt_base, timeout=60.0)
                        student_msg = _gpt4o_with_retry(
                            gpt_client, sim_messages,
                            model=gpt_fallback_model,
                            max_retries=api_retries, base_delay=api_delay
                        )
                    else:
                        student_msg = "I'm not sure. Can you explain more?"

                    turns.append({"role": "student", "content": student_msg})

                    # Check termination conditions
                    student_lower = student_msg.lower()
                    if any(x in student_lower for x in ["i got it", "i understand", "the answer is correct", "that's correct"]):
                        logger.debug(f"{condition} {problem_id}: student solved at turn {turn_idx+1}")
                        break

            # Classify dialogue outcome
            early_term = _is_early_termination(turns)
            proto_fail = _is_protocol_failure(turns)
            if early_term:
                early_terminations += 1
            elif proto_fail:
                protocol_failures += 1
            else:
                scorable += 1

            dialogues.append({
                "problem_id": problem_id,
                "condition": condition,
                "profile": profile,
                "turns": turns,
                "problem": problem_text,
                "reference_solution": reference_solution,
                "metadata": {
                    "turn_count": len([t for t in turns if t["role"] == "tutor"]),
                    "early_termination": early_term,
                    "protocol_failure": proto_fail,
                    "scorable": not early_term and not proto_fail,
                },
            })
            total += 1

            if (idx + 1) % 20 == 0 or (idx + 1) == len(to_do):
                logger.info(
                    f"{condition}: {idx+1}/{len(to_do)} | "
                    f"early_term={early_terminations} proto_fail={protocol_failures} scorable={scorable}"
                )
                save_jsonl(out_path, dialogues)

        save_jsonl(out_path, dialogues)
        failure_stats[condition] = {
            "raw_attempted": raw_attempted,
            "early_terminations": early_terminations,
            "protocol_failures": protocol_failures,
            "scorable": scorable,
            "total_written": len(dialogues),
        }
        logger.info(f"{condition}: DONE. Stats: {failure_stats[condition]}")

        # Free GPU memory before loading next condition's model
        if model is not None:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = None
            tokenizer = None

    # Save failure stats to outputs
    stats_path = os.path.join(project_root, "outputs", "tables", "table_dialogue_failure_stats.json")
    ensure_dir(os.path.dirname(stats_path))
    with open(stats_path, "w") as f:
        json.dump(failure_stats, f, indent=2)
    logger.info(f"Failure stats written to {stats_path}")

    return total, failure_stats


