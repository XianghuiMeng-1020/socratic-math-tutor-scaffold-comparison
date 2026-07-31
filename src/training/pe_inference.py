"""Prompt-only inference for C1 PE-Llama and C4 PE-GPT4o.

Design principles:
- pe_llama_infer: loads Llama 3.1-8B-Instruct locally. Raises RuntimeError if model
  unavailable; does NOT silently fall back to a different model.
- pe_gpt4o_infer: calls GPT-4o via API with exponential-backoff retry.
- api_infer_with_retry: generic API helper for any OpenAI-compatible model.
"""
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from ..utils.io import load_yaml, ensure_dir
from ..utils.logging import get_logger
from ..utils.seed import set_global_seed


def _load_socratic_prompt(prompts_dir: str) -> str:
    """Load Socratic system prompt from disk."""
    p = os.path.join(prompts_dir, "socratic_system_prompt.txt")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return (
        "You are a Socratic math tutor. Guide the student using questions only. "
        "Never reveal the answer directly. Ask one focused, scaffolded question per turn."
    )


def api_infer_with_retry(
    client,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> str:
    """Call any OpenAI-compatible API with exponential-backoff retry. Raises on final failure."""
    logger = get_logger("pe_inference.api")
    last_exc = None
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
            last_exc = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)
    raise RuntimeError(f"API call failed after {max_retries} attempts") from last_exc


def pe_llama_infer(
    problem: str,
    history: List[Dict[str, str]],
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    use_4bit: bool = True,
) -> str:
    """Single-turn PE inference with local Llama 3.1-8B-Instruct.

    Raises RuntimeError if the model cannot be loaded. Does NOT silently fall back
    to any other model or API.
    """
    set_global_seed(42)
    prompts_dir = (
        os.path.join(os.path.dirname(config_path), "prompts") if config_path else "configs/prompts"
    )
    sys_prompt = _load_socratic_prompt(prompts_dir)
    history_str = ""
    for h in history:
        role = "Tutor" if h.get("role") in ("assistant", "tutor") else "Student"
        history_str += f"{role}: {h.get('content', '')}\n"
    user_content = (
        f"Problem: {problem}\n\nConversation so far:\n{history_str}\nGenerate your next Socratic question:"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
    base = model_path or "meta-llama/Llama-3.1-8B-Instruct"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        quant_config = None
        if use_4bit and torch.cuda.is_available():
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                quant_config = None
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and quant_config is None) else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.eval()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6) if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        raise RuntimeError(f"pe_llama_infer: failed to load/run {base}: {e}") from e


def pe_gpt4o_infer(
    problem: str,
    history: List[Dict[str, str]],
    api_key: Optional[str] = None,
    config_path: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    cache_path: Optional[str] = None,
    model: str = "qwen3-max",
    max_retries: int = 5,
    base_delay: float = 2.0,
    api_base: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> str:
    """Single-turn PE inference via API with retry and optional caching.

    Supports any OpenAI-compatible endpoint (Qwen DashScope, OpenAI, etc.).
    """
    set_global_seed(42)
    prompts_dir = (
        os.path.join(os.path.dirname(config_path), "prompts") if config_path else "configs/prompts"
    )
    sys_prompt = _load_socratic_prompt(prompts_dir)
    history_str = ""
    for h in history:
        role = "Tutor" if h.get("role") in ("assistant", "tutor") else "Student"
        history_str += f"{role}: {h.get('content', '')}\n"
    user_content = (
        f"Problem: {problem}\n\nConversation so far:\n{history_str}\nGenerate your next Socratic question:"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    cache_key = str(hash((model, problem, str(history))))
    if cache_path:
        ensure_dir(cache_path)
        safe_model = model.replace("/", "_")
        cache_file = os.path.join(cache_path, f"{safe_model}_cache.json")
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
                if cache_key in cache:
                    return cache[cache_key]
            except Exception:
                cache = {}
        else:
            cache = {}

    from openai import OpenAI
    key_env = api_key_env or "QWEN_API_KEY"
    key = api_key or os.environ.get(key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(f"pe_gpt4o_infer: neither {key_env} nor OPENAI_API_KEY is set")
    base = api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(api_key=key, base_url=base, timeout=90.0)
    result = api_infer_with_retry(
        client, messages, model=model,
        max_tokens=max_tokens, temperature=temperature,
        max_retries=max_retries, base_delay=base_delay,
    )
    if cache_path:
        safe_model = model.replace("/", "_")
        cache[cache_key] = result
        cache_file = os.path.join(cache_path, f"{safe_model}_cache.json")
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    return result
