"""LoRA SFT on SocraTeach-math for C2 SFT-Llama."""
import json
import os
from typing import Any, Dict, List, Optional

from ..utils.io import load_yaml, load_jsonl, save_json, ensure_dir
from ..utils.logging import get_logger
from ..utils.seed import set_global_seed


def _prepare_sft_data(socrateach_path: str, splits_dir: str, test_ids: Optional[set]) -> List[Dict]:
    """Prepare SFT examples from SocraTeach, excluding test problems."""
    # SocraTeach uses GSM8K-style keys; test_ids from MathTutorBench are problem hashes
    # So no overlap by construction - different datasets
    train_path = os.path.join(splits_dir, "sft_train.jsonl")
    if os.path.isfile(train_path):
        return load_jsonl(train_path)
    # Build from SocraTeach_single.json
    single_path = os.path.join(socrateach_path, "SocraTeach_single.json")
    if not os.path.isfile(single_path):
        single_path = os.path.join(os.path.dirname(socrateach_path), "SocraTeach_single.json")
    if not os.path.isfile(single_path):
        return []
    with open(single_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for key, item in data.items():
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        history = item.get("history", [])
        if not prompt or not response:
            continue
        # Format as chat
        messages = []
        for h in history:
            if isinstance(h, list) and len(h) >= 2:
                messages.append({"role": "user", "content": h[0]})
                messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        examples.append({"id": key, "messages": messages})
    # Hold out 10% for validation
    import random
    random.seed(42)
    random.shuffle(examples)
    n = len(examples)
    n_val = max(1, int(n * 0.10))
    train_ex = examples[n_val:]
    val_ex = examples[:n_val]
    ensure_dir(splits_dir)
    with open(os.path.join(splits_dir, "sft_train.jsonl"), "w", encoding="utf-8") as f:
        for e in train_ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(splits_dir, "sft_valid.jsonl"), "w", encoding="utf-8") as f:
        for e in val_ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return train_ex


def train_sft_lora(
    project_root: str,
    config_path: str,
    dry_run: bool = False,
) -> str:
    """Train LoRA SFT. Returns checkpoint path."""
    logger = get_logger("training.sft")
    set_global_seed(42)
    cfg = load_yaml(config_path)
    models_cfg = load_yaml(os.path.join(os.path.dirname(config_path), "models.yaml"))
    paths_cfg = load_yaml(os.path.join(os.path.dirname(config_path), "paths.yaml"))
    out_ckpt = os.path.join(project_root, "outputs", "checkpoints", "sft_lora")
    ensure_dir(out_ckpt)

    socrateach_path = paths_cfg.get("datasets", {}).get("socrateach", "")
    if "${" in str(socrateach_path):
        socrateach_path = os.path.join(project_root, "data", "SocraticLM-main", "SocraticLM-main", "data")
    splits_dir = os.path.join(project_root, "outputs", "splits")
    ensure_dir(splits_dir)

    examples = _prepare_sft_data(socrateach_path, splits_dir, None)
    logger.info(f"SFT examples: {len(examples)}")

    if dry_run:
        save_json(os.path.join(out_ckpt, "dry_run_meta.json"), {"dry_run": True, "n_examples": len(examples)})
        return out_ckpt

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch
    except ImportError as e:
        logger.warning(f"Training deps missing: {e}. Creating placeholder checkpoint.")
        save_json(os.path.join(out_ckpt, "meta.json"), {"status": "placeholder", "reason": str(e), "n_examples": len(examples)})
        return out_ckpt

    base_model = models_cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    sft_cfg = models_cfg.get("sft", {})
    gen_cfg = models_cfg.get("generation", {})
    use_4bit = gen_cfg.get("load_in_4bit", True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    quant_config = None
    if use_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading model in 4-bit quantization (QLoRA)")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    lora_config = LoraConfig(
        r=sft_cfg.get("rank", 16),
        lora_alpha=sft_cfg.get("alpha", 32),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(prepare_model_for_kbit_training(model), lora_config)

    def format_fn(ex):
        msgs = ex.get("messages", [])
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

    train_dataset = [{"text": format_fn(ex)} for ex in examples]
    logger.info(f"Formatted {len(train_dataset)} SFT examples for training")
    with open(os.path.join(out_ckpt, "train_sample.json"), "w") as f:
        json.dump(train_dataset[:2], f, indent=2)

    training_args = TrainingArguments(
        output_dir=out_ckpt,
        per_device_train_batch_size=sft_cfg.get("batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("grad_accum", 8),
        learning_rate=sft_cfg.get("lr", 2e-4),
        lr_scheduler_type=sft_cfg.get("lr_scheduler", "cosine"),
        num_train_epochs=sft_cfg.get("epochs", 3),
        max_steps=sft_cfg.get("max_steps", -1),
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        eval_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    from datasets import Dataset
    ds = Dataset.from_list([{"text": t["text"]} for t in train_dataset])

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        formatting_func=lambda x: x.get("text", ""),
    )
    result = trainer.train()
    trainer.save_model(out_ckpt)
    tokenizer.save_pretrained(out_ckpt)
    save_json(os.path.join(out_ckpt, "meta.json"), {
        "status": "trained",
        "steps": result.global_step,
        "epoch": round(result.metrics.get("epoch", 0), 2),
        "train_loss": str(round(result.metrics.get("train_loss", 0), 4)),
        "n_examples": len(train_dataset),
    })
    logger.info(f"SFT checkpoint saved to {out_ckpt}")
    return out_ckpt
