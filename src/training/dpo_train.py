"""DPO training for C3 DPO-Llama, initialized from SFT checkpoint."""
import json
import os
from typing import Any, Dict, List, Optional

from ..utils.io import load_yaml, load_jsonl, save_json, ensure_dir
from ..utils.logging import get_logger
from ..utils.seed import set_global_seed


def _load_dpo_pairs(pairs_path: str) -> List[Dict]:
    """Load DPO pairs."""
    if not os.path.isfile(pairs_path):
        return []
    return load_jsonl(pairs_path)


def train_dpo(
    project_root: str,
    config_path: str,
    sft_checkpoint: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """Train DPO from SFT checkpoint. Returns checkpoint path."""
    logger = get_logger("training.dpo")
    set_global_seed(42)
    cfg = load_yaml(config_path)
    models_cfg = load_yaml(os.path.join(os.path.dirname(config_path), "models.yaml"))
    paths_cfg = load_yaml(os.path.join(os.path.dirname(config_path), "paths.yaml"))
    pairs_path = os.path.join(project_root, "outputs", "splits", "dpo_pairs.jsonl")
    out_ckpt = os.path.join(project_root, "outputs", "checkpoints", "dpo_lora")
    ensure_dir(out_ckpt)

    if sft_checkpoint is None:
        sft_checkpoint = os.path.join(project_root, "outputs", "checkpoints", "sft_lora")

    pairs = _load_dpo_pairs(pairs_path)
    if not pairs:
        logger.warning("No DPO pairs found. Creating placeholder.")
        save_json(os.path.join(out_ckpt, "meta.json"), {"status": "placeholder", "reason": "no_pairs"})
        return out_ckpt

    if dry_run:
        save_json(os.path.join(out_ckpt, "dry_run_meta.json"), {"dry_run": True, "n_pairs": len(pairs)})
        return out_ckpt

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from peft import PeftModel
        import torch
    except ImportError as e:
        logger.warning(f"DPO deps missing: {e}. Creating placeholder.")
        save_json(os.path.join(out_ckpt, "meta.json"), {"status": "placeholder", "reason": str(e)})
        return out_ckpt

    base_model = models_cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    dpo_cfg = models_cfg.get("dpo", {})
    gen_cfg = models_cfg.get("generation", {})
    use_4bit = gen_cfg.get("load_in_4bit", True)

    tokenizer_path = sft_checkpoint if os.path.isfile(os.path.join(sft_checkpoint, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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
        logger.info("DPO: Loading model in 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if os.path.isdir(sft_checkpoint) and os.path.isfile(os.path.join(sft_checkpoint, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def format_prompt_pair(p):
        chosen = p.get("chosen", "")
        rejected = p.get("rejected", "")
        prompt = p.get("prompt", "") or " "
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    formatted = [format_prompt_pair(p) for p in pairs]
    logger.info(f"DPO: using {len(formatted)} pairs for training")
    from datasets import Dataset
    ds = Dataset.from_list(formatted)

    training_args = DPOConfig(
        output_dir=out_ckpt,
        per_device_train_batch_size=dpo_cfg.get("batch_size", 1),
        gradient_accumulation_steps=dpo_cfg.get("grad_accum", 16),
        learning_rate=dpo_cfg.get("lr", 5e-5),
        num_train_epochs=dpo_cfg.get("epochs", 1),
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        beta=dpo_cfg.get("beta", 0.1),
        max_length=1024,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(out_ckpt)
    tokenizer.save_pretrained(out_ckpt)
    save_json(os.path.join(out_ckpt, "meta.json"), {
        "status": "trained",
        "global_step": result.global_step,
        "epoch": round(result.metrics.get("epoch", 0), 2),
        "n_pairs": len(formatted),
    })
    logger.info(f"DPO checkpoint saved to {out_ckpt}")
    return out_ckpt
