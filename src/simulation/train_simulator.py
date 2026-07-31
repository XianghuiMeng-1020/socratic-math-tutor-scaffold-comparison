"""Train student simulator from MathDial student turns."""
import os
from typing import Optional

from ..utils.io import load_yaml, load_jsonl, save_json, ensure_dir
from ..utils.logging import get_logger
from ..utils.seed import set_global_seed


def train_simulator(
    project_root: str,
    config_path: str,
    dry_run: bool = False,
) -> str:
    """Train frozen student simulator. Returns checkpoint path."""
    logger = get_logger("simulation.train")
    set_global_seed(42)
    cfg = load_yaml(config_path)
    models_cfg = load_yaml(os.path.join(os.path.dirname(config_path), "models.yaml"))
    out_ckpt = os.path.join(project_root, "outputs", "checkpoints", "simulator")
    ensure_dir(out_ckpt)
    sim_path = os.path.join(project_root, "outputs", "splits", "simulator_train.jsonl")
    if not os.path.isfile(sim_path):
        logger.warning("No simulator dataset. Creating placeholder.")
        save_json(os.path.join(out_ckpt, "meta.json"), {"status": "placeholder", "reason": "no_dataset"})
        return out_ckpt
    examples = load_jsonl(sim_path)
    if not examples:
        save_json(os.path.join(out_ckpt, "meta.json"), {"status": "placeholder", "reason": "empty"})
        return out_ckpt
    if dry_run:
        save_json(os.path.join(out_ckpt, "dry_run_meta.json"), {"dry_run": True, "n_examples": len(examples)})
        return out_ckpt
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch
        from datasets import Dataset
    except ImportError as e:
        logger.warning(f"Simulator deps missing: {e}. Creating placeholder.")
        save_json(os.path.join(out_ckpt, "meta.json"), {"status": "placeholder", "reason": str(e)})
        return out_ckpt
    base = models_cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    gen_cfg = models_cfg.get("generation", {})
    use_4bit = gen_cfg.get("load_in_4bit", True)

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
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
        logger.info("Simulator: Loading model in 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(prepare_model_for_kbit_training(model), lora)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def fmt(ex):
        prompt = f"Profile: {ex['learner_profile']}\nTutor: {ex['tutor_message']}\n\nContext:\n{ex['conversation_history']}\n\nStudent:"
        return {"text": prompt + " " + ex["student_response"]}
    ds = Dataset.from_list([fmt(e) for e in examples])
    logger.info(f"Simulator: training on {len(ds)} examples")
    args = TrainingArguments(
        output_dir=out_ckpt,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        formatting_func=lambda x: x.get("text", ""),
    )
    result = trainer.train()
    trainer.save_model(out_ckpt)
    tokenizer.save_pretrained(out_ckpt)
    save_json(os.path.join(out_ckpt, "meta.json"), {
        "status": "trained",
        "global_step": result.global_step,
        "epoch": round(result.metrics.get("epoch", 0), 2),
        "n_examples": len(ds),
    })
    logger.info(f"Simulator checkpoint: {out_ckpt}")
    return out_ckpt
