#!/usr/bin/env python3
"""
scripts/03_train_sequence_kd.py
================================
Stage 2: Sequence-Level Knowledge Distillation.
Uses teacher-generated responses as training targets (SFT on teacher outputs).
Structurally identical to script 02, but explicitly labelled as sequence-KD
for ablation tracking.

Sequence KD ≡ SFT with teacher responses — the teacher's text becomes the
ground truth; no logits are used at this stage.

CLUSTER SAFETY: Same guards as 02.

Usage:
    python scripts/03_train_sequence_kd.py \
        --student medium \
        --config configs/models.yaml \
        --data-config configs/data.yaml
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def check_disk_space(path: Path, required_gb: float, label: str = "") -> None:
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = stat.free / (1024 ** 3)
    log.info(f"💾 Disk [{label}]: {free_gb:.1f} GB free")
    if free_gb < required_gb:
        raise RuntimeError(
            f"DISK ERROR: Need {required_gb:.1f} GB, only {free_gb:.1f} free.\n"
            f"  → df -h | du -sh {path.parent}/*"
        )


def is_path_approved(path: Path) -> bool:
    for var in ["HOME", "SCRATCH"]:
        root = os.environ.get(var)
        if root:
            try:
                path.resolve().relative_to(Path(root).resolve())
                return True
            except ValueError:
                continue
    return False


def safe_makedirs(path: Path) -> None:
    if not is_path_approved(path):
        raise PermissionError(f"SAFETY BLOCK: {path} outside approved dirs.")
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_env(cfg):
    import re
    def _r(v):
        if isinstance(v, str):
            return re.sub(r"\$\{(\w+)(?::-(.*?))?\}",
                          lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
        if isinstance(v, dict): return {k: _r(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_r(i) for i in v]
        return v
    return _r(cfg)


def load_seq_kd_dataset(data_cfg: dict, tokenizer):
    """
    For sequence KD, dataset is identical to SFT — teacher response is ground truth.
    We tag records so downstream analysis can distinguish the stage.
    """
    from datasets import Dataset

    teacher_file = Path(data_cfg["paths"]["teacher_cache"]) / "teacher_responses.jsonl"
    if not teacher_file.exists():
        raise FileNotFoundError(
            f"Teacher data missing: {teacher_file}\n"
            "Run: python scripts/01_generate_teacher_data.py"
        )

    records, skipped = [], 0
    with open(teacher_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not rec.get("teacher_response"):
                    skipped += 1
                    continue
                records.append(rec)
            except json.JSONDecodeError:
                skipped += 1

    log.info(f"  Loaded {len(records)} records ({skipped} skipped — empty responses)")

    def _fmt(rec):
        user = rec["instruction"]
        if rec.get("input"):
            user += f"\n\nContext: {rec['input']}"
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": rec["teacher_response"]}],
                tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            text = f"<|user|>\n{user}\n<|assistant|>\n{rec['teacher_response']}"
        return {"text": text, "language": rec.get("language", "en"), "stage": "seq_kd"}

    formatted = [_fmt(r) for r in records]
    val_n = max(100, int(len(formatted) * 0.02))
    return Dataset.from_list(formatted[val_n:]), Dataset.from_list(formatted[:val_n])


def train(student_key: str, model_cfg: dict, data_cfg: dict) -> Path:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, TaskType
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(f"Missing: {e}\npip install transformers peft trl --break-system-packages")

    student_cfg = model_cfg["students"][student_key]
    train_cfg = student_cfg["training"]
    lora_cfg = student_cfg["lora"]

    model_name = student_cfg["name"]
    short_name = student_cfg["short_name"]
    output_dir = Path(model_cfg["checkpoint_root"]) / f"{short_name}_seq_kd"
    safe_makedirs(output_dir)
    check_disk_space(output_dir, required_gb=15.0, label=f"{short_name} seq-KD")

    log.info(f"\n{'='*60}\nSEQ-KD: {short_name}\n{'='*60}")

    cache_dir = Path(student_cfg["cache_dir"])
    safe_makedirs(cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(cache_dir), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, val_ds = load_seq_kd_dataset(data_cfg, tokenizer)
    log.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.config.use_cache = False
    except torch.cuda.OutOfMemoryError:
        log.error("GPU OOM on model load → nvidia-smi")
        raise

    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"], task_type=TaskType.CAUSAL_LM,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy="epoch",
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        run_name=f"{short_name}_seq_kd",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        tokenizer=tokenizer,
    )

    log.info("🚀 Training sequence KD...")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        log.error("OOM during training. Reduce batch size or seq length.")
        raise

    check_disk_space(output_dir, required_gb=5.0, label="save checkpoint")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    with open(output_dir / "training_meta.json", "w") as f:
        json.dump({"model": model_name, "stage": "seq_kd",
                   "train_n": len(train_ds), "val_n": len(val_ds)}, f, indent=2)

    log.info(f"✅ Seq-KD complete → {output_dir}/final")
    return output_dir / "final"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", choices=["medium", "small", "both"], default="medium")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    model_cfg = resolve_env(load_yaml(args.config))
    data_cfg = resolve_env(load_yaml(args.data_config))

    students = ["medium", "small"] if args.student == "both" else [args.student]
    for sk in students:
        try:
            train(sk, model_cfg, data_cfg)
        except Exception as e:
            log.error(f"FAILED [{sk}]: {e}")
            sys.exit(1)

    log.info("✅ All seq-KD done. Run 04_train_token_kd.py next.")


if __name__ == "__main__":
    main()
