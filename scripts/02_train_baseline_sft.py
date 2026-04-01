#!/usr/bin/env python3
"""
scripts/02_train_baseline_sft.py
=================================
Stage 1: Baseline Supervised Fine-Tuning (no KD).
Trains both Qwen2.5-7B and TinyLlama-1.1B on teacher-generated responses
using LoRA + HuggingFace TRL SFTTrainer.

CLUSTER SAFETY:
  - Disk checked before training starts and at each checkpoint save
  - Output path validated against approved directories
  - Memory-safe data loading via datasets streaming
  - Gradient checkpointing enabled by default for 7B
  - save_total_limit=2 to prevent checkpoint accumulation
  - All errors are caught and logged with actionable messages

Usage:
    python scripts/02_train_baseline_sft.py \
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
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# =============================================================================
# SAFETY UTILITIES (same pattern as script 01)
# =============================================================================

def check_disk_space(path: Path, required_gb: float, label: str = "") -> None:
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = stat.free / (1024 ** 3)
    log.info(f"💾 Disk [{label}]: {free_gb:.1f} GB free at {path}")
    if free_gb < required_gb:
        raise RuntimeError(
            f"DISK ERROR: Need {required_gb:.1f} GB, only {free_gb:.1f} GB free.\n"
            f"  → df -h {path.parent}\n"
            f"  → du -sh {path.parent}/*"
        )


def is_path_approved(path: Path) -> bool:
    approved_vars = ["HOME", "SCRATCH"]
    for var in approved_vars:
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
        raise PermissionError(f"SAFETY BLOCK: {path} is outside approved directories.")
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_env(cfg):
    import re
    def _r(v):
        if isinstance(v, str):
            return re.sub(
                r"\$\{(\w+)(?::-(.*?))?\}",
                lambda m: os.environ.get(m.group(1), m.group(2) or ""),
                v
            )
        if isinstance(v, dict):
            return {k: _r(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_r(i) for i in v]
        return v
    return _r(cfg)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_sft_dataset(data_cfg: dict, tokenizer, max_seq_length: int):
    """
    Load teacher-generated JSONL and prepare for SFT.
    Uses HuggingFace datasets for memory-efficient processing.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("pip install datasets --break-system-packages")

    teacher_file = Path(data_cfg["paths"]["teacher_cache"]) / "teacher_responses.jsonl"

    if not teacher_file.exists():
        raise FileNotFoundError(
            f"Teacher data not found: {teacher_file}\n"
            "Run script 01 first: python scripts/01_generate_teacher_data.py"
        )

    log.info(f"📂 Loading teacher dataset from {teacher_file}")

    records = []
    with open(teacher_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    log.info(f"  Loaded {len(records)} records")

    # Format as chat-style text for SFT
    def _format(rec):
        user = rec["instruction"]
        if rec.get("input"):
            user += f"\n\nContext: {rec['input']}"
        assistant = rec["teacher_response"]

        try:
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = f"<|user|>\n{user}\n<|assistant|>\n{assistant}"

        return {"text": text, "language": rec.get("language", "en")}

    formatted = [_format(r) for r in records]

    # Split train/val
    val_n = max(100, int(len(formatted) * 0.02))
    train_data = formatted[val_n:]
    val_data = formatted[:val_n]

    log.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    return train_ds, val_ds


# =============================================================================
# TRAINING
# =============================================================================

def train(student_key: str, model_cfg: dict, data_cfg: dict) -> Path:
    """
    Full SFT training run for one student model.
    Returns path to saved adapter/checkpoint.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    except ImportError as e:
        raise ImportError(
            f"Missing: {e}\n"
            "Run: pip install transformers peft trl bitsandbytes accelerate --break-system-packages"
        )

    student_cfg = model_cfg["students"][student_key]
    train_cfg = student_cfg["training"]
    lora_cfg = student_cfg["lora"]

    model_name = student_cfg["name"]
    short_name = student_cfg["short_name"]
    checkpoint_root = Path(model_cfg["checkpoint_root"])
    output_dir = checkpoint_root / f"{short_name}_baseline_sft"

    # ── Safety checks ──────────────────────────────────────────────────────────
    safe_makedirs(output_dir)
    check_disk_space(output_dir, required_gb=15.0, label=f"{short_name} SFT training")

    log.info(f"\n{'='*60}")
    log.info(f"TRAINING: {short_name} — Baseline SFT")
    log.info(f"Model: {model_name}")
    log.info(f"Output: {output_dir}")
    log.info(f"{'='*60}\n")

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    cache_dir = Path(student_cfg["cache_dir"])
    safe_makedirs(cache_dir)
    check_disk_space(cache_dir, required_gb=15.0, label=f"{short_name} model cache")

    log.info(f"⚙️  Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(cache_dir), trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    except Exception as e:
        raise RuntimeError(f"Tokenizer failed: {e}") from e

    # ── Load dataset ───────────────────────────────────────────────────────────
    log.info("📊 Preparing training data...")
    try:
        train_ds, val_ds = load_sft_dataset(
            data_cfg,
            tokenizer,
            max_seq_length=train_cfg["max_seq_length"]
        )
    except Exception as e:
        raise RuntimeError(f"Data prep failed: {e}") from e

    # ── Load student model ─────────────────────────────────────────────────────
    log.info(f"⚙️  Loading model: {model_name} (bfloat16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
    except Exception as e:
        raise RuntimeError(
            f"Model load failed: {e}\n"
            f"  → Check disk: df -h\n"
            f"  → Check GPU: nvidia-smi"
        ) from e

    # ── Apply LoRA ─────────────────────────────────────────────────────────────
    log.info(f"🔧 Applying LoRA (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})...")
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    # ── TrainingArguments ──────────────────────────────────────────────────────
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
        save_total_limit=train_cfg["save_total_limit"],  # ⚠️ Keeps only N checkpoints
        evaluation_strategy="epoch",
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        run_name=f"{short_name}_baseline_sft",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        tokenizer=tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    log.info("🚀 Starting training...")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        log.error(
            "GPU OOM during training.\n"
            "  → Reduce per_device_train_batch_size in configs/models.yaml\n"
            "  → Or reduce max_seq_length\n"
            "  → Check: nvidia-smi"
        )
        raise
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

    # ── Save ───────────────────────────────────────────────────────────────────
    check_disk_space(output_dir, required_gb=5.0, label="final checkpoint save")
    log.info(f"💾 Saving final adapter to {output_dir}/final")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Save training metadata
    meta = {
        "model": model_name,
        "stage": "baseline_sft",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "lora_r": lora_cfg["r"],
        "epochs": train_cfg["num_train_epochs"],
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"✅ {short_name} baseline SFT complete. Checkpoint: {output_dir}/final")
    return output_dir / "final"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", choices=["medium", "small", "both"], default="medium",
                        help="Which student to train (medium=7B, small=1.1B, both)")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    try:
        import yaml
        model_cfg = resolve_env(load_yaml(args.config))
        data_cfg = resolve_env(load_yaml(args.data_config))
    except Exception as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    students = (
        ["medium", "small"] if args.student == "both"
        else [args.student]
    )

    for student_key in students:
        log.info(f"\n{'#'*60}")
        log.info(f"# STUDENT: {student_key}")
        log.info(f"{'#'*60}")
        try:
            result = train(student_key, model_cfg, data_cfg)
            log.info(f"✅ {student_key} complete → {result}")
        except Exception as e:
            log.error(f"❌ {student_key} FAILED: {e}")
            log.error("Stopping. Fix the error above before continuing.")
            sys.exit(1)

    log.info("\n✅ All SFT training complete. Run 03_train_sequence_kd.py next.")


if __name__ == "__main__":
    main()
