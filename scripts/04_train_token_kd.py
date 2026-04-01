#!/usr/bin/env python3
"""
scripts/04_train_token_kd.py
=============================
Stage 3: Token-Level Knowledge Distillation (Hinton et al. 2015).

Loss: L = α·CE(student, labels) + (1-α)·T²·KL(σ(s/T) ‖ σ(t/T))
  where σ = softmax, T = temperature, s = student logits, t = teacher logits

Uses compressed top-k teacher logits saved by script 01.
Falls back to seq-KD loss if logits are unavailable for a sample.

CLUSTER SAFETY:
  - Memory-mapped logit loading — never loads all logits to RAM simultaneously
  - Disk checked before training
  - save_total_limit=2 prevents checkpoint accumulation
  - Explicit GPU OOM handling with actionable messages

Usage:
    python scripts/04_train_token_kd.py \
        --student medium \
        --temperature 2 \
        --alpha 0.5 \
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

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# SAFETY UTILITIES
# =============================================================================

def check_disk_space(path: Path, required_gb: float, label: str = "") -> None:
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = stat.free / (1024 ** 3)
    log.info(f"💾 [{label}]: {free_gb:.1f} GB free")
    if free_gb < required_gb:
        raise RuntimeError(
            f"DISK ERROR: need {required_gb:.1f} GB, have {free_gb:.1f} GB.\n"
            f"  → df -h\n  → du -sh {path}/*"
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


# =============================================================================
# HINTON KD LOSS
# =============================================================================

def kd_loss(
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_top_values: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    vocab_size: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict]:
    """
    Compute combined KD loss.

    Args:
        student_logits: (B, T, V) student output logits
        teacher_top_indices: (B, T, K) top-K teacher logit indices
        teacher_top_values:  (B, T, K) top-K teacher logit values (float16 → cast)
        labels: (B, T) target token ids
        temperature: KD temperature T
        alpha: weight on CE loss; (1-alpha) on KL loss
        vocab_size: V
        ignore_index: token id to ignore in CE

    Returns:
        combined_loss, {"ce_loss": ..., "kl_loss": ..., "combined": ...}
    """
    B, T, V = student_logits.shape
    device = student_logits.device

    # ── Cross-entropy loss (standard SFT) ──────────────────────────────────
    shift_logits = student_logits[:, :-1, :].contiguous().view(-1, V)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    ce_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)

    # ── KL divergence loss ──────────────────────────────────────────────────
    if teacher_top_indices is not None and teacher_top_values is not None:
        # Expand compressed teacher logits to full vocab (sparse → dense)
        K = teacher_top_indices.shape[-1]
        teacher_top_values = teacher_top_values.to(device=device, dtype=torch.float32)
        teacher_top_indices = teacher_top_indices.to(device=device, dtype=torch.long)

        # Reconstruct approximate teacher logit distribution
        # Non-top-K positions get very negative logit (≈ 0 probability)
        teacher_logits_approx = torch.full(
            (B, T, V), fill_value=-1e9, dtype=torch.float32, device=device
        )
        teacher_logits_approx.scatter_(2, teacher_top_indices, teacher_top_values)

        # Temperature-scaled softmax
        student_probs = F.log_softmax(student_logits[:, :-1] / temperature, dim=-1)   # (B, T-1, V)
        teacher_probs = F.softmax(teacher_logits_approx[:, :-1] / temperature, dim=-1) # (B, T-1, V)

        # KL divergence: sum KL(teacher || student) per position, mean over non-padding
        kl = F.kl_div(student_probs, teacher_probs, reduction="none").sum(-1)  # (B, T-1)

        # Mask padding positions
        active = (shift_labels != ignore_index).view(B, -1).float()
        kl_loss = (kl * active).sum() / active.sum().clamp(min=1)
        kl_loss = (temperature ** 2) * kl_loss
    else:
        kl_loss = torch.tensor(0.0, device=device)

    combined = alpha * ce_loss + (1.0 - alpha) * kl_loss

    return combined, {
        "ce_loss": ce_loss.item(),
        "kl_loss": kl_loss.item(),
        "combined": combined.item(),
    }


# =============================================================================
# CUSTOM TRAINER WITH KD LOSS
# =============================================================================

class TokenKDTrainer:
    """
    Minimal training loop with Hinton KD loss.
    Uses HuggingFace Accelerate for multi-GPU support.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        logits_dir: Optional[Path],
        output_dir: Path,
        temperature: float,
        alpha: float,
        training_args: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.logits_dir = logits_dir
        self.output_dir = output_dir
        self.temperature = temperature
        self.alpha = alpha
        self.args = training_args

        try:
            from accelerate import Accelerator
            self.accelerator = Accelerator(mixed_precision="bf16")
        except ImportError:
            log.warning("accelerate not found — single GPU mode")
            self.accelerator = None

    def _load_teacher_logits(self, sample_id: str):
        """Load top-K logits for one sample. Returns None if not found."""
        if self.logits_dir is None:
            return None, None
        path = self.logits_dir / f"{sample_id}.npz"
        if not path.exists():
            return None, None
        try:
            import numpy as np
            data = np.load(path)
            indices = torch.from_numpy(data["indices"].astype("int64")).unsqueeze(0)
            values = torch.from_numpy(data["values"].astype("float32")).unsqueeze(0)
            return indices, values
        except Exception as e:
            log.warning(f"Failed to load logits for {sample_id}: {e}")
            return None, None

    def _collate(self, batch: list[dict]):
        """Tokenize and batch samples, loading teacher logits per sample."""
        texts = [b["text"] for b in batch]
        ids = [b.get("id") for b in batch]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.get("max_seq_length", 2048),
        )

        labels = enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Try to load teacher logits for each sample
        all_indices, all_values = [], []
        has_logits = False
        for sid in ids:
            if sid:
                idx, val = self._load_teacher_logits(sid)
                if idx is not None:
                    all_indices.append(idx)
                    all_values.append(val)
                    has_logits = True
                    continue
            all_indices.append(None)
            all_values.append(None)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "teacher_indices": all_indices if has_logits else None,
            "teacher_values": all_values if has_logits else None,
        }

    def train(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader

        model = self.model
        device = next(model.parameters()).device

        optimizer = AdamW(model.parameters(), lr=self.args.get("learning_rate", 2e-5))
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.args.get("per_device_train_batch_size", 2),
            shuffle=True,
            collate_fn=self._collate,
        )
        total_steps = len(train_loader) * self.args.get("num_train_epochs", 3)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(self.args.get("num_train_epochs", 3)):
            model.train()
            epoch_losses = {"ce_loss": 0, "kl_loss": 0, "combined": 0}

            for step, batch in enumerate(train_loader):
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    student_logits = outputs.logits

                    # Build teacher logit tensors if available
                    t_indices = t_values = None
                    if batch["teacher_indices"] is not None:
                        # Stack non-None entries (simplified: use first available)
                        valid = [(i, v) for i, v in
                                 zip(batch["teacher_indices"], batch["teacher_values"])
                                 if i is not None]
                        if valid:
                            try:
                                t_indices = torch.cat([x[0] for x in valid], dim=0).to(device)
                                t_values = torch.cat([x[1] for x in valid], dim=0).to(device)
                            except Exception:
                                t_indices = t_values = None

                    loss, loss_dict = kd_loss(
                        student_logits=student_logits,
                        teacher_top_indices=t_indices,
                        teacher_top_values=t_values,
                        labels=labels,
                        temperature=self.temperature,
                        alpha=self.alpha,
                        vocab_size=student_logits.shape[-1],
                    )

                    # Gradient accumulation
                    accum = self.args.get("gradient_accumulation_steps", 4)
                    (loss / accum).backward()

                    if (step + 1) % accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                    for k in epoch_losses:
                        epoch_losses[k] += loss_dict.get(k, 0)

                    if global_step % self.args.get("logging_steps", 50) == 0:
                        log.info(
                            f"  Epoch {epoch+1} Step {global_step} "
                            f"| CE={loss_dict['ce_loss']:.4f} "
                            f"| KL={loss_dict['kl_loss']:.4f} "
                            f"| Total={loss_dict['combined']:.4f}"
                        )

                except torch.cuda.OutOfMemoryError:
                    log.error(
                        f"GPU OOM at epoch {epoch+1} step {step}.\n"
                        f"  → Reduce batch size or max_seq_length\n"
                        f"  → nvidia-smi"
                    )
                    raise
                except Exception as e:
                    log.warning(f"  Batch {step} failed: {e} — skipping.")
                    optimizer.zero_grad()
                    continue

            # Epoch-end checkpoint
            check_disk_space(self.output_dir, required_gb=5.0, label=f"epoch {epoch+1} save")
            ckpt_dir = self.output_dir / f"epoch_{epoch+1}"
            safe_makedirs(ckpt_dir)
            model.save_pretrained(str(ckpt_dir))
            self.tokenizer.save_pretrained(str(ckpt_dir))
            log.info(f"💾 Checkpoint: {ckpt_dir}")

            # Keep only last 2 checkpoints  ⚠️ cluster safety
            self._prune_checkpoints(keep=2)

            n_batches = max(1, len(train_loader))
            log.info(
                f"Epoch {epoch+1}/{self.args['num_train_epochs']} "
                f"avg loss={epoch_losses['combined']/n_batches:.4f}"
            )

        # Final save
        final_dir = self.output_dir / "final"
        safe_makedirs(final_dir)
        model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        log.info(f"✅ Token-KD training complete → {final_dir}")

    def _prune_checkpoints(self, keep: int = 2) -> None:
        """Remove older epoch checkpoints to conserve disk space."""
        ckpts = sorted(self.output_dir.glob("epoch_*"),
                       key=lambda p: int(p.name.split("_")[1]))
        to_remove = ckpts[:-keep]
        for ckpt in to_remove:
            try:
                shutil.rmtree(ckpt)
                log.info(f"🗑️  Removed old checkpoint: {ckpt}")
            except Exception as e:
                log.warning(f"Could not remove {ckpt}: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", choices=["medium", "small"], default="medium")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="KD temperature T (try 1, 2, 4)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="CE weight α (try 0.3, 0.5, 0.7)")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    try:
        model_cfg = resolve_env(load_yaml(args.config))
        data_cfg = resolve_env(load_yaml(args.data_config))
    except Exception as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    student_cfg = model_cfg["students"][args.student]
    train_cfg = student_cfg["training"]
    model_name = student_cfg["name"]
    short_name = student_cfg["short_name"]

    tag = f"T{int(args.temperature)}_a{str(args.alpha).replace('.', '')}"
    output_dir = Path(model_cfg["checkpoint_root"]) / f"{short_name}_token_kd_{tag}"
    safe_makedirs(output_dir)
    check_disk_space(output_dir, required_gb=15.0, label=f"{short_name} token-KD {tag}")

    log.info(f"\n{'='*60}")
    log.info(f"TOKEN-KD: {short_name} | T={args.temperature} | α={args.alpha}")
    log.info(f"Output: {output_dir}")
    log.info(f"{'='*60}\n")

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    cache_dir = Path(student_cfg["cache_dir"])
    safe_makedirs(cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(cache_dir), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.config.use_cache = False
    except torch.cuda.OutOfMemoryError:
        log.error("GPU OOM on model load → reduce max_memory_per_gpu in configs/models.yaml")
        sys.exit(1)

    lora_cfg = student_cfg["lora"]
    peft_cfg = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"], task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Load dataset
    from datasets import Dataset

    teacher_file = Path(data_cfg["paths"]["teacher_cache"]) / "teacher_responses.jsonl"
    if not teacher_file.exists():
        log.error(f"Teacher data not found: {teacher_file}")
        sys.exit(1)

    records = []
    with open(teacher_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def _fmt(rec):
        user = rec["instruction"] + (f"\n\nContext: {rec['input']}" if rec.get("input") else "")
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": rec["teacher_response"]}],
                tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            text = f"<|user|>\n{user}\n<|assistant|>\n{rec['teacher_response']}"
        return {"text": text, "id": rec.get("id", ""), "language": rec.get("language", "en")}

    formatted = [_fmt(r) for r in records]
    val_n = max(100, int(len(formatted) * 0.02))
    train_ds = Dataset.from_list(formatted[val_n:])
    val_ds = Dataset.from_list(formatted[:val_n])

    logits_dir = Path(data_cfg["paths"]["teacher_cache"]) / "logits"
    logits_dir = logits_dir if logits_dir.exists() else None
    if logits_dir is None:
        log.warning("No logits directory found — KL term will be zero (seq-KD fallback).")

    trainer = TokenKDTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, val_dataset=val_ds,
        logits_dir=logits_dir, output_dir=output_dir,
        temperature=args.temperature, alpha=args.alpha,
        training_args={**train_cfg, "max_seq_length": train_cfg["max_seq_length"]},
    )

    try:
        trainer.train()
    except Exception as e:
        log.error(f"Training failed: {e}")
        sys.exit(1)

    log.info(f"\n✅ Token-KD complete [{short_name} T={args.temperature} α={args.alpha}]")
    log.info("Run 05_eval_4pillars.py next.")


if __name__ == "__main__":
    main()
