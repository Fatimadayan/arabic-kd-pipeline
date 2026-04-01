#!/usr/bin/env python3
"""
scripts/01_generate_teacher_data.py
====================================
Generate teacher (Qwen2.5-32B-Instruct) responses for KD dataset.

CLUSTER SAFETY FEATURES:
  - Disk space checked before every write operation
  - All paths validated against approved directories only
  - 4-bit quantization to minimise VRAM
  - Checkpoint/resume — safe to cancel and restart
  - try/except on every IO and model operation
  - No sudo, no system modifications, no background daemons
  - Estimated storage printed before any download

Usage:
    python scripts/01_generate_teacher_data.py --config configs/models.yaml \
        --data-config configs/data.yaml --dry-run
    # Remove --dry-run when satisfied with estimates
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Optional

# ── Safety: fail immediately on any unhandled exception ──────────────────────
import traceback


def _global_exception_handler(exc_type, exc_value, exc_tb):
    logging.error("UNHANDLED EXCEPTION — stopping immediately.")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.exit(1)

sys.excepthook = _global_exception_handler

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# =============================================================================
# CLUSTER SAFETY UTILITIES
# =============================================================================

APPROVED_ROOT_VARS = ["SCRATCH", "HOME"]


def get_approved_roots() -> list[Path]:
    """Return list of approved write directories based on env vars."""
    roots = []
    for var in APPROVED_ROOT_VARS:
        val = os.environ.get(var)
        if val:
            roots.append(Path(val).resolve())
    return roots


def is_path_approved(path: Path) -> bool:
    """Reject any path outside $HOME or $SCRATCH."""
    approved = get_approved_roots()
    resolved = path.resolve()
    for root in approved:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def check_disk_space(path: Path, required_gb: float, label: str = "") -> None:
    """
    Raise RuntimeError if available disk space < required_gb.
    Always call this before writing large files.
    """
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024 ** 3)
        total_gb = stat.total / (1024 ** 3)
        log.info(
            f"💾 Disk check [{label}]: {available_gb:.1f} GB free / "
            f"{total_gb:.1f} GB total at {path}"
        )
        if available_gb < required_gb:
            raise RuntimeError(
                f"DISK SPACE ERROR: Need {required_gb:.1f} GB but only "
                f"{available_gb:.1f} GB available at {path}.\n"
                f"  → Free up space or point --output-dir to a larger partition.\n"
                f"  → Run: du -sh {path}/* to find large files."
            )
        log.info(f"✅ Disk check passed — {available_gb:.1f} GB available.")
    except FileNotFoundError:
        log.warning(f"Disk check path {path} not found — skipping check.")


def safe_makedirs(path: Path) -> None:
    """Create directory only if inside approved roots."""
    if not is_path_approved(path):
        raise PermissionError(
            f"SAFETY BLOCK: Refusing to create directory outside approved paths: {path}\n"
            f"Approved roots: {get_approved_roots()}"
        )
    path.mkdir(parents=True, exist_ok=True)
    log.info(f"📁 Directory ready: {path}")


def safe_write_json(data: dict | list, path: Path, min_free_gb: float = 0.5) -> None:
    """Write JSON only after disk check."""
    check_disk_space(path.parent, min_free_gb, label=path.name)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.rename(path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Write failed for {path}: {e}") from e


def load_yaml_config(path: str) -> dict:
    """Load YAML config with clear error on missing file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML not installed. Run: pip install pyyaml --break-system-packages")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_env_vars(config: dict) -> dict:
    """Replace ${VAR:-default} patterns in config string values."""
    import re
    def _resolve(v):
        if isinstance(v, str):
            def _sub(m):
                var, default = m.group(1), m.group(2) or ""
                return os.environ.get(var, default)
            return re.sub(r"\$\{(\w+)(?::-(.*?))?\}", _sub, v)
        elif isinstance(v, dict):
            return {k: _resolve(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [_resolve(i) for i in v]
        return v
    return _resolve(config)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_datasets(data_cfg: dict, dry_run: bool = False) -> list[dict]:
    """
    Load raw instruction data from HuggingFace datasets.
    Returns list of {"instruction": str, "input": str|None, "language": str, "source": str}
    """
    try:
        from datasets import load_dataset, disable_progress_bar
    except ImportError:
        raise ImportError(
            "HuggingFace datasets not installed.\n"
            "Run: pip install datasets --break-system-packages"
        )

    all_samples = []
    target_total = data_cfg["processing"]["target_total"]
    ar_frac = data_cfg["processing"]["arabic_fraction"]
    en_frac = data_cfg["processing"]["english_fraction"]

    target_ar = int(target_total * ar_frac)
    target_en = int(target_total * en_frac)

    log.info(f"🎯 Target dataset: {target_total} samples ({target_ar} AR, {target_en} EN)")

    if dry_run:
        log.info("DRY RUN: Skipping actual dataset download.")
        # Return synthetic samples for validation
        for i in range(10):
            all_samples.append({
                "instruction": f"[DRY RUN AR] اشرح المفهوم التالي: {i}",
                "input": None,
                "language": "ar",
                "source": "dry_run",
            })
            all_samples.append({
                "instruction": f"[DRY RUN EN] Explain the following concept: {i}",
                "input": None,
                "language": "en",
                "source": "dry_run",
            })
        return all_samples

    for source_group, group_key in [
        (data_cfg["sources"]["arabic"], "ar"),
        (data_cfg["sources"]["english"], "en"),
    ]:
        group_target = target_ar if group_key == "ar" else target_en
        group_collected = 0

        for src in source_group:
            if group_collected >= group_target:
                break
            remaining = group_target - group_collected
            n = min(src.get("max_samples", remaining), remaining)

            log.info(f"📥 Loading {src['name']} ({n} samples)...")
            try:
                kwargs = {"split": src.get("split", "train")}
                if "config" in src:
                    ds = load_dataset(src["hf_path"], src["config"], **kwargs)
                else:
                    ds = load_dataset(src["hf_path"], **kwargs)

                ds = ds.select(range(min(n, len(ds))))

                for row in ds:
                    sample = _parse_row(row, src)
                    if sample:
                        all_samples.append(sample)
                        group_collected += 1

                log.info(f"  ✅ Loaded {group_collected} {group_key} samples so far")

            except Exception as e:
                log.warning(f"  ⚠️  Failed to load {src['name']}: {e} — skipping source.")
                continue

    log.info(f"📊 Total samples loaded: {len(all_samples)}")
    return all_samples


def _parse_row(row: dict, source_cfg: dict) -> Optional[dict]:
    """Convert a dataset row into our standard schema."""
    src_name = source_cfg["name"]
    lang = source_cfg.get("language", "en")

    # ArabicMMLU / MMLU → convert MC to instruction
    if "question" in row and "choices" in row:
        choices_str = "\n".join(
            [f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"])]
        )
        instruction = f"{row['question']}\n\n{choices_str}"
        return {"instruction": instruction, "input": None, "language": lang, "source": src_name}

    # Alpaca / ORCA style
    if "instruction" in row:
        return {
            "instruction": row["instruction"],
            "input": row.get("input") or row.get("context") or None,
            "language": lang,
            "source": src_name,
        }

    # GSM8K
    if "question" in row:
        return {"instruction": row["question"], "input": None, "language": lang, "source": src_name}

    # XQuAD
    if "context" in row and "question" in row:
        return {
            "instruction": row["question"],
            "input": row["context"][:300],
            "language": lang,
            "source": src_name,
        }

    return None  # Unrecognised format — skip


# =============================================================================
# TEACHER MODEL LOADING
# =============================================================================

def load_teacher_model(model_cfg: dict, dry_run: bool = False):
    """
    Load Qwen2.5-32B-Instruct with 4-bit quantization.
    ⚠️  This downloads ~18GB — always check disk first.
    """
    if dry_run:
        log.info("DRY RUN: Skipping model load.")
        return None, None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Run: pip install transformers bitsandbytes accelerate --break-system-packages"
        )

    model_name = model_cfg["teacher"]["name"]
    cache_dir = Path(model_cfg["teacher"]["cache_dir"])

    # ── Disk check before model download ──────────────────────────────────────
    safe_makedirs(cache_dir)
    check_disk_space(cache_dir, required_gb=20.0, label="teacher model cache")

    log.info(f"⚙️  Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        raise RuntimeError(f"Tokenizer load failed: {e}") from e

    log.info(f"⚙️  Loading model with 4-bit quantization: {model_name}")
    log.info("   This may take 5-10 minutes on first run (download ~18GB)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"Model load failed: {e}\n"
            "  → Check GPU memory with: nvidia-smi\n"
            "  → Check disk with: df -h"
        ) from e

    log.info("✅ Teacher model loaded successfully.")
    return model, tokenizer


# =============================================================================
# TEACHER INFERENCE
# =============================================================================

def generate_teacher_responses(
    model,
    tokenizer,
    samples: list[dict],
    gen_cfg: dict,
    output_dir: Path,
    checkpoint_every: int = 500,
    dry_run: bool = False,
    store_logits: bool = True,
    top_k_logits: int = 50,
) -> Path:
    """
    Generate teacher responses with checkpointing.
    Saves results to output_dir/teacher_responses.jsonl (incremental).
    Returns path to completed dataset.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not installed.")

    safe_makedirs(output_dir)

    output_file = output_dir / "teacher_responses.jsonl"
    logits_dir = output_dir / "logits"

    # Check for existing checkpoint (resume support)
    completed_ids = set()
    if output_file.exists():
        log.info(f"🔄 Resuming from checkpoint: {output_file}")
        with open(output_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec.get("id"))
                except json.JSONDecodeError:
                    continue
        log.info(f"  Found {len(completed_ids)} completed samples — skipping these.")

    if store_logits and not dry_run:
        safe_makedirs(logits_dir)

    # Estimate output size before starting
    est_gb = len(samples) * 512 * 4 / (1024 ** 3)  # rough: 512 tokens × 4 bytes
    log.info(f"📐 Estimated output size: ~{est_gb:.2f} GB for {len(samples)} samples")
    check_disk_space(output_dir, required_gb=max(est_gb * 1.5, 2.0), label="teacher outputs")

    pending = [s for s in samples if _sample_id(s) not in completed_ids]
    log.info(f"▶️  Generating responses for {len(pending)} samples...")

    gen_params = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", 512),
        "temperature": gen_cfg.get("temperature", 0.7),
        "top_p": gen_cfg.get("top_p", 0.9),
        "do_sample": gen_cfg.get("do_sample", True),
        "repetition_penalty": gen_cfg.get("repetition_penalty", 1.1),
    }

    batch_size = gen_cfg.get("generation_batch_size", 4)
    t_start = time.time()

    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, sample in enumerate(pending):
            if dry_run:
                # Synthetic response for validation
                record = {
                    "id": _sample_id(sample),
                    "instruction": sample["instruction"],
                    "input": sample.get("input"),
                    "teacher_response": "[DRY RUN RESPONSE]",
                    "language": sample["language"],
                    "source": sample["source"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            try:
                prompt = _format_prompt(sample, tokenizer)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(model.device)

                with torch.no_grad():
                    if store_logits:
                        outputs = model.generate(
                            **inputs,
                            **gen_params,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                        generated_ids = outputs.sequences
                        scores = outputs.scores  # tuple of (vocab_size,) tensors

                        # Save compressed top-k logits
                        logits_path = _save_logits(
                            scores, logits_dir, _sample_id(sample), top_k=top_k_logits
                        )
                    else:
                        generated_ids = model.generate(**inputs, **gen_params)
                        logits_path = None

                response_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                record = {
                    "id": _sample_id(sample),
                    "instruction": sample["instruction"],
                    "input": sample.get("input"),
                    "teacher_response": response_text,
                    "language": sample["language"],
                    "source": sample["source"],
                }
                if logits_path:
                    record["teacher_logits_path"] = str(logits_path)

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            except torch.cuda.OutOfMemoryError:
                log.error(
                    f"GPU OOM at sample {i} — reduce batch_size or max_new_tokens.\n"
                    "  → Run: nvidia-smi\n"
                    "  → Checkpoint saved — safe to restart."
                )
                raise
            except Exception as e:
                log.warning(f"  ⚠️  Sample {i} failed: {e} — skipping.")
                continue

            # Progress + checkpoint disk check
            if (i + 1) % checkpoint_every == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (len(pending) - i - 1) / rate / 60
                log.info(
                    f"  [{i+1}/{len(pending)}] "
                    f"{rate:.1f} samples/s — {remaining:.0f} min remaining"
                )
                # Periodic disk check
                check_disk_space(output_dir, required_gb=1.0, label="checkpoint")

    log.info(f"✅ Teacher generation complete: {output_file}")
    return output_file


def _sample_id(sample: dict) -> str:
    """Deterministic ID from instruction content."""
    content = (sample.get("instruction", "") + sample.get("source", "")).encode()
    return hashlib.md5(content).hexdigest()[:12]


def _format_prompt(sample: dict, tokenizer) -> str:
    """Format instruction as chat template."""
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content = f"{user_content}\n\nContext: {sample['input']}"

    messages = [{"role": "user", "content": user_content}]

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback if no chat template
        return f"<|user|>\n{user_content}\n<|assistant|>\n"


def _save_logits(scores, logits_dir: Path, sample_id: str, top_k: int = 50) -> Path:
    """Save compressed top-k logits. Returns save path."""
    import torch
    import numpy as np

    logits_file = logits_dir / f"{sample_id}.npz"
    try:
        # Convert scores (tuple of tensors) to top-k compressed format
        top_indices = []
        top_values = []
        for token_scores in scores:
            probs = torch.softmax(token_scores[0], dim=-1)
            topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
            top_indices.append(topk.indices.cpu().numpy().astype(np.int32))
            top_values.append(topk.values.cpu().numpy().astype(np.float16))

        np.savez_compressed(
            logits_file,
            indices=np.array(top_indices),
            values=np.array(top_values),
        )
        return logits_file
    except Exception as e:
        log.warning(f"Failed to save logits for {sample_id}: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher responses for KD dataset"
    )
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (must be under $HOME or $SCRATCH)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths/configs without downloading or generating")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total samples (for testing)")
    parser.add_argument("--no-logits", action="store_true",
                        help="Skip logits storage (saves ~5GB)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("01_generate_teacher_data.py")
    log.info(f"Dry run: {args.dry_run}")
    log.info("=" * 60)

    # ── Load configs ──────────────────────────────────────────────────────────
    try:
        model_cfg = resolve_env_vars(load_yaml_config(args.config))
        data_cfg = resolve_env_vars(load_yaml_config(args.data_config))
    except Exception as e:
        log.error(f"Config load failed: {e}")
        sys.exit(1)

    # ── Resolve output dir ────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(data_cfg["paths"]["teacher_cache"])

    if not is_path_approved(output_dir):
        log.error(
            f"SAFETY BLOCK: Output directory {output_dir} is outside approved paths.\n"
            f"Approved: $HOME ({os.environ.get('HOME')}) or $SCRATCH ({os.environ.get('SCRATCH', 'not set')})"
        )
        sys.exit(1)

    # Set up log file inside project
    log_dir = Path(data_cfg["paths"]["logs"])
    safe_makedirs(log_dir)
    log_file = log_dir / "01_generate_teacher_data.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(file_handler)
    log.info(f"Logging to: {log_file}")

    # ── Pre-flight disk check ─────────────────────────────────────────────────
    log.info("\n📋 PRE-FLIGHT CHECKS")
    log.info(f"  HOME    = {os.environ.get('HOME', 'not set')}")
    log.info(f"  SCRATCH = {os.environ.get('SCRATCH', 'not set (using HOME)')}")
    log.info(f"  Output  = {output_dir}")

    check_disk_space(
        output_dir.parent if not output_dir.exists() else output_dir,
        required_gb=10.0,
        label="pre-flight"
    )

    # ── Load datasets ─────────────────────────────────────────────────────────
    log.info("\n📥 LOADING DATASETS")
    try:
        samples = load_datasets(data_cfg, dry_run=args.dry_run)
    except Exception as e:
        log.error(f"Dataset loading failed: {e}")
        sys.exit(1)

    if args.max_samples:
        samples = samples[:args.max_samples]
        log.info(f"⚙️  Limited to {len(samples)} samples (--max-samples)")

    if args.dry_run:
        log.info(f"\n✅ DRY RUN COMPLETE")
        log.info(f"  Samples that would be processed: {len(samples)}")
        log.info(f"  Output directory (would create): {output_dir}")
        log.info(f"  Estimated generation time @ 2 samples/min: {len(samples)/2/60:.1f} hours")
        log.info(f"  Estimated storage: ~{len(samples)*512*4/1024**3:.1f} GB")
        log.info(f"\n  Run without --dry-run to proceed.")
        sys.exit(0)

    # ── Save raw processed samples ────────────────────────────────────────────
    safe_makedirs(Path(data_cfg["paths"]["processed"]))
    raw_out = Path(data_cfg["paths"]["processed"]) / "raw_instructions.json"
    log.info(f"\n💾 Saving processed instructions to {raw_out}")
    safe_write_json(samples, raw_out, min_free_gb=1.0)

    # ── Load teacher model ────────────────────────────────────────────────────
    log.info("\n🤖 LOADING TEACHER MODEL")
    try:
        model, tokenizer = load_teacher_model(model_cfg, dry_run=args.dry_run)
    except Exception as e:
        log.error(f"Teacher model load failed: {e}")
        sys.exit(1)

    # ── Generate responses ────────────────────────────────────────────────────
    log.info("\n🎓 GENERATING TEACHER RESPONSES")
    try:
        result_path = generate_teacher_responses(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            gen_cfg={
                **model_cfg["teacher"]["generation"],
                "generation_batch_size": data_cfg["processing"]["generation_batch_size"],
            },
            output_dir=output_dir,
            store_logits=not args.no_logits and data_cfg["logits"]["store_logits"],
            top_k_logits=data_cfg["logits"]["top_k"],
        )
    except Exception as e:
        log.error(f"Generation failed: {e}")
        sys.exit(1)

    # ── Final disk report ─────────────────────────────────────────────────────
    log.info("\n📊 FINAL REPORT")
    log.info(f"  Output: {result_path}")
    if result_path.exists():
        size_mb = result_path.stat().st_size / (1024 ** 2)
        log.info(f"  File size: {size_mb:.1f} MB")
        with open(result_path) as f:
            n_lines = sum(1 for _ in f)
        log.info(f"  Records written: {n_lines}")

    check_disk_space(output_dir, required_gb=0.1, label="final")
    log.info("\n✅ Phase 1 complete. Run 02_train_baseline_sft.py next.")


if __name__ == "__main__":
    main()
