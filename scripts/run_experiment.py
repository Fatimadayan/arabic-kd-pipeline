#!/usr/bin/env python3
"""
scripts/run_experiment.py
==========================
Master pipeline orchestrator.

Usage examples:
    # Validate everything (no GPU, no data needed)
    python scripts/run_experiment.py --dry-run

    # Quick smoke test with first model config
    python scripts/run_experiment.py --test-run --model-index 0

    # Generate teacher data only
    python scripts/run_experiment.py --stage generate

    # Train 7B SFT baseline
    python scripts/run_experiment.py --stage train_sft --student medium

    # Train token KD (T=2, alpha=0.5)
    python scripts/run_experiment.py --stage train_kd --student medium --temperature 2 --alpha 0.5

    # Evaluate a specific checkpoint
    python scripts/run_experiment.py --stage eval --model-name qwen7b_sft \
        --model-path checkpoints/qwen7b_baseline_sft/final

    # Run full analysis (after all models evaluated)
    python scripts/run_experiment.py --stage analysis

    # Show experiment status (what's done, what's pending)
    python scripts/run_experiment.py --status

CLUSTER SAFETY:
    - Disk checked before every stage
    - All paths validated against /data/datasets/$USER
    - --dry-run always safe (no downloads, no GPU)
    - --test-run uses tiny samples (50) to validate pipeline
    - Never writes to $HOME
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# =============================================================================
# SAFETY UTILITIES
# =============================================================================

DATA_ROOT = Path(f"/data/datasets/{os.environ.get('USER', 'user')}")
APPROVED_ROOTS = [DATA_ROOT]


def _get_approved_roots() -> list[Path]:
    """Get approved write locations for Benefit Lab."""
    roots = [DATA_ROOT]
    # Also allow $SCRATCH if set
    scratch = os.environ.get("SCRATCH")
    if scratch:
        roots.append(Path(scratch))
    return roots


def is_path_safe(path: Path) -> bool:
    """Reject any path not under /data/datasets/$USER."""
    resolved = path.resolve()
    for root in _get_approved_roots():
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def check_disk(path: Path, required_gb: float, label: str = "") -> None:
    """Abort if insufficient disk space."""
    check_path = path if path.exists() else path.parent
    stat = shutil.disk_usage(check_path)
    free_gb = stat.free / (1024 ** 3)
    log.info(f"💾 Disk [{label}]: {free_gb:.1f} GB free at {check_path}")
    if free_gb < required_gb:
        raise RuntimeError(
            f"DISK ERROR: Need {required_gb:.1f} GB but only {free_gb:.1f} GB free.\n"
            f"  → Run: df -h /data/datasets\n"
            f"  → Run: du -sh /data/datasets/{os.environ.get('USER','')}/*"
        )


def safe_makedirs(path: Path) -> None:
    if not is_path_safe(path):
        raise PermissionError(
            f"SAFETY BLOCK: {path} is outside /data/datasets/$USER.\n"
            f"Approved roots: {_get_approved_roots()}"
        )
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml")
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_env(cfg) -> dict:
    """Expand ${USER} and similar in config values."""
    import re
    def _r(v):
        if isinstance(v, str):
            # Handle ${USER} and ${VAR:-default}
            v = v.replace("${USER}", os.environ.get("USER", "user"))
            v = re.sub(
                r"\$\{(\w+)(?::-(.*?))?\}",
                lambda m: os.environ.get(m.group(1), m.group(2) or ""),
                v
            )
            return v
        if isinstance(v, dict): return {k: _r(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_r(i) for i in v]
        return v
    return _r(cfg)


# =============================================================================
# STATUS REPORTING
# =============================================================================

def show_status(cfg: dict) -> None:
    """Print experiment progress: what exists, what's missing."""
    paths = cfg["paths"]
    checkpoint_root = Path(paths["checkpoint_root"])
    results_root = Path(paths["results_root"])
    data_root = Path(paths["data_root"])

    print("\n" + "="*56)
    print("  EXPERIMENT STATUS")
    print("="*56)

    # Data
    teacher_file = data_root / "teacher_responses.jsonl"
    if teacher_file.exists():
        n = sum(1 for _ in open(teacher_file))
        print(f"  ✅ Teacher data: {n:,} samples ({teacher_file})")
    else:
        print(f"  ⬜ Teacher data: NOT GENERATED ({teacher_file})")

    print("")
    print("  Checkpoints:")
    for model in cfg["models"]:
        if model.get("role") == "teacher":
            continue
        ckpt = checkpoint_root / model["checkpoint_subdir"]
        status = "✅" if ckpt.exists() else "⬜"
        print(f"    {status} [{model['index']}] {model['name']} → {model['checkpoint_subdir']}")

    print("")
    print("  Evaluation results:")
    for model in cfg["models"]:
        if model.get("role") == "teacher":
            continue
        result_dir = results_root / model["name"]
        summary = result_dir / "summary.json"
        status = "✅" if summary.exists() else "⬜"
        print(f"    {status} [{model['index']}] {model['name']}")

    # Disk
    print("")
    if DATA_ROOT.exists():
        stat = shutil.disk_usage("/data/datasets")
        free_gb = stat.free / (1024 ** 3)
        print(f"  💾 /data/datasets: {free_gb:.1f} GB free")

    print("="*56 + "\n")


# =============================================================================
# STAGE RUNNERS
# =============================================================================

def run_generate(cfg: dict, dry_run: bool = False, test_run: bool = False,
                 max_samples: Optional[int] = None) -> None:
    """Stage 1: Generate teacher data."""
    log.info("\n" + "="*56)
    log.info("STAGE: Generate Teacher Data (Qwen2.5-32B)")
    log.info("="*56)

    data_root = Path(cfg["paths"]["data_root"])
    safe_makedirs(data_root)
    check_disk(data_root, required_gb=10.0, label="teacher generation")

    script = Path(cfg["paths"]["project_root"]) / "scripts/01_generate_teacher_data.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [
        sys.executable, str(script),
        "--config", "configs/models.yaml",
        "--data-config", "configs/data.yaml",
    ]

    if dry_run:
        cmd.append("--dry-run")
        log.info("DRY RUN: Would run teacher generation.")
    if test_run or max_samples:
        n = max_samples or cfg["test_run"]["max_samples"]
        cmd += ["--max-samples", str(n)]
        log.info(f"TEST RUN: Limited to {n} samples.")

    log.info(f"Command: {' '.join(cmd)}")
    if not dry_run:
        _run_subprocess(cmd)


def run_train(cfg: dict, student: str, stage: str,
              temperature: float = 2.0, alpha: float = 0.5,
              dry_run: bool = False, test_run: bool = False) -> None:
    """Stages 2-4: Train SFT / Seq-KD / Token-KD."""
    stage_map = {
        "sft": ("02_train_baseline_sft.py", None, None),
        "seq_kd": ("03_train_sequence_kd.py", None, None),
        "token_kd": ("04_train_token_kd.py", temperature, alpha),
    }

    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}. Choose: sft, seq_kd, token_kd")

    script_name, T, a = stage_map[stage]
    log.info(f"\n{'='*56}\nSTAGE: Train [{stage}] — Student: {student}\n{'='*56}")

    checkpoint_root = Path(cfg["paths"]["checkpoint_root"])
    safe_makedirs(checkpoint_root)
    check_disk(checkpoint_root, required_gb=15.0, label=f"{stage} training")

    script = Path(cfg["paths"]["project_root"]) / "scripts" / script_name

    cmd = [
        sys.executable, str(script),
        "--student", student,
        "--config", "configs/models.yaml",
        "--data-config", "configs/data.yaml",
    ]

    if stage == "token_kd":
        cmd += ["--temperature", str(T), "--alpha", str(a)]

    log.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        log.info("DRY RUN: Would run training. Skipping.")
        return

    _run_subprocess(cmd)


def run_eval(cfg: dict, model_name: str, model_path: str,
             pillars: Optional[list] = None,
             dry_run: bool = False) -> None:
    """Stage 5: 4-pillar evaluation."""
    log.info(f"\n{'='*56}\nSTAGE: Evaluate — {model_name}\n{'='*56}")

    results_root = Path(cfg["paths"]["results_root"])
    safe_makedirs(results_root / model_name)
    check_disk(results_root, required_gb=1.0, label="eval output")

    script = Path(cfg["paths"]["project_root"]) / "scripts/05_eval_4pillars.py"

    cmd = [
        sys.executable, str(script),
        "--model-path", model_path,
        "--model-name", model_name,
        "--config", "configs/eval.yaml",
    ]
    if pillars:
        cmd += ["--pillars"] + pillars

    log.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        log.info("DRY RUN: Would run evaluation. Skipping.")
        return

    _run_subprocess(cmd)


def run_analysis(cfg: dict, dry_run: bool = False) -> None:
    """Stage 6: Analysis and publication tables."""
    log.info(f"\n{'='*56}\nSTAGE: Analysis\n{'='*56}")

    analysis_root = Path(cfg["paths"]["analysis_root"])
    safe_makedirs(analysis_root)
    check_disk(analysis_root, required_gb=0.5, label="analysis output")

    script = Path(cfg["paths"]["project_root"]) / "scripts/06_analysis.py"

    cmd = [
        sys.executable, str(script),
        "--results-dir", cfg["paths"]["results_root"],
        "--output-dir", cfg["paths"]["analysis_root"],
    ]

    log.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        log.info("DRY RUN: Would run analysis. Skipping.")
        return

    _run_subprocess(cmd)


def run_all_pipeline(cfg: dict, student: str = "medium",
                     dry_run: bool = False, test_run: bool = False) -> None:
    """Run complete pipeline: generate → SFT → seq-KD → token-KD → eval → analysis."""
    log.info("\n🚀 FULL PIPELINE: generate → train → eval → analysis")

    run_generate(cfg, dry_run=dry_run, test_run=test_run)
    run_train(cfg, student=student, stage="sft", dry_run=dry_run, test_run=test_run)
    run_train(cfg, student=student, stage="seq_kd", dry_run=dry_run, test_run=test_run)
    run_train(cfg, student=student, stage="token_kd",
              temperature=2.0, alpha=0.5, dry_run=dry_run, test_run=test_run)

    # Evaluate all trained checkpoints
    checkpoint_root = Path(cfg["paths"]["checkpoint_root"])
    for model in cfg["models"]:
        if model.get("role") == "teacher":
            continue
        ckpt = checkpoint_root / model["checkpoint_subdir"]
        if ckpt.exists():
            run_eval(cfg, model["name"], str(ckpt), dry_run=dry_run)
        else:
            log.warning(f"Checkpoint not found — skipping eval for {model['name']}: {ckpt}")

    run_analysis(cfg, dry_run=dry_run)
    log.info("\n✅ Full pipeline complete.")


# =============================================================================
# SUBPROCESS RUNNER
# =============================================================================

def _run_subprocess(cmd: list[str]) -> None:
    """Run subprocess with real-time output. Raises on non-zero exit."""
    log.info(f"▶️  Running: {' '.join(str(c) for c in cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed with exit code {proc.returncode}.\n"
                f"  → Check logs above for the error.\n"
                f"  → Fix the error before continuing to the next stage."
            )
    except KeyboardInterrupt:
        proc.terminate()
        log.warning("KeyboardInterrupt — process terminated.")
        sys.exit(1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master experiment runner for Arabic KD pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py --dry-run
  python scripts/run_experiment.py --test-run --model-index 0
  python scripts/run_experiment.py --stage generate
  python scripts/run_experiment.py --stage train_sft --student medium
  python scripts/run_experiment.py --stage train_kd --student medium --temperature 2 --alpha 0.5
  python scripts/run_experiment.py --stage eval --model-name qwen7b_sft --model-path checkpoints/qwen7b_baseline_sft/final
  python scripts/run_experiment.py --stage analysis
  python scripts/run_experiment.py --status
  python scripts/run_experiment.py --all
        """
    )

    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--stage",
                        choices=["generate", "train_sft", "train_seq_kd",
                                 "train_kd", "eval", "analysis"],
                        help="Pipeline stage to run")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--status", action="store_true", help="Show experiment status")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without downloading or running GPU code")
    parser.add_argument("--test-run", action="store_true",
                        help="Quick smoke test with tiny samples (50)")
    parser.add_argument("--model-index", type=int,
                        help="Model index from experiment_config.yaml for test-run")

    # Training args
    parser.add_argument("--student", choices=["medium", "small"], default="medium")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)

    # Eval args
    parser.add_argument("--model-name", help="Name for eval output folder")
    parser.add_argument("--model-path", help="Path to checkpoint for eval")
    parser.add_argument("--pillars", nargs="+", choices=["1","2","3","4"])

    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        cfg = resolve_env(load_yaml(args.config))
    except Exception as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    # ── Validate project root is safe ─────────────────────────────────────────
    project_root = Path(cfg["paths"]["project_root"])
    if not is_path_safe(project_root):
        log.error(
            f"SAFETY BLOCK: project_root={project_root} is outside /data/datasets/$USER.\n"
            "Edit configs/experiment_config.yaml and set paths.project_root correctly."
        )
        sys.exit(1)

    # ── Change to project root ────────────────────────────────────────────────
    if project_root.exists():
        os.chdir(project_root)
        log.info(f"CWD: {project_root}")

    # Set HF cache env var if not set
    hf_cache = cfg["paths"]["hf_cache"]
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = hf_cache
        log.warning(f"HF_HOME not set — using {hf_cache}. Add to ~/.bashrc for persistence.")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    try:
        if args.status:
            show_status(cfg)

        elif args.dry_run:
            log.info("\n🔍 DRY RUN — validating config and paths only")
            log.info(f"  Project root: {project_root}")
            log.info(f"  Data root:    {cfg['paths']['data_root']}")
            log.info(f"  Checkpoints:  {cfg['paths']['checkpoint_root']}")
            log.info(f"  Results:      {cfg['paths']['results_root']}")
            log.info(f"  HF cache:     {cfg['paths']['hf_cache']}")
            log.info(f"  Models configured: {len(cfg['models'])}")
            log.info(f"  Safety check: run bash scripts/00_safety_check.sh")

            # Check disk even in dry run
            if DATA_ROOT.exists():
                check_disk(DATA_ROOT, required_gb=cfg["safety"]["warn_free_disk_gb"],
                           label="dry-run disk check")

            log.info("\n✅ Dry run complete. All paths are safe.")

        elif args.test_run:
            model_idx = args.model_index or 0
            model = cfg["models"][model_idx]
            log.info(f"\n🧪 TEST RUN — model index {model_idx}: {model['name']}")
            log.info(f"  Using {cfg['test_run']['max_samples']} samples only.")
            run_generate(cfg, test_run=True, max_samples=cfg["test_run"]["max_samples"])
            log.info("✅ Test run complete.")

        elif args.all:
            run_all_pipeline(cfg, student=args.student,
                             dry_run=args.dry_run, test_run=args.test_run)

        elif args.stage == "generate":
            run_generate(cfg, dry_run=args.dry_run, test_run=args.test_run)

        elif args.stage in ("train_sft", "train_seq_kd", "train_kd"):
            stage_key = {
                "train_sft": "sft",
                "train_seq_kd": "seq_kd",
                "train_kd": "token_kd",
            }[args.stage]
            run_train(cfg, student=args.student, stage=stage_key,
                      temperature=args.temperature, alpha=args.alpha,
                      dry_run=args.dry_run, test_run=args.test_run)

        elif args.stage == "eval":
            if not args.model_name or not args.model_path:
                log.error("--stage eval requires --model-name and --model-path")
                sys.exit(1)
            run_eval(cfg, model_name=args.model_name,
                     model_path=args.model_path,
                     pillars=args.pillars,
                     dry_run=args.dry_run)

        elif args.stage == "analysis":
            run_analysis(cfg, dry_run=args.dry_run)

        else:
            parser.print_help()
            log.info("\nTip: Start with: python scripts/run_experiment.py --dry-run")

    except RuntimeError as e:
        log.error(f"\n❌ STAGE FAILED:\n{e}")
        log.error("Fix the error above before continuing.")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("\nInterrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
