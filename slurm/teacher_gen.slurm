#!/bin/bash
# =============================================================================
# slurm/teacher_gen.slurm
# BENEFIT LAB (UOB) — Teacher (Qwen2.5-32B) data generation
#
# ⚠️ BEFORE SUBMITTING:
#   1. Check partition:   sinfo
#   2. Check GPU names:   sinfo -o "%P %G"
#   3. Check disk:        df -h /data/datasets
#   4. Dry-run first:     python scripts/01_generate_teacher_data.py --dry-run
#
# Submit:  sbatch slurm/teacher_gen.slurm
# Monitor: squeue -u $USER
#          tail -f /data/datasets/$USER/qwen-arabic-kd/logs/teacher_gen_<ID>.out
# =============================================================================

#SBATCH --job-name=kd_teacher_gen
#SBATCH --partition=gpu              # ← VERIFY: sinfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00              # Hard limit — prevents runaway jobs
#SBATCH --output=/data/datasets/%u/qwen-arabic-kd/logs/teacher_gen_%j.out
#SBATCH --error=/data/datasets/%u/qwen-arabic-kd/logs/teacher_gen_%j.err

set -euo pipefail

# ── Benefit Lab: ALL data under /data/datasets/$USER ─────────────────────────
export DATA_ROOT="/data/datasets/$USER"
export PROJECT_ROOT="${DATA_ROOT}/qwen-arabic-kd"
export HF_HOME="${DATA_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"

echo "=============================================="
echo "  Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "  Node: ${SLURMD_NODENAME}"
echo "  Start: $(date)"
echo "  Project: ${PROJECT_ROOT}"
echo "=============================================="

mkdir -p "${PROJECT_ROOT}/logs" "${DATA_ROOT}/.cache/huggingface"

# ── Disk check ────────────────────────────────────────────────────────────────
AVAIL_GB=$(df -BG /data/datasets | awk 'NR==2 {gsub("G",""); print $4}')
echo "💾 /data/datasets: ${AVAIL_GB}G available"
[ "${AVAIL_GB}" -lt 30 ] && echo "ERROR: Need 30G, have ${AVAIL_GB}G" && exit 1
HOME_AVAIL=$(df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $4}')
[ "${HOME_AVAIL}" -lt 1 ] && echo "ERROR: HOME almost full" && exit 1

# ── Activate venv ─────────────────────────────────────────────────────────────
source /data/software/miniconda3/etc/profile.d/conda.sh && conda activate /data/datasets/user151/conda-envs/kd_env \
    || { echo "ERROR: Venv not found. Run: bash ${PROJECT_ROOT}/setup.sh"; exit 1; }

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

cd "${PROJECT_ROOT}"

# ── Dry-run validation first ──────────────────────────────────────────────────
echo "🔍 Dry-run validation..."
python scripts/01_generate_teacher_data.py \
    --config configs/models.yaml \
    --data-config configs/data.yaml \
    --dry-run
echo "✅ Dry run passed — starting generation"

# ── Full generation ───────────────────────────────────────────────────────────
# Start with 5000 for first real run. Once confirmed, remove --max-samples.
python scripts/01_generate_teacher_data.py \
    --config configs/models.yaml \
    --data-config configs/data.yaml \
    --max-samples 5000                 # ← CHANGE TO 100000 FOR FULL RUN

echo ""
echo "✅ Teacher generation complete: $(date)"
du -sh "${PROJECT_ROOT}/data/"* 2>/dev/null || true
df -h /data/datasets
