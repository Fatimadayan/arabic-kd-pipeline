#!/bin/bash
# =============================================================================
# slurm/train_7b.slurm
# BENEFIT LAB (UOB) — Student training (SFT / Seq-KD stages)
#
# Submit examples:
#   # 7B Baseline SFT
#   sbatch --export=STUDENT=medium,STAGE=sft slurm/train_7b.slurm
#
#   # 7B Sequence KD
#   sbatch --export=STUDENT=medium,STAGE=seq_kd slurm/train_7b.slurm
#
#   # 1.1B Baseline SFT
#   sbatch --export=STUDENT=small,STAGE=sft slurm/train_7b.slurm
# =============================================================================

#SBATCH --job-name=kd_train_student
#SBATCH --partition=gpu              # ← VERIFY: sinfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                 # 7B fits on 1 GPU with LoRA. VERIFY: sinfo -o "%P %G"
#SBATCH --time=24:00:00
#SBATCH --output=/data/datasets/%u/qwen-arabic-kd/logs/train_%x_%j.out
#SBATCH --error=/data/datasets/%u/qwen-arabic-kd/logs/train_%x_%j.err

set -euo pipefail

export DATA_ROOT="/data/datasets/$USER"
export PROJECT_ROOT="${DATA_ROOT}/qwen-arabic-kd"
export HF_HOME="${DATA_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"

# Training parameters (override with --export when submitting)
STUDENT="${STUDENT:-medium}"     # medium=7B, small=1.1B
STAGE="${STAGE:-sft}"            # sft | seq_kd

echo "=============================================="
echo "  Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "  Start: $(date)"
echo "  Student: ${STUDENT} | Stage: ${STAGE}"
echo "=============================================="

mkdir -p "${PROJECT_ROOT}/logs"

# ── Disk check ────────────────────────────────────────────────────────────────
AVAIL_GB=$(df -BG /data/datasets | awk 'NR==2 {gsub("G",""); print $4}')
echo "💾 Disk: ${AVAIL_GB}G available"
[ "${AVAIL_GB}" -lt 20 ] && echo "ERROR: Need 20G" && exit 1

source "${DATA_ROOT}/venvs/kd_env/bin/activate" \
    || { echo "ERROR: Venv missing"; exit 1; }

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
cd "${PROJECT_ROOT}"

# ── Route to correct script ───────────────────────────────────────────────────
if [ "${STAGE}" = "sft" ]; then
    echo "🚀 Stage: Baseline SFT"
    python scripts/02_train_baseline_sft.py \
        --student "${STUDENT}" \
        --config configs/models.yaml \
        --data-config configs/data.yaml

elif [ "${STAGE}" = "seq_kd" ]; then
    echo "🚀 Stage: Sequence KD"
    python scripts/03_train_sequence_kd.py \
        --student "${STUDENT}" \
        --config configs/models.yaml \
        --data-config configs/data.yaml

else
    echo "ERROR: Unknown STAGE=${STAGE}. Use sft or seq_kd."
    echo "  For token-KD, use: sbatch slurm/train_kd_7b.slurm"
    exit 1
fi

echo ""
echo "✅ Training complete: $(date)"
echo "Checkpoint sizes:"
du -sh "${DATA_ROOT}/qwen-arabic-kd/checkpoints/"* 2>/dev/null || true
df -h /data/datasets
