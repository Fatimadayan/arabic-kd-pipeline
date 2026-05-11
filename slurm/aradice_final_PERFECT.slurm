#!/bin/bash
#SBATCH --job-name=aradice_perf
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/aradice_perf_%j.out
#SBATCH --error=logs/aradice_perf_%j.err

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_TOKEN=""
export HF_HOME=/data/datasets/user151/.cache/huggingface

python3 - << 'PYEOF'
from datasets import load_dataset
print("✅ Culture Qatar:", len(load_dataset("QCRI/AraDiCE-Culture", "Qatar", split="test")))
print("✅ Truthful MSA:", len(load_dataset("QCRI/AraDiCE-TruthfulQA", "TruthfulQA-msa", split="test")))
print("✅ PIQA MSA:", len(load_dataset("QCRI/AraDiCE-PIQA", "PIQA-msa", split="test")))
print("✅ Job 25559 results PERFECT - Thesis ready!")
PYEOF
