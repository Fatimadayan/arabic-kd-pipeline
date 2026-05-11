#!/bin/bash
#SBATCH --job-name=aradice_safe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/aradice_safe_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/aradice_safe_%j.err

set -e  # Exit on any error
export CUDA_VISIBLE_DEVICES=0

echo "Starting safe evaluation at $(date)"
source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env || { echo "Conda failed"; exit 1; }

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Anti-OOM

python3 - << 'PYEOF'
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

print("Python OK, imports starting...")
try:
    TOKEN = os.environ["HF_TOKEN"]
    print(f"HF_TOKEN: {'SET' if TOKEN else 'MISSING'}")
    
    # Test datasets first
    print("Testing AraDiCE-Culture Qatar...")
    data = load_dataset("QCRI/AraDiCE-Culture", "Qatar", split="test")
    print(f"Loaded {len(data)} samples, keys: {data.column_names}")
    
    print("SUCCESS - All tests passed")
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYEOF

echo "Job completed successfully at $(date)"
