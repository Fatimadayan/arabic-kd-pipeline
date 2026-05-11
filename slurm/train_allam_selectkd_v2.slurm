#!/bin/bash
#SBATCH --job-name=allam_lasttry
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/allam_lasttry_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/allam_lasttry_%j.err
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

pip install tiktoken --quiet

python -c "
import torch, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

TOKEN = os.environ['HF_TOKEN']
ALLAM = 'humain-ai/ALLaM-7B-Instruct-preview'

print('Step 1: Loading tokenizer...')
try:
    tok = AutoTokenizer.from_pretrained(ALLAM, token=TOKEN, trust_remote_code=True)
    print('Tokenizer loaded (fast)')
except Exception as e:
    print(f'Fast tokenizer failed: {e}')
    try:
        tok = AutoTokenizer.from_pretrained(ALLAM, token=TOKEN, use_fast=False, trust_remote_code=True)
        print('Tokenizer loaded (slow)')
    except Exception as e2:
        print(f'Slow tokenizer also failed: {e2}')
        sys.exit(1)

print(f'Vocab size: {len(tok)}')
print('Step 2: Loading model...')
model = AutoModelForCausalLM.from_pretrained(ALLAM, torch_dtype=torch.bfloat16, device_map='auto', token=TOKEN, trust_remote_code=True)
print(f'Model loaded: {model.config.model_type}')
print('ALLaM is compatible')
"
