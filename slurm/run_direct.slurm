#!/bin/bash
#SBATCH --job-name=kd_direct
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/direct_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/direct_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export TRANSFORMERS_CACHE=/data/datasets/user151/.cache/huggingface/transformers
export HF_TOKEN=PASTE_NEW_TOKEN_HERE
export SCRATCH=/data/datasets/user151
export OUTPUT=/data/datasets/user151/qwen-arabic-kd/data

mkdir -p $OUTPUT

nvidia-smi
echo "Token starts with: ${HF_TOKEN:0:10}..."
echo "Starting at: $(date)"

python - << 'PYEOF'
import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm

OUTPUT = "/data/datasets/user151/qwen-arabic-kd/data"
os.makedirs(OUTPUT, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    token=os.environ["HF_TOKEN"]
)

print("Loading model in 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    token=os.environ["HF_TOKEN"]
)
print(f"Model loaded on: {next(model.parameters()).device}")

print("Loading dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")
print(f"Dataset size: {len(dataset)}")

results = []
for i, ex in enumerate(tqdm(dataset, desc="Generating")):
    prompt = ex["instruction"]
    if ex.get("input"):
        prompt += "\n" + ex["input"]
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results.append({"instruction": prompt, "teacher_response": response})
    
    if (i+1) % 50 == 0:
        with open(f"{OUTPUT}/teacher_responses_partial.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {i+1} samples")

with open(f"{OUTPUT}/teacher_responses.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"DONE! Saved {len(results)} samples to {OUTPUT}/teacher_responses.jsonl")
PYEOF
