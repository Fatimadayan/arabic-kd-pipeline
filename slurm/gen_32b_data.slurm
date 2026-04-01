#!/bin/bash
#SBATCH --job-name=kd_32b_data
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/gen_32b_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/gen_32b_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02


#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=HF_TOKEN_REMOVED
export PYTORCH_ALLOC_CONF=expandable_segments:True

nvidia-smi
echo "Starting 32B data generation at: $(date)"

python - << 'PYEOF'
import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

TOKEN   = os.environ["HF_TOKEN"]
OUT     = "/data/datasets/user151/qwen-arabic-kd/data"
OUTFILE = f"{OUT}/teacher_32b_responses.jsonl"
os.makedirs(OUT, exist_ok=True)

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")

# Arabic instructions — proper instruction dataset
arabic_ds = load_dataset("Helsinki-NLP/opus-100", "ar-en", split="train[:2500]")
print(f"Arabic samples: {len(arabic_ds)}")

# English instructions
english_ds = load_dataset("tatsu-lab/alpaca", split="train[:2500]")
print(f"English samples: {len(english_ds)}")

# ── Load 32B teacher in 4-bit ─────────────────────────────────────────────────
print("Loading Qwen2.5-32B in 4-bit (this takes ~10 minutes first time)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct", token=TOKEN
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    token=TOKEN
)
model.eval()
print(f"32B model loaded on: {next(model.parameters()).device}")

def generate(prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=1024).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

def save(results):
    with open(OUTFILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

results = []

# ── Generate Arabic responses ─────────────────────────────────────────────────
print("\nGenerating Arabic responses from 32B teacher...")
for i, ex in enumerate(tqdm(arabic_ds, desc="Arabic")):
    prompt = ex.get("translation", {}).get("ar", "")
    if not prompt or len(prompt.strip()) < 5:
        continue
    try:
        response = generate(prompt)
        results.append({
            "instruction": prompt,
            "language": "ar",
            "teacher_model": "Qwen2.5-32B-Instruct",
            "teacher_response": response
        })
    except Exception as e:
        print(f"Error at {i}: {e}")
        continue

    if (i+1) % 100 == 0:
        save(results)
        print(f"Saved {len(results)} samples so far")

# ── Generate English responses ────────────────────────────────────────────────
print("\nGenerating English responses from 32B teacher...")
for i, ex in enumerate(tqdm(english_ds, desc="English")):
    prompt = ex["instruction"]
    if ex.get("input"):
        prompt += "\n" + ex["input"]
    if not prompt or len(prompt.strip()) < 5:
        continue
    try:
        response = generate(prompt)
        results.append({
            "instruction": prompt,
            "language": "en",
            "teacher_model": "Qwen2.5-32B-Instruct",
            "teacher_response": response
        })
    except Exception as e:
        print(f"Error at {i}: {e}")
        continue

    if (i+1) % 100 == 0:
        save(results)
        print(f"Saved {len(results)} samples so far")

# ── Final save ────────────────────────────────────────────────────────────────
save(results)
ar  = sum(1 for r in results if r["language"] == "ar")
en  = sum(1 for r in results if r["language"] == "en")
print(f"\nDONE! Total: {len(results)} | Arabic: {ar} | English: {en}")
print(f"Saved to: {OUTFILE}")
PYEOF
