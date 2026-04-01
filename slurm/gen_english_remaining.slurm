#!/bin/bash
#SBATCH --job-name=kd_en_remaining
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/en_remaining_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/en_remaining_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=HF_TOKEN_REMOVED

nvidia-smi
echo "Generating remaining English samples at: $(date)"

python - << 'PYEOF'
import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

TOKEN   = os.environ["HF_TOKEN"]
OUTFILE = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"

# Load existing data to find which English samples already done
print("Loading existing data...")
existing = []
with open(OUTFILE, encoding="utf-8") as f:
    for line in f:
        existing.append(json.loads(line))

existing_en = [r for r in existing if r["language"] == "en"]
done_instructions = set(r["instruction"][:50] for r in existing_en)
print(f"Already have: {len(existing_en)} English samples")
print(f"Need: {2500 - len(existing_en)} more")

# Load full English dataset
english_ds = load_dataset("tatsu-lab/alpaca", split="train[:2500]")

# Filter to only missing ones
missing = []
for ex in english_ds:
    prompt = ex["instruction"]
    if ex.get("input"):
        prompt += "\n" + ex["input"]
    if prompt[:50] not in done_instructions:
        missing.append(prompt)

print(f"Missing samples to generate: {len(missing)}")

if len(missing) == 0:
    print("Nothing to generate — already complete!")
    exit(0)

# Load 32B model
print("Loading Qwen2.5-32B in 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    token=TOKEN
)
model.eval()
print(f"Model loaded!")

def generate(prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt",
                      truncation=True, max_length=1024).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

# Generate missing samples and APPEND to existing file
new_results = []
for i, prompt in enumerate(tqdm(missing, desc="English remaining")):
    try:
        response = generate(prompt)
        new_results.append({
            "instruction": prompt,
            "language": "en",
            "teacher_model": "Qwen2.5-32B-Instruct",
            "teacher_response": response
        })
    except Exception as e:
        print(f"Error at {i}: {e}")
        continue

    if (i+1) % 50 == 0:
        with open(OUTFILE, "a", encoding="utf-8") as f:
            for r in new_results[-50:]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Appended {i+1} new samples")

# Append any remaining
remainder = len(new_results) % 50
if remainder > 0:
    with open(OUTFILE, "a", encoding="utf-8") as f:
        for r in new_results[-remainder:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

total = len(existing) + len(new_results)
print(f"\nDONE!")
print(f"Previous: {len(existing)} | New: {len(new_results)} | Total: {total}")
PYEOF
