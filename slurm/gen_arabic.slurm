#!/bin/bash
#SBATCH --job-name=kd_arabic_data
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/arabic_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/arabic_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=HF_TOKEN_REMOVED

nvidia-smi
echo "Generating Arabic+English data at: $(date)"

python - << 'PYEOF'
import os, json, torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

TOKEN  = os.environ["HF_TOKEN"]
OUT    = "/data/datasets/user151/qwen-arabic-kd/data"
OUTFILE = f"{OUT}/teacher_responses_arabic.jsonl"
os.makedirs(OUT, exist_ok=True)

print("Loading Arabic + English datasets...")
# Arabic dataset
arabic = load_dataset("Helsinki-NLP/opus-100", "ar-en", split="train[:250]")
# English dataset  
english = load_dataset("tatsu-lab/alpaca", split="train[:250]")

print(f"Arabic samples: {len(arabic)}")
print(f"English samples: {len(english)}")

print("Loading Qwen-7B teacher in 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    token=TOKEN
)
print(f"Model on: {next(model.parameters()).device}")

def generate(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

results = []

# Generate Arabic responses
print("Generating Arabic responses...")
for i, ex in enumerate(tqdm(arabic, desc="Arabic")):
    prompt = ex.get("translation", {}).get("ar", "")
    if not prompt:
        continue
    response = generate(prompt)
    results.append({"instruction": prompt, "language": "ar", "teacher_response": response})
    if (i+1) % 50 == 0:
        with open(OUTFILE, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} samples")

# Generate English responses
print("Generating English responses...")
for i, ex in enumerate(tqdm(english, desc="English")):
    prompt = ex["instruction"]
    if ex.get("input"):
        prompt += "\n" + ex["input"]
    response = generate(prompt)
    results.append({"instruction": prompt, "language": "en", "teacher_response": response})
    if (i+1) % 50 == 0:
        with open(OUTFILE, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} samples")

with open(OUTFILE, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

ar_count = sum(1 for r in results if r["language"] == "ar")
en_count = sum(1 for r in results if r["language"] == "en")
print(f"DONE! Total: {len(results)} | Arabic: {ar_count} | English: {en_count}")
print(f"Saved to {OUTFILE}")
PYEOF
