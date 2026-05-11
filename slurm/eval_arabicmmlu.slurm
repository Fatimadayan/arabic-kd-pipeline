#!/bin/bash
#SBATCH --job-name=kd_arabicmmlu
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/arabicmmlu_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/arabicmmlu_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

python - << 'PYEOF'
import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/arabicmmlu_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":      None,
    "sft_7b":       "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":    "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# Load ArabicMMLU — General Knowledge split, 100 samples
print("Loading ArabicMMLU...")
ds = load_dataset("MBZUAI/ArabicMMLU", "General Knowledge", split="test[:100]")
print(f"Loaded {len(ds)} samples")

# Show first item to verify
item0 = ds[0]
print(f"Sample question: {item0['Question'][:80]}")
print(f"Answer Key: {item0['Answer Key']}")
print(f"Options: {item0['Option 1']} / {item0['Option 2']} / {item0['Option 3']}")

def build_prompt(item):
    q = item["Question"]
    opts = []
    labels = ["A","B","C","D","E"]
    for i in range(1,6):
        opt = item.get(f"Option {i}")
        if opt is not None:
            opts.append(f"{labels[i-1]}) {opt}")
    opts_str = "\n".join(opts)
    return f"{q}\n{opts_str}\nأجب بحرف واحد فقط من: {', '.join(labels[:len(opts)])}"

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def generate(tok, model, prompt, max_new=10):
    msgs = [{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=768).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp.strip()

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print(f"SKIP — not found: {ckpt}")
        continue

    tok, model = load_model(ckpt)
    correct = 0
    details = []

    for item in tqdm(ds, desc=model_name):
        prompt = build_prompt(item)
        answer_key = item["Answer Key"].strip().upper()
        resp = generate(tok, model, prompt)
        # Check if correct letter appears at start of response
        resp_upper = resp.upper().strip()
        is_correct = resp_upper.startswith(answer_key) or f" {answer_key}" in resp_upper[:10]
        if is_correct:
            correct += 1
        details.append({
            "question": item["Question"][:100],
            "answer_key": answer_key,
            "response": resp[:50],
            "correct": is_correct
        })

    acc = round(correct/len(ds)*100, 1)
    print(f"  ArabicMMLU accuracy: {acc}% ({correct}/{len(ds)})")
    all_results[model_name] = {"accuracy": acc, "correct": correct, "total": len(ds)}

    del model
    torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nARABICMMLU RESULTS TABLE:")
print(f"{'Model':<16} {'ArabicMMLU':>11}")
print("-"*30)
for name, res in all_results.items():
    print(f"{name:<16} {res['accuracy']:>10}%")
print(f"\nResults saved to {OUT}")
PYEOF
