#!/bin/bash
#SBATCH --job-name=kd_conf_analysis
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/conf_analysis_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/conf_analysis_%j.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

python - << 'PYEOF'
import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
DATA  = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/confidence_analysis.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

print("Loading data...")
with open(DATA) as f:
    rows = [json.loads(l) for l in f]
ar_rows = [r for r in rows if r.get("language","en") == "ar"][:50]
en_rows = [r for r in rows if r.get("language","en") == "en"][:50]
print(f"AR samples: {len(ar_rows)}, EN samples: {len(en_rows)}")

print("Loading teacher model...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
)
model.eval()

def get_token_confidences(row, max_len=256):
    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
    inputs = tok(text, return_tensors="pt", truncation=True,
                 max_length=max_len, padding=False).to("cuda:0")
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0]  # [seq_len, vocab]
    confs  = F.softmax(logits, dim=-1).max(dim=-1).values
    return confs.cpu().tolist()

print("Computing token confidences...")
ar_confs_all, en_confs_all = [], []

for row in tqdm(ar_rows, desc="Arabic"):
    try:
        c = get_token_confidences(row)
        ar_confs_all.extend(c)
    except: pass

for row in tqdm(en_rows, desc="English"):
    try:
        c = get_token_confidences(row)
        en_confs_all.extend(c)
    except: pass

# Statistics
import numpy as np
ar = np.array(ar_confs_all)
en = np.array(en_confs_all)

T = 0.7
ar_sel = (ar > T).mean() * 100
en_sel = (en > T).mean() * 100

results = {
    "arabic": {
        "n_tokens": len(ar),
        "mean_conf": round(float(ar.mean()*100), 2),
        "median_conf": round(float(np.median(ar)*100), 2),
        "std_conf": round(float(ar.std()*100), 2),
        "pct_above_07": round(ar_sel, 2),
        "pct_above_05": round((ar > 0.5).mean()*100, 2),
        "pct_above_09": round((ar > 0.9).mean()*100, 2),
    },
    "english": {
        "n_tokens": len(en),
        "mean_conf": round(float(en.mean()*100), 2),
        "median_conf": round(float(np.median(en)*100), 2),
        "std_conf": round(float(en.std()*100), 2),
        "pct_above_07": round(en_sel, 2),
        "pct_above_05": round((en > 0.5).mean()*100, 2),
        "pct_above_09": round((en > 0.9).mean()*100, 2),
    },
    "gap": {
        "mean_conf_gap": round(float((en.mean()-ar.mean())*100), 2),
        "selection_gap_at_07": round(en_sel - ar_sel, 2),
        "ratio_en_over_ar": round(en_sel / max(ar_sel, 0.01), 2),
    }
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*50}")
print("CONFIDENCE DISTRIBUTION ANALYSIS")
print(f"{'='*50}")
print(f"{'Metric':<30} {'Arabic':>10} {'English':>10} {'Gap':>10}")
print("-"*65)
print(f"{'Mean confidence':<30} {results['arabic']['mean_conf']:>9}% {results['english']['mean_conf']:>9}% {results['gap']['mean_conf_gap']:>9}%")
print(f"{'Median confidence':<30} {results['arabic']['median_conf']:>9}% {results['english']['median_conf']:>9}%")
print(f"{'% tokens above 0.5':<30} {results['arabic']['pct_above_05']:>9}% {results['english']['pct_above_05']:>9}%")
print(f"{'% tokens above 0.7 (selected)':<30} {results['arabic']['pct_above_07']:>9}% {results['english']['pct_above_07']:>9}% {results['gap']['selection_gap_at_07']:>9}%")
print(f"{'% tokens above 0.9':<30} {results['arabic']['pct_above_09']:>9}% {results['english']['pct_above_09']:>9}%")
print(f"\nEN/AR selection ratio at θ=0.7: {results['gap']['ratio_en_over_ar']}×")
print(f"\nKEY FINDING FOR PAPER:")
print(f"Arabic tokens are {results['gap']['ratio_en_over_ar']}× less likely to exceed θ=0.7")
print(f"Mean confidence gap: {results['gap']['mean_conf_gap']}% (EN higher than AR)")
print(f"\nSaved to {OUT}")
PYEOF
