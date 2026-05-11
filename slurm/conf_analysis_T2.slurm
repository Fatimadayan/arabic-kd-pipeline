#!/bin/bash
#SBATCH --job-name=kd_conf_T2
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/conf_T2_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/conf_T2_%j.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

python - << 'PYEOF'
import os, json, torch, numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
DATA  = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/confidence_analysis_T2.json"

with open(DATA) as f:
    rows = [json.loads(l) for l in f]
ar_rows = [r for r in rows if r.get("language","en")=="ar"][:50]
en_rows = [r for r in rows if r.get("language","en")=="en"][:50]
print(f"AR: {len(ar_rows)}, EN: {len(en_rows)}")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN)
model.eval()

def get_confs(row, T=1.0, max_len=256):
    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to("cuda:0")
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0]
    confs  = F.softmax(logits/T, dim=-1).max(dim=-1).values
    return confs.cpu().tolist()

results = {}
for T in [1.0, 2.0, 4.0]:
    ar_all, en_all = [], []
    for row in tqdm(ar_rows, desc=f"T={T} AR"):
        try: ar_all.extend(get_confs(row, T))
        except: pass
    for row in tqdm(en_rows, desc=f"T={T} EN"):
        try: en_all.extend(get_confs(row, T))
        except: pass

    ar = np.array(ar_all); en = np.array(en_all)
    theta = 0.7
    ar_sel = (ar > theta).mean()*100
    en_sel = (en > theta).mean()*100
    ratio  = en_sel / max(ar_sel, 0.01)

    results[f"T{T}"] = {
        "AR_mean": round(float(ar.mean()*100),2),
        "EN_mean": round(float(en.mean()*100),2),
        "AR_sel_pct": round(ar_sel,2),
        "EN_sel_pct": round(en_sel,2),
        "ratio_EN_over_AR": round(ratio,2),
        "mean_gap": round(float((en.mean()-ar.mean())*100),2)
    }
    print(f"\nT={T}: AR_sel={ar_sel:.1f}% EN_sel={en_sel:.1f}% ratio={ratio:.2f}x gap={results[f'T{T}']['mean_gap']}%")

with open(OUT,"w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("TEMPERATURE EFFECT ON CONFIDENCE GAP:")
print(f"{'T':<6} {'AR sel%':>9} {'EN sel%':>9} {'Ratio':>8} {'Gap%':>8}")
print("-"*45)
for k,v in results.items():
    print(f"{k:<6} {v['AR_sel_pct']:>8}% {v['EN_sel_pct']:>8}% {v['ratio_EN_over_AR']:>7}x {v['mean_gap']:>7}%")
print(f"\nKey: T=2 matches SelecTKD training configuration")
print(f"Saved to {OUT}")
PYEOF
