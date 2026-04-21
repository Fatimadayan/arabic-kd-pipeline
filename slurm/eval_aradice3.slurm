#!/bin/bash
#SBATCH --job-name=kd_aradice3
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/aradice3_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/aradice3_%j.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

python - << 'PYEOF'
import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/aradice_final.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":     None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# Load AraDiCE-Culture Qatar — Option A is always correct (Qatar-specific)
print("Loading AraDiCE-Culture Qatar...")
qatar = load_dataset("QCRI/AraDiCE-Culture", "Qatar", split="test")
print(f"Qatar: {len(qatar)} samples")
print(f"Sample: {qatar[0]['Question'][:60]}")
print(f"Option A (correct): {qatar[0]['Option A'][:60]}")

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, q, max_new=10):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=768).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             output_scores=True, return_dict_in_generate=True)
    resp = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    conf = sum(F.softmax(s,dim=-1).max().item() for s in out.scores)/max(len(out.scores),1)
    return resp.strip(), conf

def eval_aradice(tok, model, data):
    correct = 0
    details = []
    for item in tqdm(data, desc="AraDiCE-Qatar"):
        q = item["Question"]
        opt_a = item["Option A"]
        opt_b = item["Option B"]
        opt_c = item["Option C"]
        # Build MCQ prompt
        prompt = (f"{q}\n"
                  f"أ) {opt_a[:80]}\n"
                  f"ب) {opt_b[:80]}\n"
                  f"ج) {opt_c[:80]}\n"
                  f"أجب بـ أ أو ب أو ج فقط.")
        resp, conf = gen(tok, model, prompt)
        # Option A is the Qatar-correct answer
        is_correct = resp.strip().startswith("أ") or "أ)" in resp[:10] or resp.strip() == "A"
        if is_correct:
            correct += 1
        details.append({
            "question": q[:80],
            "correct_option": opt_a[:60],
            "response": resp[:30],
            "correct": is_correct,
            "confidence": round(conf, 4)
        })
    return round(correct/len(data)*100, 1), details

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue

    tok, model = load_model(ckpt)
    score, details = eval_aradice(tok, model, qatar)
    print(f"  AraDiCE-Culture Qatar: {score}%")

    all_results[model_name] = {
        "aradice_qatar": score,
        "n_samples": len(qatar),
        "details": details[:5]
    }
    del model; torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nARaDiCE-Culture Qatar RESULTS:")
print(f"{'Model':<16} {'AraDiCE-Qatar%':>15}")
print("-"*35)
for name, res in all_results.items():
    print(f"{name:<16} {str(res['aradice_qatar'])+'%':>15}")
print(f"\nSaved to {OUT}")
PYEOF
