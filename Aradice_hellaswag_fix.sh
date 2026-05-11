#!/bin/bash
#SBATCH --job-name=aradice_final_eval
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/aradice_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/aradice_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

python3 << 'PYEOF'
import os, json, torch, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = "/data/datasets/user151/qwen-arabic-kd/results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "aradice_full_eval.json")
OUT_JSONL = os.path.join(OUT_DIR, "aradice_full_predictions.jsonl")

CKPTS = {"base_7b": None, "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b", "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b"}
TASKS = [
    {"name": "Culture_Qatar", "dataset": ("QCRI/AraDiCE-Culture", "Qatar", "test"), "kind": "mcq3"},
    {"name": "PIQA_msa", "dataset": ("QCRI/AraDiCE-PIQA", "PIQA-msa", "test"), "kind": "binary"},
    {"name": "HellaSwag_msa", "dataset": ("QCRI/AraDiCE-HellaSwag", "HellaSwag-msa", "validation"), "kind": "mcq4"},
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    return tok, model

def gen(tok, model, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False, temperature=0.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r'[\u064B-\u0652]', '', x)
    return re.sub(r'\s+', ' ', x)

def extract_letter(pred_text):
    text = pred_text.lower().strip()
    # 1. Direct match at the beginning
    if text.startswith(('a)', 'a ', 'أ')): return 'a'
    if text.startswith(('b)', 'b ', 'ب')): return 'b'
    if text.startswith(('c)', 'c ', 'ج')): return 'c'
    if text.startswith(('d)', 'd ', 'd')): return 'd'
    
    # 2. Search for the label in the first 50 chars
    search_space = text[:50]
    for char in ['a', 'b', 'c', 'd']:
        if re.search(fr'\b(option\s+)?{char}\b', search_space): return char
    
    # 3. Last-ditch: find the first occurrence of a letter
    for char in ['a', 'b', 'c', 'd']:
        if char in text[:10]: return char
    return None

def eval_culture_qatar(item, tok, model):
    q, A, B, C = item.get("Question", ""), item.get("Option A", ""), item.get("Option B", ""), item.get("Option C", "")
    if not all([q, A, B, C]): return "A", "", False
    prompt = f"{q}\nA) {A}\nB) {B}\nC) {C}\nأجب بحرف واحد فقط: A أو B أو C"
    pred = gen(tok, model, prompt)
    p = norm(pred[:20])
    ok = p.startswith("a") or p.startswith("أ")
    return "A", pred, ok

def eval_piqa(item, tok, model):
    goal, sol1, sol2 = item.get("goal", ""), item.get("sol1", ""), item.get("sol2", "")
    if not all([goal, sol1, sol2]): return "1", "", False
    prompt = f"لتحقيق الهدف التالي، أي حل أفضل؟\nالهدف: {goal}\n1) {sol1}\n2) {sol2}\nأجب بـ 1 أو 2 فقط."
    pred = gen(tok, model, prompt)
    ok = norm(pred[:20]).startswith("1")
    return "1", pred, ok

def eval_hellaswag(item, tok, model):
    ctx = item.get("ctx", "")
    endings = item.get("endings", [])
    # 1. Get the gold label as an integer
    try:
        gold_label = int(item.get("label", -1))
    except:
        return "", "", False
    
    if not ctx or not endings or gold_label < 0 or gold_label > 3:
        return "", "", False
    
    # 2. Prompt for a numeric answer (0-3)
    prompt = f"{ctx}\n" + "\n".join([f"{i}) {endings[i]}" for i in range(4)]) + "\nأجب برقم الخيار الصحيح فقط (0 أو 1 أو 2 أو 3):"
    
    # 3. Generate the response
    pred = gen(tok, model, prompt)
    
    # 4. Extract the last digit 0-3 found in the prediction
    # This is highly robust against "The answer is 1" or "1)"
    digits = re.findall(r'\b[0-3]\b', pred)
    pred_label = int(digits[-1]) if digits else -1
    
    # 5. Compare numeric labels
    return str(gold_label), pred, (pred_label == gold_label)
    
eval_funcs = {"mcq3": eval_culture_qatar, "binary": eval_piqa, "mcq4": eval_hellaswag}
results = {}

with open(OUT_JSONL, "w", encoding="utf-8") as jsonl_f:
    for model_name, ckpt in CKPTS.items():
        print(f"\n{'='*70}\nMODEL: {model_name}\n{'='*70}")
        if ckpt and not os.path.exists(os.path.join(ckpt, "adapter_config.json")):
            print("⚠️  Checkpoint not found"); continue
        tok, model = load_model(ckpt)
        results[model_name] = {}
        for task in TASKS:
            print(f"\n{task['name']}:", end=" ", flush=True)
            ds = load_dataset(task["dataset"][0], task["dataset"][1], split=task["dataset"][2])
            ds = ds.select(range(min(500, len(ds))))
            correct = 0
            for i, item in enumerate(tqdm(ds, desc=task["name"], leave=False)):
                try:
                    gold, pred, ok = eval_funcs[task["kind"]](item, tok, model)
                    correct += int(ok)
                    jsonl_f.write(json.dumps({"model": model_name, "task": task["name"], "idx": i, "gold": gold[:100], "pred": pred[:100], "ok": int(ok)}, ensure_ascii=False) + "\n")
                except:
                    jsonl_f.write(json.dumps({"model": model_name, "task": task["name"], "idx": i, "gold": "", "pred": "", "ok": 0}, ensure_ascii=False) + "\n")
            acc = 100.0 * correct / len(ds)
            results[model_name][task["name"]] = {"acc": round(acc, 1), "n": len(ds)}
            print(f"✓ {acc:.1f}%")
        del model, tok
        torch.cuda.empty_cache()

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*70 + "\nFINAL RESULTS\n" + "="*70)
for model_name in CKPTS.keys():
    if model_name in results:
        print(f"\n{model_name}:")
        for task in TASKS:
            if task["name"] in results[model_name]:
                info = results[model_name][task["name"]]
                print(f"  {task['name']:20s}: {info['acc']:6.1f}% (N={info['n']})")
print(f"\n✅ Saved: {OUT_JSON}\n✅ Saved: {OUT_JSONL}")
PYEOF
