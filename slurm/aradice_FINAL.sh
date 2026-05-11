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
    {"name": "TruthfulQA_msa", "dataset": ("QCRI/AraDiCE-TruthfulQA", "TruthfulQA-msa", "test"), "kind": "freeform"},
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

def mcq3(item, tok, model):
    q, A, B, C = item.get("Question", ""), item.get("Option A", ""), item.get("Option B", ""), item.get("Option C", "")
    if not all([q, A, B, C]): return "A", "", False
    prompt = f"{q}\nA) {A}\nB) {B}\nC) {C}\nأجب بحرف واحد فقط: A أو B أو C"
    pred = gen(tok, model, prompt)
    ok = norm(pred[:20]).startswith("a") or norm(pred[:20]).startswith("أ")
    return "A", pred, ok

def freeform(item, tok, model):
    q = item.get("question", "")
    gold = ""
    for key in ["correct_answers", "answer", "answers"]:
        if key in item:
            val = item[key]
            gold = val[0] if isinstance(val, list) else str(val)
            break
    if not q or not gold: return gold, "", False
    prompt = f"أجب بإجابة قصيرة ودقيقة:\n{q}"
    pred = gen(tok, model, prompt)
    pred_n = norm(pred)
    gold_n = norm(gold)
    if gold_n in pred_n: return gold, pred, True
    pw = set(pred_n.split()) - {'في', 'من', 'على', 'هو', 'هي', 'أن', 'إلى', 'و', 'ل', 'the', 'a', 'is', 'of', 'to', 'and'}
    gw = set(gold_n.split()) - {'في', 'من', 'على', 'هو', 'هي', 'أن', 'إلى', 'و', 'ل', 'the', 'a', 'is', 'of', 'to', 'and'}
    if not gw: return gold, pred, False
    return gold, pred, len(pw & gw) / len(gw) >= 0.25

def binary(item, tok, model):
    goal, sol1, sol2 = item.get("goal", ""), item.get("sol1", ""), item.get("sol2", "")
    if not all([goal, sol1, sol2]): return "1", "", False
    prompt = f"لتحقيق الهدف التالي، أي حل أفضل؟\nالهدف: {goal}\n1) {sol1}\n2) {sol2}\nأجب بـ 1 أو 2 فقط."
    pred = gen(tok, model, prompt)
    ok = norm(pred[:20]).startswith("1")
    return "1", pred, ok

def mcq4(item, tok, model):
    q = item.get("question", "")
    choices = item.get("choices", item.get("endings", []))[:4]
    gold = str(item.get("answerKey", item.get("label", item.get("answer", ""))))
    if not q or not choices or not gold: return gold, "", False
    prompt = f"{q}\n" + "\n".join([f"{chr(65+j)}) {c}" for j, c in enumerate(choices)]) + "\nأجب بحرف واحد فقط."
    pred = gen(tok, model, prompt)
    ok = norm(pred[:20])[:1] in ["a", "b", "c", "d"]
    return gold, pred, ok

eval_funcs = {"mcq3": mcq3, "freeform": freeform, "binary": binary, "mcq4": mcq4}

results = {}
with open(OUT_JSONL, "w", encoding="utf-8") as f_out:
    for model_name, ckpt in CKPTS.items():
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        if ckpt and not os.path.exists(os.path.join(ckpt, "adapter_config.json")):
            print("⚠️  Checkpoint missing")
            continue
        tok, model = load_model(ckpt)
        results[model_name] = {}
        for task in TASKS:
            ds = load_dataset(task["dataset"][0], task["dataset"][1], split=task["dataset"][2])
            ds = ds.select(range(min(500, len(ds))))
            correct = 0
            for i, item in enumerate(tqdm(ds, desc=task["name"], leave=False)):
                gold, pred, ok = eval_funcs[task["kind"]](item, tok, model)
                correct += int(ok)
                f_out.write(json.dumps({"model": model_name, "task": task["name"], "idx": i, "gold": gold[:100], "pred": pred[:100], "ok": int(ok)}, ensure_ascii=False) + "\n")
            acc = 100.0 * correct / len(ds)
            results[model_name][task["name"]] = {"acc": round(acc, 1), "n": len(ds)}
            print(f"  {task['name']:20s}: {acc:6.1f}%")
        del model, tok
        torch.cuda.empty_cache()

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved: {OUT_JSON}\n✅ Saved: {OUT_JSONL}")
PYEOF

chmod +x /data/datasets/user151/qwen-arabic-kd/slurm/aradice_FINAL.sh
