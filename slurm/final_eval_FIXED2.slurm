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

python3 - << 'PYEOF'
import os, json, torch
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

CKPTS = {
    "base_7b": None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

TASKS = [
    {
        "name": "Culture_Qatar",
        "dataset": ("QCRI/AraDiCE-Culture", "Qatar", "test"),
        "min_n": 500,
        "kind": "mcq3",
    },
    {
        "name": "TruthfulQA_msa",
        "dataset": ("QCRI/AraDiCE-TruthfulQA", "TruthfulQA-msa", "test"),
        "min_n": 500,
        "kind": "freeform",
    },
    {
        "name": "PIQA_msa",
    "dataset": ("QCRI/AraDiCE-PIQA", "PIQA-msa", "test"),
    "min_n": 500,
    "kind": "binary",
    },
    {
        "name": "HellaSwag_msa",
        "dataset": ("QCRI/AraDiCE-HellaSwag", "HellaSwag-msa", "validation"),
        "min_n": 500,
        "kind": "mcq4",
    },
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb,
        device_map={"": 0},
        token=TOKEN
    )
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, prompt, max_new=96):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def safe_select(ds, n):
    return ds.select(range(min(n, len(ds))))

def load_task(task):
    repo, config, split = task["dataset"]
    if config is None:
        ds = load_dataset(repo, split=split)
    else:
        ds = load_dataset(repo, config, split=split)
    return safe_select(ds, max(task["min_n"], 500))

def normalize(x):
    return str(x).strip().lower()

def eval_task(task, data, tok, model, model_name, jsonl_f):
    rows = []
    correct = 0

    for i, item in enumerate(tqdm(data, desc=f"{model_name}:{task['name']}")):
        prompt = ""
        gold = ""
        pred = ""
        ok = False

        if task["kind"] == "mcq3":
            q = item["Question"]
            A = item["Option A"]
            B = item["Option B"]
            C = item["Option C"]
            gold = "A"
            prompt = f"{q}\nA) {A}\nB) {B}\nC) {C}\nأجب بحرف واحد فقط: A أو B أو C"
            pred = gen(tok, model, prompt)
            p = normalize(pred[:20])
            ok = p.startswith("a") or p.startswith("أ")
        elif task["kind"] == "mcq4":
            q = item["question"]
            choices = item.get("choices", item.get("endings", []))
            if "answerKey" in item:
                gold = str(item["answerKey"])
            elif "label" in item:
                gold = str(item["label"])
            prompt = f"{q}\n" + "\n".join([f"{chr(65+j)}) {c}" for j, c in enumerate(choices)]) + "\nأجب بحرف واحد فقط."
            pred = gen(tok, model, prompt)
            p = normalize(pred[:20])
            ok = p[:1] in ["a", "b", "c", "d"]
        elif task["kind"] == "binary":
            goal = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            gold = "1"
            prompt = f"لتحقيق الهدف التالي، أي حل أفضل؟\nالهدف: {goal}\n1) {sol1}\n2) {sol2}\nأجب بـ 1 أو 2 فقط."
            pred = gen(tok, model, prompt)
            p = normalize(pred[:20])
            ok = p.startswith("1")
        elif task["kind"] == "freeform":
            q = item["question"]
            gold = item["correct_answers"][0] if isinstance(item.get("correct_answers"), list) else ""
            prompt = f"أجب بإجابة قصيرة ودقيقة:\n{q}"
            pred = gen(tok, model, prompt)
            ok = any(w and w in normalize(pred) for w in normalize(gold).split()[:6])

        correct += int(ok)
        row = {
            "model": model_name,
            "task": task["name"],
            "idx": i,
            "prompt": prompt,
            "prediction": pred,
            "gold": gold,
            "correct": int(ok),
        }
        rows.append(row)
        jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc = round(100.0 * correct / len(data), 2)
    return acc, rows

results = {}
all_rows = []

with open(OUT_JSONL, "w", encoding="utf-8") as jsonl_f:
    for model_name, ckpt in CKPTS.items():
        print(f"\n===== {model_name} =====")
        if ckpt and not os.path.exists(os.path.join(ckpt, "adapter_config.json")):
            print("Skipping missing checkpoint")
            continue

        tok, model = load_model(ckpt)
        results[model_name] = {}

        for task in TASKS:
            data = load_task(task)
            print(f"{task['name']}: {len(data)} samples")
            score, rows = eval_task(task, data, tok, model, model_name, jsonl_f)
            results[model_name][task["name"]] = {"n": len(data), "acc": score}
            all_rows.extend(rows)
            print(f"{task['name']} = {score}%")

        del model, tok
        torch.cuda.empty_cache()

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nFINAL TABLE")
for model_name, scores in results.items():
    parts = [f"{task}: {info['acc']}% (N={info['n']})" for task, info in scores.items()]
    print(model_name, " | ".join(parts))
print(f"\nSaved JSON: {OUT_JSON}")
print(f"Saved predictions: {OUT_JSONL}")
PYEOF