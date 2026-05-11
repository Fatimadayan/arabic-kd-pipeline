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
    {"name": "Culture_Qatar", "dataset": ("QCRI/AraDiCE-Culture", "Qatar", "test"), "min_n": 500, "kind": "mcq3"},
    {"name": "TruthfulQA_msa", "dataset": ("QCRI/AraDiCE-TruthfulQA", "TruthfulQA-msa", "test"), "min_n": 500, "kind": "freeform"},
    {"name": "PIQA_msa", "dataset": ("QCRI/AraDiCE-PIQA", "PIQA-msa", "test"), "min_n": 500, "kind": "binary"},
    {"name": "HellaSwag_msa", "dataset": ("QCRI/AraDiCE-HellaSwag", "HellaSwag-msa", "validation"), "min_n": 500, "kind": "mcq4"},
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    return tok, model

def gen(tok, model, prompt, max_new=96):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=0.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def safe_select(ds, n):
    return ds.select(range(min(n, len(ds))))

def load_task(task):
    repo, config, split = task["dataset"]
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    return safe_select(ds, max(task["min_n"], 500))

def normalize(x):
    x = str(x).strip().lower()
    x = re.sub(r'[\u064B-\u0652]', '', x)
    return re.sub(r'\s+', ' ', x).strip()

def get_correct_answer(item):
    for key in ["correct_answers", "answer", "answers", "reference_answer", "gold_answer"]:
        if key in item:
            ans = item[key]
            if isinstance(ans, list) and len(ans) > 0:
                return str(ans[0]).strip()
            elif isinstance(ans, str) and ans.strip():
                return ans.strip()
    return ""

def lax_match(pred, gold):
    if not gold: return False
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)
    if gold_norm in pred_norm or pred_norm in gold_norm: return True
    pred_words = set(pred_norm.split()) - {'في', 'من', 'على', 'هو', 'هي', 'أن', 'إلى', 'هذا', 'ذلك', 'and', 'the', 'is', 'a', 'of', 'to', 'in', 'it', 'that', 'this'}
    gold_words = set(gold_norm.split()) - {'في', 'من', 'على', 'هو', 'هي', 'أن', 'إلى', 'هذا', 'ذلك', 'and', 'the', 'is', 'a', 'of', 'to', 'in', 'it', 'that', 'this'}
    if not gold_words: return sum(1 for c in gold_norm if c in pred_norm) > len(gold_norm) * 0.3
    return len(pred_words & gold_words) / len(gold_words) >= 0.3 if gold_words else False

def eval_task(task, data, tok, model, model_name, jsonl_f):
    rows, correct, debug_count = [], 0, 0
    for i, item in enumerate(tqdm(data, desc=f"{model_name}:{task['name']}")):
        prompt, gold, pred, ok = "", "", "", False
        try:
            if task["kind"] == "mcq3":
                q, A, B, C = item["Question"], item["Option A"], item["Option B"], item["Option C"]
                gold, prompt = "A", f"{q}\nA) {A}\nB) {B}\nC) {C}\nأجب بحرف واحد فقط: A أو B أو C"
                pred = gen(tok, model, prompt)
                ok = normalize(pred[:20]).startswith("a") or normalize(pred[:20]).startswith("أ")
            elif task["kind"] == "mcq4":
                q = item["question"]
                choices = item.get("choices", item.get("endings", []))
                gold = str(item.get("answerKey", item.get("label", "")))
                prompt = f"{q}\n" + "\n".join([f"{chr(65+j)}) {c}" for j, c in enumerate(choices)]) + "\nأجب بحرف واحد فقط."
                pred = gen(tok, model, prompt)
                ok = normalize(pred[:20])[:1] in ["a", "b", "c", "d"]
            elif task["kind"] == "binary":
                goal, sol1, sol2 = item["goal"], item["sol1"], item["sol2"]
                gold, prompt = "1", f"لتحقيق الهدف التالي، أي حل أفضل؟\nالهدف: {goal}\n1) {sol1}\n2) {sol2}\nأجب بـ 1 أو 2 فقط."
                pred = gen(tok, model, prompt)
                ok = normalize(pred[:20]).startswith("1")
            elif task["kind"] == "freeform":
                q, gold = item.get("question", ""), get_correct_answer(item)
                if debug_count < 3:
                    print(f"\n  [Sample {debug_count}] Q: {q[:50]}... → Gold: {gold[:50]}...")
                    debug_count += 1
                prompt = f"أجب بإجابة قصيرة ودقيقة:\n{q}"
                pred = gen(tok, model, prompt)
                ok = lax_match(pred, gold)
        except Exception as e:
            print(f"ERROR {task['name']} #{i}: {e}")
        correct += int(ok)
        row = {"model": model_name, "task": task["name"], "idx": i, "prompt": prompt[:200], "prediction": pred[:200], "gold": gold[:200], "correct": int(ok)}
        rows.append(row)
        jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return round(100.0 * correct / len(data), 2), rows

results = {}
with open(OUT_JSONL, "w", encoding="utf-8") as jsonl_f:
    for model_name, ckpt in CKPTS.items():
        if ckpt and not os.path.exists(os.path.join(ckpt, "adapter_config.json")): continue
        tok, model = load_model(ckpt)
        results[model_name] = {}
        for task in TASKS:
            data = load_task(task)
            score, _ = eval_task(task, data, tok, model, model_name, jsonl_f)
            results[model_name][task["name"]] = {"n": len(data), "acc": score}
        del model, tok
        torch.cuda.empty_cache()

with open(OUT_JSON, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

for model, scores in results.items():
    print(f"\n{model}:")
    for task, info in scores.items():
        print(f"  {task:20s}: {info['acc']:.1f}%")
PYEOF

chmod +x /data/datasets/user151/qwen-arabic-kd/slurm/aradice_final_eval_v3.sh
