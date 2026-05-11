#!/bin/bash
#SBATCH --job-name=kd_final_eval
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/final_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/final_%j.err
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

# ================= CONFIG =================
TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/final_eval.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CKPTS = {
    "base_7b": None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# ================= LOAD DATA =================
print("Loading AraDiCE-Culture Qatar...")
qatar = load_dataset("QCRI/AraDiCE-Culture", "Qatar", split="test")
print(f"Loaded {len(qatar)} samples")

# ================= MODEL =================
def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)

    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb,
        device_map={"":0},
        token=TOKEN
    )

    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

# ================= GENERATION =================
def gen(tok, model, q, max_new=10):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=768).to("cuda:0")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    resp = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    conf = sum(F.softmax(s, dim=-1).max().item() for s in out.scores) / max(len(out.scores), 1)
    return resp.strip(), conf

# ================= EVALUATION =================

# ---- 1. AraDiCE MCQ (Publication Standard) ----
def eval_aradice_mcq(tok, model, data):
    correct = 0

    for item in tqdm(data, desc="AraDiCE-MCQ"):
        q = item["Question"]
        A = item["Option A"]
        B = item["Option B"]
        C = item["Option C"]

        prompt = (
            f"{q}\n"
            f"أ) {A[:80]}\n"
            f"ب) {B[:80]}\n"
            f"ج) {C[:80]}\n"
            f"أجب بحرف واحد فقط: أ أو ب أو ج"
        )

        resp, _ = gen(tok, model, prompt)

        if resp.strip().startswith("أ") or resp.strip() == "A":
            correct += 1

    return round(correct / len(data) * 100, 1)

# ---- 2. Custom Bahraini Evaluation ----
CUSTOM = [
    {"q":"شلونك اليوم؟","kw":["بخير","زين","تمام"]},
    {"q":"وايد حار اليوم","kw":["حار","جو","صيف"]},
    {"q":"الحين بروح السوق","kw":["أذهب","سوق","شراء"]},
    {"q":"اشرح معنى 'صج'","kw":["حقا","صحيح","فعلا"]},
    {"q":"حول للفصحى: وايد زين","kw":["جيد","ممتاز"]},
    {"q":"The meeting وايد مهم","kw":["اجتماع","مهم"]},
]

def eval_custom(tok, model):
    ok = 0

    for item in tqdm(CUSTOM, desc="Bahraini"):
        resp, _ = gen(tok, model, item["q"])
        if any(k in resp for k in item["kw"]):
            ok += 1

    return round(ok / len(CUSTOM) * 100, 1)

# ================= RUN =================
results = {}

for name, ckpt in CKPTS.items():
    print(f"\n===== {name} =====")

    if ckpt and not os.path.exists(ckpt + "/adapter_config.json"):
        print("Skipping missing checkpoint")
        continue

    tok, model = load_model(ckpt)

    aradice_score = eval_aradice_mcq(tok, model, qatar)
    custom_score  = eval_custom(tok, model)

    results[name] = {
        "AraDiCE_MCq": aradice_score,
        "Bahraini_Custom": custom_score
    }

    print(f"AraDiCE: {aradice_score}%")
    print(f"Bahraini: {custom_score}%")

    del model
    torch.cuda.empty_cache()

# ================= SAVE =================
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ================= PRINT TABLE =================
print("\nFINAL TABLE (Thesis Ready)")
print(f"{'Model':<15} {'AraDiCE%':>10} {'Bahraini%':>12}")
print("-"*40)

for k,v in results.items():
    print(f"{k:<15} {str(v['AraDiCE_MCq'])+'%':>10} {str(v['Bahraini_Custom'])+'%':>12}")

print(f"\nSaved to {OUT}")
PYEOF
