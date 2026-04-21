#!/bin/bash
#SBATCH --job-name=kd_mmlu
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/mmlu_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/mmlu_%j.err
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
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
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/layer1_benchmarks.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":      None,
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

print("Loading MMLU (100 samples)...")
mmlu = load_dataset("cais/mmlu", "all", split="test[:100]")

print("Loading ArabicMMLU (100 samples)...")
try:
    arabic_mmlu = load_dataset("MBZUAI/ArabicMMLU", "General Knowledge", split="test[:100]")
    has_ar_mmlu = True
    print("ArabicMMLU loaded!")
except Exception as e:
    print(f"ArabicMMLU not available: {e}")
    has_ar_mmlu = False

# GSM8K math problems (50 samples)
print("Loading GSM8K (50 samples)...")
try:
    gsm8k = load_dataset("gsm8k", "main", split="test[:50]")
    has_gsm = True
except:
    has_gsm = False
    print("GSM8K not available")

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def generate(tok, model, prompt, max_new=50):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=768).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print(f"SKIP — not found: {ckpt}")
        continue

    tok, model = load_model(ckpt)
    results = {}

    # MMLU English
    print("Running MMLU English...")
    choices_map = {0:"A", 1:"B", 2:"C", 3:"D"}
    mmlu_correct = 0
    for item in tqdm(mmlu, desc="MMLU-EN"):
        q = item["question"]
        ch = item["choices"]
        prompt = f"{q}\nA) {ch[0]}\nB) {ch[1]}\nC) {ch[2]}\nD) {ch[3]}\nAnswer with A, B, C, or D only."
        resp = generate(tok, model, prompt, max_new=5)
        if choices_map[item["answer"]] in resp.upper()[:5]:
            mmlu_correct += 1
    mmlu_acc = round(mmlu_correct/len(mmlu)*100, 1)
    results["MMLU_EN"] = mmlu_acc
    print(f"  MMLU-EN: {mmlu_acc}%")

    # ArabicMMLU
    if has_ar_mmlu:
        print("Running ArabicMMLU...")
        ar_correct = 0
        for item in tqdm(arabic_mmlu, desc="ArabicMMLU"):
            q = item.get("question","")
            ch = item.get("choices", item.get("options",[]))
            if not q or not ch: continue
            choice_labels = ["أ","ب","ج","د"] if len(ch)==4 else ["A","B","C","D"]
            prompt = f"{q}\n"
            for i,c in enumerate(ch[:4]):
                prompt += f"{choice_labels[i]}) {c}\n"
            prompt += "أجب بحرف واحد فقط."
            resp = generate(tok, model, prompt, max_new=5)
            ans_idx = item.get("answer", item.get("label", 0))
            if isinstance(ans_idx, int) and ans_idx < len(choice_labels):
                if choice_labels[ans_idx] in resp[:10]:
                    ar_correct += 1
        ar_mmlu_acc = round(ar_correct/len(arabic_mmlu)*100, 1)
        results["ArabicMMLU"] = ar_mmlu_acc
        print(f"  ArabicMMLU: {ar_mmlu_acc}%")

    # GSM8K Math
    if has_gsm:
        print("Running GSM8K...")
        gsm_correct = 0
        for item in tqdm(gsm8k, desc="GSM8K"):
            prompt = f"{item['question']}\nSolve step by step and give final answer as a number."
            resp = generate(tok, model, prompt, max_new=200)
            answer = item["answer"].split("####")[-1].strip().replace(",","")
            if answer in resp.replace(",",""):
                gsm_correct += 1
        gsm_acc = round(gsm_correct/len(gsm8k)*100, 1)
        results["GSM8K"] = gsm_acc
        print(f"  GSM8K: {gsm_acc}%")

    all_results[model_name] = results
    del model
    torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nLAYER 1 — STANDARD BENCHMARKS TABLE:")
print(f"{'Model':<16} {'MMLU-EN':>8} {'ArabicMMLU':>11} {'GSM8K':>7}")
print("-"*46)
for name, res in all_results.items():
    mmlu = f"{res.get('MMLU_EN','—')}%"
    ar   = f"{res.get('ArabicMMLU','—')}%"
    gsm  = f"{res.get('GSM8K','—')}%"
    print(f"{name:<16} {mmlu:>8} {ar:>11} {gsm:>7}")

print(f"\nResults saved to {OUT}")
PYEOF
