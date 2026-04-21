#!/bin/bash
#SBATCH --job-name=kd_table1
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/table1_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/table1_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

python - << 'PYEOF'
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/layer1_complete.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

# All 5 models
CKPTS = {
    "base_7b":     None,
    "sft_7b":      "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":   "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

print("Loading benchmarks...")
mmlu  = load_dataset("cais/mmlu", "all", split="test[:100]")
gsm8k = load_dataset("gsm8k", "main", split="test[:50]")
armmlu = load_dataset("MBZUAI/ArabicMMLU", "General Knowledge", split="test[:100]")
print(f"MMLU:{len(mmlu)} GSM8K:{len(gsm8k)} ArabicMMLU:{len(armmlu)}")

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, prompt, max_new=50):
    msgs = [{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=768).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def eval_mmlu(tok, model):
    cm = {0:"A",1:"B",2:"C",3:"D"}; ok=0
    for item in tqdm(mmlu, desc="MMLU"):
        ch = item["choices"]
        q = f"{item['question']}\nA) {ch[0]}\nB) {ch[1]}\nC) {ch[2]}\nD) {ch[3]}\nAnswer A,B,C or D only."
        r = gen(tok, model, q, max_new=5)
        if cm[item["answer"]] in r.upper()[:5]: ok+=1
    return round(ok/len(mmlu)*100,1)

def eval_gsm8k(tok, model):
    ok=0
    for item in tqdm(gsm8k, desc="GSM8K"):
        q = f"{item['question']}\nSolve and give final numeric answer only."
        r = gen(tok, model, q, max_new=200)
        ans = item["answer"].split("####")[-1].strip().replace(",","")
        if ans in r.replace(",",""): ok+=1
    return round(ok/len(gsm8k)*100,1)

def eval_armmlu(tok, model):
    ok=0
    for item in tqdm(armmlu, desc="ArabicMMLU"):
        q = item["Question"]; opts=[]
        labels=["A","B","C","D","E"]
        for i in range(1,6):
            o=item.get(f"Option {i}")
            if o: opts.append(f"{labels[i-1]}) {o}")
        prompt = q+"\n"+"\n".join(opts)+"\nأجب بحرف واحد فقط."
        r = gen(tok, model, prompt, max_new=5)
        ak = item["Answer Key"].strip().upper()
        if r.upper().strip().startswith(ak) or f" {ak}" in r.upper()[:10]: ok+=1
    return round(ok/len(armmlu)*100,1)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue
    tok, model = load_model(ckpt)
    mmlu_acc  = eval_mmlu(tok, model)
    gsm_acc   = eval_gsm8k(tok, model)
    arml_acc  = eval_armmlu(tok, model)
    print(f"  MMLU={mmlu_acc}% GSM8K={gsm_acc}% ArabicMMLU={arml_acc}%")
    all_results[model_name] = {
        "MMLU_EN": mmlu_acc, "GSM8K": gsm_acc, "ArabicMMLU": arml_acc
    }
    del model; torch.cuda.empty_cache()

with open(OUT,"w") as f:
    json.dump(all_results,f,indent=2)

print(f"\nTABLE 1 — COMPLETE:")
print(f"{'Model':<16} {'MMLU-EN':>8} {'GSM8K':>7} {'ArabicMMLU':>11}")
print("-"*46)
for name,res in all_results.items():
    print(f"{name:<16} {str(res['MMLU_EN'])+'%':>8} {str(res['GSM8K'])+'%':>7} {str(res['ArabicMMLU'])+'%':>11}")
PYEOF
