#!/bin/bash
#SBATCH --job-name=kd_check
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/check_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/check_%j.err
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

python - << 'PYEOF'
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"

CKPTS = {
    "base_7b":     None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

TEST_QS = [
    ("AR factual",   "ما هي عاصمة فرنسا؟"),
    ("AR dialect",   "ما معنى وايد في اللهجة البحرينية؟"),
    ("AR normalize", "حوّل للفصحى: الحين بروح السوق"),
    ("AR reasoning", "إذا كان لدي 10 تفاحات وأعطيت 3، كم تبقى؟"),
    ("EN factual",   "What is the capital of France?"),
    ("EN reasoning", "If I have 10 apples and give 3 away, how many remain?"),
    ("GCC domain",   "ما هو دور مصرف البحرين المركزي CBB؟"),
    ("Code-switch",  "وايد ضغط في work اليوم، كيف أتعامل معه؟"),
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def ask(tok, model, q):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inputs = tok(text,return_tensors="pt",max_length=512,truncation=True).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs,max_new_tokens=120,do_sample=False,
                            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*60}\nMODEL: {model_name}\n{'='*60}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue
    tok, model = load_model(ckpt)
    for label, q in TEST_QS:
        resp = ask(tok, model, q)
        cjk = any('\u4e00'<=c<='\u9fff' for c in resp)
        flag = " *** CJK DETECTED ***" if cjk else ""
        print(f"\n  [{label}]{flag}")
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
    del model; torch.cuda.empty_cache()
PYEOF
