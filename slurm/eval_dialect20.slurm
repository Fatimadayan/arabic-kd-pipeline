#!/bin/bash
#SBATCH --job-name=kd_dialect20
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/dialect20_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/dialect20_%j.err
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
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/dialect20_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":     None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# EXACT 20 PROMPTS — 3 categories
DIALECT_ID = [
    {"q": "شلونك اليوم؟", "kw": ["بحريني","خليجي","gulf","bahraini","شلون","كيف"]},
    {"q": "وايد تعبت من الشغل", "kw": ["بحريني","خليجي","gulf","وايد","كثير","تعب"]},
    {"q": "الحين بروح السوق", "kw": ["بحريني","خليجي","gulf","الحين","الآن","بروح"]},
    {"q": "ما عندي وقت الحين", "kw": ["بحريني","خليجي","gulf","الحين","وقت"]},
    {"q": "وينك من زمان ما شفناك", "kw": ["بحريني","خليجي","gulf","وين","أين"]},
    {"q": "هالشي وايد صعب", "kw": ["بحريني","خليجي","gulf","وايد","كثير","صعب"]},
    {"q": "خلنا نخلص الموضوع بسرعة", "kw": ["بحريني","خليجي","gulf","خلنا","دعنا"]},
]

NORMALIZE = [
    {"q": "حول للفصحى: شلونك", "kw": ["كيف حالك","كيف","حال"]},
    {"q": "حول للفصحى: وايد تعبت", "kw": ["كثيرا","كثير","تعبت","أتعبت"]},
    {"q": "حول للفصحى: الحين بروح", "kw": ["الآن سأذهب","الآن","سأذهب","أذهب"]},
    {"q": "حول للفصحى: وينك من زمان", "kw": ["أين كنت","أين","كنت","طويلاً"]},
    {"q": "حول للفصحى: ما عندي وقت", "kw": ["ليس لدي وقت","لا وقت","ليس"]},
    {"q": "حول للفصحى: هالشي وايد صعب", "kw": ["هذا صعب","صعب جداً","كثيراً"]},
    {"q": "حول للفصحى: خلنا نخلص", "kw": ["دعنا ننهي","لننه","دعنا"]},
]

CODE_SWITCH = [
    {"q": "خلنا schedule meeting الحين", "kw": ["اجتماع","جدول","meeting","schedule","موعد"]},
    {"q": "وايد ضغط في work اليوم", "kw": ["ضغط","عمل","work","كثير","شغل"]},
    {"q": "أبغى update عن المشروع بسرعة", "kw": ["تحديث","update","مشروع","تقرير"]},
    {"q": "خلصت التقرير بس يحتاج review", "kw": ["مراجعة","review","تقرير","انتهيت"]},
    {"q": "الحين نقدر deploy النظام؟", "kw": ["نشر","deploy","نظام","تشغيل"]},
    {"q": "في مشكلة في system لازم نحلها", "kw": ["مشكلة","نظام","system","حل"]},
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, q, max_new=120):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score(resp, kws):
    return any(k.lower() in resp.lower() for k in kws)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue

    tok, model = load_model(ckpt)
    details = []

    # Dialect ID
    did_ok = 0
    for item in tqdm(DIALECT_ID, desc="Dialect ID"):
        resp = gen(tok, model, f"حدد لهجة هذه الجملة وهل هي بحرينية خليجية: '{item['q']}'")
        ok = score(resp, item["kw"])
        did_ok += ok
        details.append({"cat":"dialect_id","q":item["q"],"resp":resp[:150],"ok":ok})

    # Normalize
    norm_ok = 0
    for item in tqdm(NORMALIZE, desc="Normalize"):
        resp = gen(tok, model, item["q"])
        ok = score(resp, item["kw"])
        norm_ok += ok
        details.append({"cat":"normalize","q":item["q"],"resp":resp[:150],"ok":ok})

    # Code-Switch
    cs_ok = 0
    for item in tqdm(CODE_SWITCH, desc="Code-Switch"):
        resp = gen(tok, model, item["q"])
        ok = score(resp, item["kw"])
        cs_ok += ok
        details.append({"cat":"code_switch","q":item["q"],"resp":resp[:150],"ok":ok})

    did_pct  = round(did_ok/len(DIALECT_ID)*100,1)
    norm_pct = round(norm_ok/len(NORMALIZE)*100,1)
    cs_pct   = round(cs_ok/len(CODE_SWITCH)*100,1)
    overall  = round((did_ok+norm_ok+cs_ok)/20*100,1)

    print(f"\n  Dialect ID:   {did_pct}%  ({did_ok}/{len(DIALECT_ID)})")
    print(f"  Normalize:    {norm_pct}%  ({norm_ok}/{len(NORMALIZE)})")
    print(f"  Code-Switch:  {cs_pct}%  ({cs_ok}/{len(CODE_SWITCH)})")
    print(f"  Overall:      {overall}%")

    all_results[model_name] = {
        "dialect_id":  did_pct,
        "normalize":   norm_pct,
        "code_switch": cs_pct,
        "overall":     overall,
        "details":     details
    }
    del model; torch.cuda.empty_cache()

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nTABLE 4.2 — BAHRAINI DIALECT EVALUATION:")
print(f"{'Model':<16} {'Dialect ID%':>12} {'Normalize%':>11} {'Code-Switch%':>13} {'Overall%':>9}")
print("-"*65)
for name,res in all_results.items():
    star = " ★" if name=="selectkd_7b" else ""
    print(f"{name+star:<18} {str(res['dialect_id'])+'%':>12} {str(res['normalize'])+'%':>11} {str(res['code_switch'])+'%':>13} {str(res['overall'])+'%':>9}")
print(f"\nSaved to {OUT}")
PYEOF
