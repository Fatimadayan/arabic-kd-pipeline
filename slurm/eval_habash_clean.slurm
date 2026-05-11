#!/bin/bash
#SBATCH --job-name=kd_habash
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/habash_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/habash_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python - << 'PYEOF'
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN     = os.environ["HF_TOKEN"]
BASE      = "Qwen/Qwen2.5-7B-Instruct"
PROMPTS   = "/data/datasets/user151/qwen-arabic-kd/data/habash_prompts.json"
OUT       = "/data/datasets/user151/qwen-arabic-kd/results/habash_eval_results.json"

with open(PROMPTS, encoding="utf-8") as f:
    data = json.load(f)

DIALECT_ID  = data["dialect_id"]
NORMALIZE   = data["normalize"]
CODE_SWITCH = data["code_switch"]

print(f"Loaded: {len(DIALECT_ID)+len(NORMALIZE)+len(CODE_SWITCH)} prompts total")
print(f"GPU memory free: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB / {torch.cuda.mem_get_info()[1]/1024**3:.1f}GB")

CKPTS = {
    "base_7b":     None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

def has_cjk(text):
    return any('\u4e00' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7a3' for c in text)

def load_model(ckpt):
    torch.cuda.empty_cache()
    print(f"  GPU free before load: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN,
        low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    print(f"  GPU free after load:  {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")
    return tok, model

def gen(tok, model, q, max_new=80):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=384).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score(resp, kws):
    if has_cjk(resp): return False
    if len(resp.strip()) < 10: return False
    return any(k.lower() in resp.lower() for k in kws)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*55}\nEvaluating: {model_name}\n{'='*55}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP — checkpoint not found"); continue

    tok, model = load_model(ckpt)
    did_ok=0; norm_ok=0; cs_ok=0; cjk_total=0
    details = []

    for item in tqdm(DIALECT_ID, desc="Dialect ID"):
        resp = gen(tok, model, item["q"])
        cjk  = has_cjk(resp)
        ok   = score(resp, item["kw"])
        if cjk: cjk_total+=1
        if ok:  did_ok+=1
        details.append({"cat":"dialect_id","q":item["q"][:60],
                        "resp":resp[:100],"ok":ok,"cjk":cjk})

    for item in tqdm(NORMALIZE, desc="Normalize"):
        resp = gen(tok, model, item["q"])
        cjk  = has_cjk(resp)
        ok   = score(resp, item["kw"])
        if cjk: cjk_total+=1
        if ok:  norm_ok+=1
        details.append({"cat":"normalize","q":item["q"][:60],
                        "resp":resp[:100],"ok":ok,"cjk":cjk})

    for item in tqdm(CODE_SWITCH, desc="Code-Switch"):
        resp = gen(tok, model, item["q"])
        cjk  = has_cjk(resp)
        ok   = score(resp, item["kw"])
        if cjk: cjk_total+=1
        if ok:  cs_ok+=1
        details.append({"cat":"code_switch","q":item["q"][:60],
                        "resp":resp[:100],"ok":ok,"cjk":cjk})

    n = 100
    did_pct  = round(did_ok/n*100, 1)
    norm_pct = round(norm_ok/n*100, 1)
    cs_pct   = round(cs_ok/n*100, 1)
    overall  = round((did_ok+norm_ok+cs_ok)/(n*3)*100, 1)
    cjk_pct  = round(cjk_total/(n*3)*100, 1)

    print(f"\n  Dialect ID:   {did_pct}%  ({did_ok}/{n})")
    print(f"  Normalize:    {norm_pct}%  ({norm_ok}/{n})")
    print(f"  Code-Switch:  {cs_pct}%  ({cs_ok}/{n})")
    print(f"  Overall:      {overall}%")
    print(f"  CJK failures: {cjk_total}/{n*3} ({cjk_pct}%)")

    all_results[model_name] = {
        "dialect_id": did_pct, "normalize": norm_pct,
        "code_switch": cs_pct, "overall": overall,
        "cjk_failures": cjk_total, "cjk_rate": cjk_pct,
        "n_per_category": n, "details": details[:15]
    }
    del model; torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nFINAL TABLE — Habash Corpus (100 real Bahraini sentences/task, CJK-filtered):")
print(f"{'Model':<18} {'Dialect%':>9} {'Normalize%':>11} {'Code-Switch%':>13} {'Overall%':>9} {'CJK%':>7}")
print("-"*72)
for name, res in all_results.items():
    star = " *" if "selectkd" in name else ""
    print(f"{name+star:<20} {str(res['dialect_id'])+'%':>9} {str(res['normalize'])+'%':>11} {str(res['code_switch'])+'%':>13} {str(res['overall'])+'%':>9} {str(res['cjk_rate'])+'%':>7}")
print(f"\nSaved: {OUT}")
PYEOF
