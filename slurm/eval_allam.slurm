#!/bin/bash
#SBATCH --job-name=allam_eval
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/allam_eval_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/allam_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
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
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
ALLAM = "humain-ai/ALLaM-7B-Instruct-preview"
CKPT  = "/data/datasets/user151/qwen-arabic-kd/checkpoints/allam_selectkd"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/allam_eval.json"

PAIRS = [
    {"en":"What is the capital of France?","ar":"ما هي عاصمة فرنسا؟","kw_en":["paris"],"kw_ar":["باريس","paris"]},
    {"en":"What is 15 multiplied by 7?","ar":"ما ناتج 15 في 7؟","kw_en":["105"],"kw_ar":["105"]},
    {"en":"What is the largest ocean?","ar":"ما أكبر محيط؟","kw_en":["pacific"],"kw_ar":["هادئ","pacific"]},
    {"en":"How many bones in human body?","ar":"كم عدد عظام الإنسان؟","kw_en":["206"],"kw_ar":["206"]},
    {"en":"What year did Berlin Wall fall?","ar":"متى سقط جدار برلين؟","kw_en":["1989"],"kw_ar":["1989"]},
    {"en":"Who wrote Romeo and Juliet?","ar":"من كتب روميو وجولييت؟","kw_en":["shakespeare"],"kw_ar":["شكسبير"]},
    {"en":"What is the boiling point of water?","ar":"ما درجة غليان الماء؟","kw_en":["100"],"kw_ar":["100"]},
    {"en":"What is machine learning?","ar":"ما هو تعلم الآلة؟","kw_en":["data","learn"],"kw_ar":["بيانات","تعلم"]},
    {"en":"What does CBB stand for in Bahrain?","ar":"ما اختصار CBB في البحرين؟","kw_en":["central bank"],"kw_ar":["مركزي"]},
    {"en":"What is Knowledge Distillation?","ar":"ما هو تقطير المعرفة؟","kw_en":["teacher","student"],"kw_ar":["معلم","طالب"]},
    {"en":"What is the capital of Japan?","ar":"ما عاصمة اليابان؟","kw_en":["tokyo"],"kw_ar":["طوكيو","tokyo"]},
    {"en":"What is 17 multiplied by 13?","ar":"ما ناتج 17 في 13؟","kw_en":["221"],"kw_ar":["221"]},
    {"en":"What is artificial intelligence?","ar":"ما هو الذكاء الاصطناعي؟","kw_en":["machine","learn"],"kw_ar":["آلة","تعلم"]},
    {"en":"What does wayyid mean in Bahraini?","ar":"ما معنى وايد في البحريني؟","kw_en":["much","very"],"kw_ar":["كثير","جدا"]},
    {"en":"What is LoRA?","ar":"ما هو LoRA؟","kw_en":["rank","adapt"],"kw_ar":["رتبة","تكيف"]},
]

def load_model(base, ckpt=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base, token=TOKEN)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map={"":0}, token=TOKEN)
    if ckpt: m = PeftModel.from_pretrained(m, ckpt)
    m.eval()
    return tok, m

def gen(tok, model, q, max_new=100):
    # ALLaM uses standard format
    prompt = f"<s>[INST] {q} [/INST]"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp.strip()

def score(resp, kws):
    return any(k.lower() in resp.lower() for k in kws)

def eval_model(name, tok, model):
    en_ok=0; ar_ok=0; clcs=0
    for pair in tqdm(PAIRS, desc=name):
        er  = gen(tok, model, pair["en"])
        ar_r= gen(tok, model, pair["ar"])
        e_ok= score(er,   pair["kw_en"])
        a_ok= score(ar_r, pair["kw_ar"])
        en_ok += e_ok; ar_ok += a_ok
        if e_ok and a_ok: clcs += 1
    n = len(PAIRS)
    en_a = round(en_ok/n*100,1)
    ar_a = round(ar_ok/n*100,1)
    lpg  = round(abs(en_a-ar_a),1)
    print(f"  {name}: EN={en_a}% AR={ar_a}% LPG={lpg}% CLCS={round(clcs/n*100,1)}%")
    return {"EN":en_a,"AR":ar_a,"LPG":lpg,"CLCS":round(clcs/n*100,1)}

results = {}

# Base ALLaM (no KD)
print("Evaluating ALLaM base...")
tok, model = load_model(ALLAM)
results["allam_base"] = eval_model("allam_base", tok, model)
del model; torch.cuda.empty_cache()

# ALLaM + SelecTKD
print("Evaluating ALLaM SelecTKD...")
tok, model = load_model(ALLAM, CKPT)
results["allam_selectkd"] = eval_model("allam_selectkd", tok, model)
del model; torch.cuda.empty_cache()

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

print(f"\nCROSS-ARCHITECTURE COMPARISON:")
print(f"{'Model':<20} {'EN%':>6} {'AR%':>6} {'LPG':>6}")
print("-"*42)
# Qwen results from main experiment
qwen = {"base":"86.7/63.3/23.4","selectkd":"86.7/86.7/0.0"}
print(f"{'qwen_base':<20} {'86.7%':>6} {'63.3%':>6} {'23.4%':>6}")
print(f"{'qwen_selectkd':<20} {'86.7%':>6} {'86.7%':>6} {' 0.0%':>6}")
for name,res in results.items():
    print(f"{name:<20} {str(res['EN'])+'%':>6} {str(res['AR'])+'%':>6} {str(res['LPG'])+'%':>6}")
print(f"\nSaved to {OUT}")
PYEOF
