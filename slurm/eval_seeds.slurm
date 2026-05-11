#!/bin/bash
#SBATCH --job-name=kd_seeds
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/seeds_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/seeds_%j.err
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

python - << 'PYEOF'
import os, json, torch, numpy as np, math
import torch.nn.functional as F
from scipy import stats as scipy_stats
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/seeds_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":      None,
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

PAIRS = [
    {"en":"What is the capital of France?","ar":"ما هي عاصمة فرنسا؟","kw_en":["paris"],"kw_ar":["باريس","paris"]},
    {"en":"What is the capital of Japan?","ar":"ما هي عاصمة اليابان؟","kw_en":["tokyo"],"kw_ar":["طوكيو","tokyo"]},
    {"en":"What is the capital of Egypt?","ar":"ما هي عاصمة مصر؟","kw_en":["cairo"],"kw_ar":["القاهرة","cairo"]},
    {"en":"What is 15 multiplied by 7?","ar":"ما ناتج 15 في 7؟","kw_en":["105"],"kw_ar":["105"]},
    {"en":"What is 17 multiplied by 13?","ar":"ما ناتج 17 في 13؟","kw_en":["221"],"kw_ar":["221"]},
    {"en":"What is the square root of 144?","ar":"ما الجذر التربيعي لـ 144؟","kw_en":["12"],"kw_ar":["12"]},
    {"en":"How many bones in human body?","ar":"كم عدد عظام الإنسان؟","kw_en":["206"],"kw_ar":["206"]},
    {"en":"What year did Berlin Wall fall?","ar":"في أي عام سقط جدار برلين؟","kw_en":["1989"],"kw_ar":["1989"]},
    {"en":"Who wrote Romeo and Juliet?","ar":"من كتب روميو وجولييت؟","kw_en":["shakespeare"],"kw_ar":["شكسبير"]},
    {"en":"What is the largest ocean?","ar":"ما أكبر محيط؟","kw_en":["pacific"],"kw_ar":["هادئ","pacific"]},
    {"en":"What is the boiling point of water?","ar":"ما درجة غليان الماء؟","kw_en":["100"],"kw_ar":["100"]},
    {"en":"What is machine learning?","ar":"ما هو تعلم الآلة؟","kw_en":["data","learn"],"kw_ar":["بيانات","تعلم"]},
    {"en":"What is artificial intelligence?","ar":"ما هو الذكاء الاصطناعي؟","kw_en":["machine","computer"],"kw_ar":["آلة","حاسب"]},
    {"en":"What does CBB stand for in Bahrain?","ar":"ما اختصار CBB في البحرين؟","kw_en":["central bank"],"kw_ar":["مركزي","central"]},
    {"en":"What is Knowledge Distillation?","ar":"ما هو تقطير المعرفة؟","kw_en":["teacher","student"],"kw_ar":["معلم","طالب"]},
]

SEEDS = [42, 123, 777]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, q, seed=42, max_new=100):
    torch.manual_seed(seed)
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score(resp, kws):
    return any(k.lower() in resp.lower() for k in kws)

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return abs(np.mean(a)-np.mean(b))/pooled if pooled > 0 else 0

all_results = {}
raw = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print(f"SKIP"); continue

    tok, model = load_model(ckpt)
    seed_en, seed_ar, seed_lpg, seed_clcs = [], [], [], []

    for seed in SEEDS:
        en_ok, ar_ok, clcs = 0, 0, 0
        for pair in PAIRS:
            er = gen(tok, model, pair["en"], seed)
            ar_r = gen(tok, model, pair["ar"], seed)
            e_ok = score(er, pair["kw_en"])
            a_ok = score(ar_r, pair["kw_ar"])
            en_ok += e_ok; ar_ok += a_ok
            if e_ok and a_ok: clcs += 1

        n = len(PAIRS)
        en_a = en_ok/n*100; ar_a = ar_ok/n*100
        seed_en.append(en_a); seed_ar.append(ar_a)
        seed_lpg.append(abs(en_a-ar_a))
        seed_clcs.append(clcs/n*100)
        print(f"  Seed {seed}: EN={en_a:.1f}% AR={ar_a:.1f}% LPG={abs(en_a-ar_a):.1f}%")

    raw[model_name] = {"en": seed_en, "ar": seed_ar, "lpg": seed_lpg}
    all_results[model_name] = {
        "EN": f"{np.mean(seed_en):.1f} ± {np.std(seed_en):.1f}",
        "AR": f"{np.mean(seed_ar):.1f} ± {np.std(seed_ar):.1f}",
        "LPG": f"{np.mean(seed_lpg):.1f} ± {np.std(seed_lpg):.1f}",
        "CLCS": f"{np.mean(seed_clcs):.1f} ± {np.std(seed_clcs):.1f}",
    }
    del model; torch.cuda.empty_cache()

# Statistical tests — base vs selectkd (key comparison)
print("\n\nSTATISTICAL SIGNIFICANCE:")
if "base_7b" in raw and "selectkd_7b" in raw:
    # LPG comparison — main claim
    t_stat, p_val = scipy_stats.ttest_rel(raw["base_7b"]["lpg"], raw["selectkd_7b"]["lpg"])
    d = cohen_d(raw["base_7b"]["lpg"], raw["selectkd_7b"]["lpg"])
    print(f"  base vs selectkd — LPG: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")
    sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "not significant"
    print(f"  → {sig}")

    # AR accuracy comparison
    t_ar, p_ar = scipy_stats.ttest_rel(raw["base_7b"]["ar"], raw["selectkd_7b"]["ar"])
    d_ar = cohen_d(raw["base_7b"]["ar"], raw["selectkd_7b"]["ar"])
    print(f"  base vs selectkd — AR acc: t={t_ar:.3f}, p={p_ar:.4f}, Cohen's d={d_ar:.3f}")

    all_results["stats"] = {
        "LPG_ttest_p": p_val, "LPG_cohen_d": d,
        "AR_ttest_p": p_ar, "AR_cohen_d": d_ar
    }

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nFINAL TABLE (mean ± std, n=3 seeds):")
print(f"{'Model':<16} {'EN%':>16} {'AR%':>16} {'LPG%':>16} {'CLCS%':>16}")
print("-"*70)
for name, res in all_results.items():
    if name == "stats": continue
    print(f"{name:<16} {res['EN']:>16} {res['AR']:>16} {res['LPG']:>16} {res['CLCS']:>16}")
print(f"\nResults saved to {OUT}")
PYEOF
