#!/bin/bash
#SBATCH --job-name=kd_aradice
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/aradice_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/aradice_%j.err
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
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

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/aradice_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":     None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# Load AraDiCE
print("Loading QCRI/AraDiCE...")
try:
    aradice_data = load_dataset("QCRI/AraDiCE", split="train")
    print(f"AraDiCE loaded: {len(aradice_data)} samples")
    print(f"Keys: {list(aradice_data[0].keys())}")
    HAS_ARADICE = True
except Exception as e:
    print(f"AraDiCE load failed: {e}")
    HAS_ARADICE = False

# Custom dialect + cultural evaluation (GCC-specific)
# This is our Layer 2 regardless of AraDiCE availability
DIALECT_PROMPTS = [
    # Bahraini dialect comprehension
    {"q":"شلونك اليوم؟ وايد تعبان؟","type":"dialect_comp","kw":["بخير","زين","تمام","لا","ok","fine"]},
    {"q":"وين رايح الحين؟","type":"dialect_comp","kw":["أذهب","رايح","ماضي","going","أروح"]},
    {"q":"شو تبي تاكل؟ في مطعم وايد زين قريب","type":"dialect_comp","kw":["أكل","طعام","مطعم","food","eat"]},
    {"q":"الحين وايد زحمة بالسوق","type":"dialect_comp","kw":["ازدحام","ناس","زحمة","crowd","busy","سوق"]},
    {"q":"اشرح لي معنى كلمة 'حيل' بالبحريني","type":"dialect_meaning","kw":["جدا","قوي","كثير","very","much","strong"]},
    {"q":"اشرح معنى 'ما أبي' بالخليجي","type":"dialect_meaning","kw":["لا أريد","لا أبغي","لا","don't want","no"]},
    {"q":"ما معنى 'صج' في اللهجة الخليجية؟","type":"dialect_meaning","kw":["حقا","فعلا","صحيح","true","really","indeed"]},
    {"q":"اشرح كلمة 'يهال' في البحريني","type":"dialect_meaning","kw":["أطفال","صغار","ولاد","children","kids"]},
    # GCC cultural knowledge
    {"q":"ما هي رؤية البحرين 2030 وأهدافها؟","type":"cultural","kw":["اقتصاد","تطوير","تنويع","vision","economy","develop"]},
    {"q":"ما هو مهرجان ربيع البحرين؟","type":"cultural","kw":["مهرجان","فعالية","ثقافة","festival","bahrain","culture"]},
    {"q":"اشرح أهمية صناعة اللؤلؤ في تاريخ البحرين","type":"cultural","kw":["لؤلؤ","تاريخ","بحر","pearl","history","sea"]},
    {"q":"ما هي أبرز المأكولات البحرينية التقليدية؟","type":"cultural","kw":["مجبوس","هريس","طعام","food","rice","traditional"]},
    # Code-switching (common in GCC)
    {"q":"The meeting كانت productive وايد، شو رأيك؟","type":"code_switch","kw":["اجتماع","منتج","رأي","meeting","productive","good"]},
    {"q":"أبي أعمل startup في البحرين، من وين أبدأ؟","type":"code_switch","kw":["شركة","بداية","خطوات","company","start","begin"]},
    {"q":"The project deadline وايد قريب، كيف نخلص؟","type":"code_switch","kw":["موعد","إنهاء","انتهاء","deadline","finish","complete"]},
    # Dialect normalization
    {"q":"حول هذه الجملة للفصحى: 'ما قدرت أجي الحين'","type":"normalization","kw":["لم أستطع","لم يتسن","تعذر","could not","unable","attend"]},
    {"q":"حول للفصحى: 'وايد زين هالشغل'","type":"normalization","kw":["جيد","ممتاز","عمل","excellent","good","work"]},
    {"q":"ما مقابل 'بغيت' في الفصحى؟","type":"normalization","kw":["أريد","أرغب","want","wish","desire"]},
    # Dialect identification
    {"q":"حدد اللهجة: 'إيه والله كده تمام يا صاحبي'","type":"dialect_id","kw":["مصري","egypt","مصر","egyptian"]},
    {"q":"حدد اللهجة: 'والله يا خوي هاي منيح'","type":"dialect_id","kw":["شامي","سوري","لبناني","levantine","syrian"]},
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, q, max_new=150):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             output_scores=True, return_dict_in_generate=True)
    resp = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    conf = sum(F.softmax(s,dim=-1).max().item() for s in out.scores)/max(len(out.scores),1)
    return resp.strip(), conf

def score(resp, kws):
    return any(k.lower() in resp.lower() for k in kws)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue

    tok, model = load_model(ckpt)

    # --- AraDiCE evaluation (if available) ---
    aradice_score = None
    if HAS_ARADICE:
        print("Running AraDiCE...")
        ar_ok = 0
        for item in tqdm(aradice_data, desc="AraDiCE"):
            # Use whatever fields are available
            keys = list(item.keys())
            q_field = [k for k in keys if "source" in k.lower() or "input" in k.lower() or "sentence" in k.lower()]
            a_field = [k for k in keys if "target" in k.lower() or "output" in k.lower() or "translation" in k.lower()]
            if q_field and a_field:
                q = str(item[q_field[0]])
                expected = str(item[a_field[0]])
                prompt = f"ترجم أو اشرح: {q}"
                resp, _ = gen(tok, model, prompt)
                # Check overlap with expected
                exp_words = set(expected.lower().split())
                resp_words = set(resp.lower().split())
                overlap = len(exp_words & resp_words) / max(len(exp_words), 1)
                if overlap > 0.3:
                    ar_ok += 1
        aradice_score = round(ar_ok/len(aradice_data)*100, 1)
        print(f"  AraDiCE: {aradice_score}%")

    # --- Custom dialect/cultural evaluation ---
    print("Running custom GCC dialect evaluation...")
    by_type = {}
    details = []
    total_ok = 0

    for item in tqdm(DIALECT_PROMPTS, desc="GCC-Dialect"):
        resp, conf = gen(tok, model, item["q"])
        ok = score(resp, item["kw"])
        t  = item["type"]
        if t not in by_type:
            by_type[t] = {"ok":0,"total":0}
        by_type[t]["total"] += 1
        if ok:
            by_type[t]["ok"] += 1
            total_ok += 1
        details.append({
            "question": item["q"], "type": t,
            "response": resp[:200], "correct": ok, "confidence": round(conf,4)
        })

    n = len(DIALECT_PROMPTS)
    overall = round(total_ok/n*100, 1)

    type_scores = {}
    for t, v in by_type.items():
        type_scores[t] = round(v["ok"]/v["total"]*100, 1)

    print(f"\n  Overall GCC Dialect: {overall}%")
    for t, s in type_scores.items():
        print(f"    {t}: {s}%")

    all_results[model_name] = {
        "aradice_score":     aradice_score,
        "dialect_overall":   overall,
        "dialect_comp":      type_scores.get("dialect_comp", 0),
        "dialect_meaning":   type_scores.get("dialect_meaning", 0),
        "cultural":          type_scores.get("cultural", 0),
        "code_switch":       type_scores.get("code_switch", 0),
        "normalization":     type_scores.get("normalization", 0),
        "dialect_id":        type_scores.get("dialect_id", 0),
    }

    del model; torch.cuda.empty_cache()

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nTABLE 2 — ARABIC & GCC CAPABILITY:")
print(f"{'Model':<16} {'AraDiCE':>8} {'Dialect%':>9} {'Cultural%':>10} {'CodeSwitch%':>12} {'Normalize%':>11}")
print("-"*70)
for name, res in all_results.items():
    ar  = str(res['aradice_score'])+"%" if res['aradice_score'] else "—"
    d   = str(res['dialect_overall'])+"%"
    c   = str(res['cultural'])+"%"
    cs  = str(res['code_switch'])+"%"
    nm  = str(res['normalization'])+"%"
    print(f"{name:<16} {ar:>8} {d:>9} {c:>10} {cs:>12} {nm:>11}")

print(f"\nResults saved to {OUT}")
PYEOF
