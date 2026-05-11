#!/bin/bash
#SBATCH --job-name=kd_aradice2
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/aradice2_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/aradice2_%j.err
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

# Load AraDiCE-Culture — Qatar config (most GCC-relevant)
print("Loading AraDiCE-Culture (Qatar)...")
try:
    culture_qatar = load_dataset("QCRI/AraDiCE-Culture", "Qatar", split="train")
    print(f"Qatar culture: {len(culture_qatar)} samples, keys: {list(culture_qatar[0].keys())}")
    print("Sample:", culture_qatar[0])
except Exception as e:
    print(f"Qatar failed: {e}")
    culture_qatar = None

# Also load Egypt for comparison
try:
    culture_egypt = load_dataset("QCRI/AraDiCE-Culture", "Egypt", split="train")
    print(f"Egypt culture: {len(culture_egypt)} samples")
except:
    culture_egypt = None

# Custom GCC dialect evaluation (Layer 2 core)
DIALECT_PROMPTS = [
    {"q":"شلونك اليوم؟ وايد تعبان؟","type":"dialect_comp","kw":["بخير","زين","تمام","لا","fine","خير"]},
    {"q":"وين رايح الحين؟","type":"dialect_comp","kw":["أذهب","رايح","going","أروح","مكان"]},
    {"q":"الحين وايد زحمة بالسوق","type":"dialect_comp","kw":["ازدحام","زحمة","crowd","busy","كثير","ناس"]},
    {"q":"اشرح معنى كلمة 'حيل' بالبحريني","type":"dialect_meaning","kw":["جدا","قوي","كثير","very","much","strong","شديد"]},
    {"q":"ما معنى 'صج' في اللهجة الخليجية؟","type":"dialect_meaning","kw":["حقا","فعلا","صحيح","true","really","indeed","صحيح"]},
    {"q":"اشرح كلمة 'يهال' في البحريني","type":"dialect_meaning","kw":["أطفال","صغار","ولاد","children","kids","صبيان"]},
    {"q":"ما معنى 'بغيت' بالخليجي؟","type":"dialect_meaning","kw":["أريد","أرغب","want","wish","طلب"]},
    {"q":"ما هي رؤية البحرين 2030 وأهدافها؟","type":"cultural","kw":["اقتصاد","تطوير","تنويع","vision","economy","develop","مستقبل"]},
    {"q":"اشرح أهمية صناعة اللؤلؤ في تاريخ البحرين","type":"cultural","kw":["لؤلؤ","تاريخ","بحر","pearl","history","sea","غوص"]},
    {"q":"ما هي أبرز المأكولات البحرينية التقليدية؟","type":"cultural","kw":["مجبوس","هريس","طعام","food","rice","traditional","بحريني"]},
    {"q":"The meeting كانت productive وايد، شو رأيك؟","type":"code_switch","kw":["اجتماع","منتج","رأي","meeting","productive","good","ناجح"]},
    {"q":"أبي أعمل startup في البحرين، من وين أبدأ؟","type":"code_switch","kw":["شركة","بداية","خطوات","company","start","begin","تأسيس"]},
    {"q":"The project deadline وايد قريب، كيف نخلص؟","type":"code_switch","kw":["موعد","إنهاء","deadline","finish","complete","انتهاء"]},
    {"q":"حول للفصحى: 'ما قدرت أجي الحين'","type":"normalization","kw":["لم أستطع","تعذر","could not","unable","لم يتسن","أجيء"]},
    {"q":"حول للفصحى: 'وايد زين هالشغل'","type":"normalization","kw":["جيد","ممتاز","عمل","excellent","good","work","رائع"]},
    {"q":"ما مقابل 'بغيت' في الفصحى؟","type":"normalization","kw":["أريد","أرغب","want","wish","desire","طلب"]},
    {"q":"حدد اللهجة: 'إيه والله كده تمام يا صاحبي'","type":"dialect_id","kw":["مصري","egypt","مصر","egyptian","مصرية"]},
    {"q":"حدد اللهجة: 'والله يا خوي هاي منيح'","type":"dialect_id","kw":["شامي","سوري","لبناني","levantine","syrian","شامية"]},
    {"q":"حدد اللهجة: 'شلونك؟ وايد زين الحين؟'","type":"dialect_id","kw":["خليجي","بحريني","gulf","khaleeji","خليجية","كويتي"]},
    {"q":"ما هو دور CBB في حماية المستهلك البحريني؟","type":"gcc_domain","kw":["مصرف","مركزي","حماية","central","bank","consumer","تنظيم"]},
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

def eval_aradice_culture(tok, model, data, country):
    """Evaluate cultural knowledge on AraDiCE-Culture dataset"""
    if data is None: return None
    ok = 0
    keys = list(data[0].keys())
    print(f"  AraDiCE-Culture {country} keys: {keys}")
    for item in tqdm(data, desc=f"AraDiCE-{country}"):
        # Build prompt from available fields
        q_key = next((k for k in keys if any(x in k.lower() for x in ["question","query","input","text","sentence"])), keys[0])
        a_key = next((k for k in keys if any(x in k.lower() for x in ["answer","output","label","target"])), None)
        q = str(item[q_key])
        prompt = f"أجب على هذا السؤال الثقافي: {q}"
        resp, _ = gen(tok, model, prompt, max_new=100)
        if a_key:
            expected = str(item[a_key]).lower()
            if any(w in resp.lower() for w in expected.split()[:5] if len(w)>3):
                ok += 1
        else:
            if len(resp) > 10: ok += 1  # at least responded
    return round(ok/len(data)*100, 1)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue

    tok, model = load_model(ckpt)

    # AraDiCE Culture scores
    qatar_score = eval_aradice_culture(tok, model, culture_qatar, "Qatar")
    egypt_score = eval_aradice_culture(tok, model, culture_egypt, "Egypt")

    # Custom GCC dialect evaluation
    print("Running custom GCC dialect evaluation...")
    by_type = {}
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

    n = len(DIALECT_PROMPTS)
    overall = round(total_ok/n*100, 1)
    type_scores = {t: round(v["ok"]/v["total"]*100,1) for t,v in by_type.items()}

    print(f"\n  AraDiCE-Culture Qatar: {qatar_score}%")
    print(f"  AraDiCE-Culture Egypt: {egypt_score}%")
    print(f"  Overall GCC Dialect:   {overall}%")
    for t,s in type_scores.items():
        print(f"    {t}: {s}%")

    all_results[model_name] = {
        "aradice_qatar":   qatar_score,
        "aradice_egypt":   egypt_score,
        "dialect_overall": overall,
        "dialect_comp":    type_scores.get("dialect_comp", 0),
        "dialect_meaning": type_scores.get("dialect_meaning", 0),
        "cultural":        type_scores.get("cultural", 0),
        "code_switch":     type_scores.get("code_switch", 0),
        "normalization":   type_scores.get("normalization", 0),
        "dialect_id":      type_scores.get("dialect_id", 0),
        "gcc_domain":      type_scores.get("gcc_domain", 0),
    }

    del model; torch.cuda.empty_cache()

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nTABLE 2 — ARABIC & GCC CAPABILITY:")
print(f"{'Model':<16} {'AraDiCE-QA':>11} {'Dialect%':>9} {'Cultural%':>10} {'Normalize%':>11} {'CodeSwitch%':>12}")
print("-"*72)
for name, res in all_results.items():
    qa  = str(res['aradice_qatar'])+"%" if res['aradice_qatar'] is not None else "—"
    d   = str(res['dialect_overall'])+"%"
    c   = str(res['cultural'])+"%"
    nm  = str(res['normalization'])+"%"
    cs  = str(res['code_switch'])+"%"
    print(f"{name:<16} {qa:>11} {d:>9} {c:>10} {nm:>11} {cs:>12}")
print(f"\nResults saved to {OUT}")
PYEOF
