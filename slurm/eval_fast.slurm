#!/bin/bash
#SBATCH --job-name=kd_eval_fast
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/eval_fast_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/eval_fast_%j.err
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
import os, json, torch, math, numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/final_eval.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":      None,
    "sft_7b":       "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":    "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# 30 parallel pairs — fast but meaningful
PAIRS = [
    {"en":"What is the capital of France?","ar":"ما هي عاصمة فرنسا؟","kw_en":["paris"],"kw_ar":["باريس","paris"]},
    {"en":"What is the capital of Japan?","ar":"ما هي عاصمة اليابان؟","kw_en":["tokyo"],"kw_ar":["طوكيو","tokyo"]},
    {"en":"What is the capital of Egypt?","ar":"ما هي عاصمة مصر؟","kw_en":["cairo"],"kw_ar":["القاهرة","cairo"]},
    {"en":"What is the largest ocean?","ar":"ما هو أكبر محيط؟","kw_en":["pacific"],"kw_ar":["هادئ","pacific"]},
    {"en":"How many days in a week?","ar":"كم عدد أيام الأسبوع؟","kw_en":["7","seven"],"kw_ar":["7","سبعة"]},
    {"en":"What is 15 multiplied by 7?","ar":"ما ناتج 15 في 7؟","kw_en":["105"],"kw_ar":["105"]},
    {"en":"What is 17 multiplied by 13?","ar":"ما ناتج 17 في 13؟","kw_en":["221"],"kw_ar":["221"]},
    {"en":"What is the square root of 144?","ar":"ما الجذر التربيعي لـ 144؟","kw_en":["12"],"kw_ar":["12"]},
    {"en":"What is 25 percent of 200?","ar":"ما هو 25% من 200؟","kw_en":["50"],"kw_ar":["50"]},
    {"en":"What is the chemical formula for water?","ar":"ما الصيغة الكيميائية للماء؟","kw_en":["h2o"],"kw_ar":["h2o"]},
    {"en":"What is the boiling point of water?","ar":"ما درجة غليان الماء؟","kw_en":["100"],"kw_ar":["100"]},
    {"en":"How many bones in human body?","ar":"كم عدد عظام جسم الإنسان؟","kw_en":["206"],"kw_ar":["206"]},
    {"en":"In what year did World War II end?","ar":"في أي عام انتهت الحرب العالمية الثانية؟","kw_en":["1945"],"kw_ar":["1945"]},
    {"en":"In what year did Berlin Wall fall?","ar":"في أي عام سقط جدار برلين؟","kw_en":["1989"],"kw_ar":["1989"]},
    {"en":"Who wrote Romeo and Juliet?","ar":"من كتب روميو وجولييت؟","kw_en":["shakespeare"],"kw_ar":["شكسبير","shakespeare"]},
    {"en":"What is the speed of light?","ar":"ما سرعة الضوء؟","kw_en":["300000","3×10"],"kw_ar":["300000"]},
    {"en":"What planet is closest to the sun?","ar":"ما أقرب كوكب للشمس؟","kw_en":["mercury"],"kw_ar":["عطارد","mercury"]},
    {"en":"If train goes 120km in 2h what is speed?","ar":"قطار يقطع 120كم في ساعتين ما سرعته؟","kw_en":["60"],"kw_ar":["60"]},
    {"en":"What comes next: 2 4 8 16?","ar":"ما الرقم التالي: 2 4 8 16؟","kw_en":["32"],"kw_ar":["32"]},
    {"en":"What does CBB stand for in Bahrain?","ar":"ما اختصار CBB في البحرين؟","kw_en":["central bank"],"kw_ar":["مركزي","central"]},
    {"en":"What is IBAN used for?","ar":"ما استخدام رقم IBAN؟","kw_en":["bank","transfer"],"kw_ar":["حساب","تحويل"]},
    {"en":"What is artificial intelligence?","ar":"ما هو الذكاء الاصطناعي؟","kw_en":["machine","learn"],"kw_ar":["آلة","تعلم"]},
    {"en":"What is machine learning?","ar":"ما هو تعلم الآلة؟","kw_en":["data","learn"],"kw_ar":["بيانات","تعلم"]},
    {"en":"What is Knowledge Distillation?","ar":"ما هو تقطير المعرفة؟","kw_en":["teacher","student","compress"],"kw_ar":["معلم","طالب","ضغط"]},
    {"en":"What is LoRA in machine learning?","ar":"ما هو LoRA في تعلم الآلة؟","kw_en":["rank","adapt","lora"],"kw_ar":["رتبة","تكيف","lora"]},
    {"en":"What does wayyid mean in Bahraini?","ar":"ما معنى وايد في البحريني؟","kw_en":["much","lot","very"],"kw_ar":["كثير","جدا"]},
    {"en":"What does shloon mean in Gulf Arabic?","ar":"ما معنى شلون بالخليجي؟","kw_en":["how","what"],"kw_ar":["كيف","حال"]},
    {"en":"What is the Bahrain Vision 2030?","ar":"ما هي رؤية البحرين 2030؟","kw_en":["economy","develop"],"kw_ar":["اقتصاد","تطوير"]},
    {"en":"What is quantization in AI?","ar":"ما هو التكميم في الذكاء الاصطناعي؟","kw_en":["bit","compress","4bit"],"kw_ar":["بت","ضغط","تكميم"]},
    {"en":"What is fine-tuning in LLMs?","ar":"ما هو الضبط الدقيق في LLMs؟","kw_en":["train","adapt","task"],"kw_ar":["تدريب","تكيف","مهمة"]},
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, q, max_new=100):
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
    r = resp.lower()
    return any(k.lower() in r for k in kws)

def ece(confs, accs, n=10):
    bins = torch.linspace(0,1,n+1)
    e = 0.0
    for i in range(n):
        m = (confs>=bins[i])&(confs<bins[i+1])
        if m.sum()==0: continue
        e += (m.sum().item()/len(confs))*abs(accs[m].float().mean().item()-confs[m].mean().item())
    return round(e*100,2)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")

    # Skip selectkd if checkpoint doesn't exist yet
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print(f"  SKIP — checkpoint not ready: {ckpt}")
        continue

    tok, model = load_model(ckpt)
    en_ok, ar_ok, en_c, ar_c, clcs = [], [], [], [], 0

    for pair in tqdm(PAIRS, desc=model_name):
        er, ec = gen(tok, model, pair["en"])
        ar_r, ac = gen(tok, model, pair["ar"])
        e_ok = score(er, pair["kw_en"])
        a_ok = score(ar_r, pair["kw_ar"])
        en_ok.append(e_ok); ar_ok.append(a_ok)
        en_c.append(ec); ar_c.append(ac)
        if e_ok and a_ok: clcs += 1

    n = len(PAIRS)
    en_a = round(sum(en_ok)/n*100, 1)
    ar_a = round(sum(ar_ok)/n*100, 1)
    lpg  = round(abs(en_a-ar_a), 1)
    clcs_pct = round(clcs/n*100, 1)
    avg_en_c = round(sum(en_c)/n*100, 1)
    avg_ar_c = round(sum(ar_c)/n*100, 1)
    conf_gap = round(avg_en_c - avg_ar_c, 1)

    ec_t = torch.tensor(en_c); ac_t = torch.tensor(ar_c)
    eo_t = torch.tensor(en_ok); ao_t = torch.tensor(ar_ok)
    ece_en = ece(ec_t, eo_t); ece_ar = ece(ac_t, ao_t)

    summary = {
        "en_accuracy": en_a, "ar_accuracy": ar_a,
        "LPG": lpg, "CLCS": clcs_pct,
        "ECE_en": ece_en, "ECE_ar": ece_ar,
        "conf_en": avg_en_c, "conf_ar": avg_ar_c,
        "conf_gap": conf_gap
    }

    print(f"  EN={en_a}% AR={ar_a}% LPG={lpg}% CLCS={clcs_pct}%")
    print(f"  ECE_EN={ece_en}% ECE_AR={ece_ar}%")
    print(f"  Conf EN={avg_en_c}% AR={avg_ar_c}% Gap={conf_gap}%")

    all_results[model_name] = summary
    del model; torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nFINAL TABLE:")
print(f"{'Model':<16} {'EN%':>6} {'AR%':>6} {'LPG':>6} {'CLCS':>6} {'ECE_EN':>7} {'ECE_AR':>7} {'Conf_Gap':>9}")
print("-"*70)
for name, s in all_results.items():
    print(f"{name:<16} {s['en_accuracy']:>5}% {s['ar_accuracy']:>5}% {s['LPG']:>5}% {s['CLCS']:>5}% {s['ECE_en']:>6}% {s['ECE_ar']:>6}% {s['conf_gap']:>8}%")

print(f"\nKey finding:")
if "base_7b" in all_results and "token_kd_7b" in all_results:
    base_lpg = all_results["base_7b"]["LPG"]
    kd_lpg   = all_results["token_kd_7b"]["LPG"]
    print(f"  Base LPG: {base_lpg}%  Token-KD LPG: {kd_lpg}%")
if "selectkd_7b" in all_results:
    sel_lpg = all_results["selectkd_7b"]["LPG"]
    print(f"  SelecTKD LPG: {sel_lpg}%")
    print(f"  Hypothesis: SelecTKD reduces LPG further")
PYEOF
