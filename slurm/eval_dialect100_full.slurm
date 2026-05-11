#!/bin/bash
#SBATCH --job-name=kd_d100
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/d100_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/d100_%j.err
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
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/dialect100_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":     None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# 100 prompts with strict answer keywords
DIALECT_ID = [
    {"q":"شلونك اليوم؟","kw":["خليجي","بحريني","gulf","خليج","شلون"]},
    {"q":"وايد حار اليوم","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين بروح البحر","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون الشغل؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد تعبان","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين نخلص ونمشي","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون الدراسة معاك؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد زحمة الشارع","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين الجو تغير","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون الأهل؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد حلو المكان","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين خل نبدأ","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون الوضع؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد صعب الموضوع","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين عندي شغل","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون الصحة؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد سريع الإنترنت","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين بنشوف الحل","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون كان الامتحان؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد غالي السعر","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين الوقت مناسب","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلونك من زمان؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد مبسوط اليوم","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين خلصنا","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون الأمور؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد تعبنا اليوم","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين نراجع","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلون المشروع؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد ممتاز","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين نبدأ الشغل","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
    {"q":"شلونك الحين؟","kw":["خليجي","بحريني","gulf","شلون","خليج"]},
    {"q":"وايد واضح","kw":["خليجي","بحريني","gulf","وايد","خليج"]},
    {"q":"الحين خل نحل المشكلة","kw":["خليجي","بحريني","gulf","الحين","خليج"]},
]

NORMALIZE = [
    {"q":"حول للفصحى: شلونك","kw":["كيف حالك","كيف","حال"]},
    {"q":"حول للفصحى: وايد حلو","kw":["جميل جداً","جميل","جداً","كثيراً"]},
    {"q":"حول للفصحى: الحين بروح","kw":["الآن سأذهب","الآن","أذهب","سأذهب"]},
    {"q":"حول للفصحى: شلون الدراسة","kw":["كيف الدراسة","كيف","دراسة"]},
    {"q":"حول للفصحى: وايد صعب","kw":["صعب جداً","صعب","جداً","كثيراً"]},
    {"q":"حول للفصحى: الحين نبدأ","kw":["الآن نبدأ","الآن","نبدأ"]},
    {"q":"حول للفصحى: شلون الوضع","kw":["كيف الوضع","كيف","الوضع"]},
    {"q":"حول للفصحى: وايد زحمة","kw":["ازدحام شديد","ازدحام","كثير","شديد"]},
    {"q":"حول للفصحى: الحين نخلص","kw":["الآن ننهي","الآن","ننهي","نكمل"]},
    {"q":"حول للفصحى: شلونك اليوم","kw":["كيف حالك اليوم","كيف","حال"]},
    {"q":"حول للفصحى: وايد غالي","kw":["غالي جداً","غالي","جداً","مرتفع"]},
    {"q":"حول للفصحى: الحين نراجع","kw":["الآن نراجع","الآن","نراجع"]},
    {"q":"حول للفصحى: شلون المشروع","kw":["كيف المشروع","كيف","المشروع"]},
    {"q":"حول للفصحى: وايد ممتاز","kw":["ممتاز جداً","ممتاز","جداً","رائع"]},
    {"q":"حول للفصحى: وايد تعبان","kw":["متعب جداً","تعب","جداً","كثيراً"]},
    {"q":"حول للفصحى: الحين خل نبدأ","kw":["الآن لنبدأ","الآن","لنبدأ","نبدأ"]},
    {"q":"حول للفصحى: شلون كان الامتحان","kw":["كيف كان الامتحان","كيف","امتحان"]},
    {"q":"حول للفصحى: وايد واضح","kw":["واضح جداً","واضح","جداً","كثيراً"]},
    {"q":"حول للفصحى: شلونك من زمان","kw":["كيف حالك منذ","كيف","زمن","مدة"]},
    {"q":"حول للفصحى: وايد حار","kw":["حار جداً","حار","جداً","شديد"]},
    {"q":"حول للفصحى: الحين بروح السوق","kw":["الآن سأذهب للسوق","الآن","سوق","أذهب"]},
    {"q":"حول للفصحى: شلون الصحة","kw":["كيف الصحة","كيف","صحة","حال"]},
    {"q":"حول للفصحى: وايد سريع","kw":["سريع جداً","سريع","جداً","كثيراً"]},
    {"q":"حول للفصحى: الحين خلصنا","kw":["الآن انتهينا","الآن","انتهينا","أنهينا"]},
    {"q":"حول للفصحى: شلون الشغل","kw":["كيف العمل","كيف","عمل","شغل"]},
    {"q":"حول للفصحى: وايد مبسوط","kw":["سعيد جداً","سعيد","جداً","مسرور"]},
    {"q":"حول للفصحى: الحين نبدأ الشغل","kw":["الآن نبدأ العمل","الآن","نبدأ","عمل"]},
    {"q":"حول للفصحى: شلونك الحين","kw":["كيف حالك الآن","كيف","الآن","حال"]},
    {"q":"حول للفصحى: وايد صعب الموضوع","kw":["الموضوع صعب جداً","صعب","جداً","موضوع"]},
    {"q":"حول للفصحى: الحين نحل المشكلة","kw":["الآن نحل المشكلة","الآن","نحل","مشكلة"]},
    {"q":"حول للفصحى: وايد بارد","kw":["بارد جداً","بارد","جداً","كثيراً"]},
    {"q":"حول للفصحى: الحين عندي موعد","kw":["الآن لدي موعد","الآن","موعد","لدي"]},
    {"q":"حول للفصحى: شلون الأهل","kw":["كيف حال الأهل","كيف","أهل","عائلة"]},
]

CODE_SWITCH = [
    {"q":"الحين نقدر deploy النظام؟","kw":["نشر","تشغيل","deploy","نظام","يمكننا"]},
    {"q":"في مشكلة في system لازم نحلها","kw":["مشكلة","نظام","system","حل","نحتاج"]},
    {"q":"وايد slow الresponse","kw":["بطيء","استجابة","response","slow","أداء"]},
    {"q":"الحين نعمل update للنظام","kw":["تحديث","update","نظام","الآن","نقوم"]},
    {"q":"في bug في الكود لازم fix","kw":["خطأ","bug","fix","كود","إصلاح"]},
    {"q":"شلون الAPI يشتغل؟","kw":["كيف","API","يعمل","يشتغل","واجهة"]},
    {"q":"وايد good النتيجة","kw":["جيدة","نتيجة","good","ممتازة","رائعة"]},
    {"q":"الحين نبدأ testing","kw":["اختبار","testing","نبدأ","الآن","فحص"]},
    {"q":"في error في البرنامج","kw":["خطأ","error","برنامج","مشكلة","أخطاء"]},
    {"q":"شلون الdatabase متصل؟","kw":["كيف","database","متصل","قاعدة","بيانات"]},
    {"q":"وايد fast النظام","kw":["سريع","fast","نظام","جداً","أداء"]},
    {"q":"الحين نdeploy النسخة","kw":["نشر","deploy","نسخة","الآن","تشغيل"]},
    {"q":"في مشكلة في server","kw":["مشكلة","server","خادم","أعطال","توقف"]},
    {"q":"شلون الfrontend؟","kw":["كيف","frontend","واجهة","المستخدم","يعمل"]},
    {"q":"وايد clean الكود","kw":["نظيف","clean","كود","ممتاز","جيد"]},
    {"q":"الحين نعمل debug","kw":["تصحيح","debug","الآن","نقوم","أخطاء"]},
    {"q":"في issue في login","kw":["مشكلة","issue","login","دخول","تسجيل"]},
    {"q":"شلون الbackend؟","kw":["كيف","backend","الخلفية","يعمل","خادم"]},
    {"q":"وايد stable النظام","kw":["مستقر","stable","نظام","جداً","ممتاز"]},
    {"q":"الحين نpush التحديث","kw":["رفع","push","تحديث","الآن","نقوم"]},
    {"q":"في problem في network","kw":["مشكلة","problem","network","شبكة","اتصال"]},
    {"q":"شلون الUI؟","kw":["كيف","UI","واجهة","يبدو","تصميم"]},
    {"q":"وايد smooth الأداء","kw":["سلس","smooth","أداء","جداً","ممتاز"]},
    {"q":"الحين نعمل restart","kw":["إعادة تشغيل","restart","الآن","نقوم","النظام"]},
    {"q":"في bug في API","kw":["خطأ","bug","API","مشكلة","إصلاح"]},
    {"q":"شلون الdesign؟","kw":["كيف","design","تصميم","يبدو","جميل"]},
    {"q":"وايد responsive الموقع","kw":["متجاوب","responsive","موقع","جداً","يعمل"]},
    {"q":"الحين نعمل test case","kw":["اختبار","test","الآن","نقوم","حالة"]},
    {"q":"في crash في النظام","kw":["توقف","crash","نظام","مشكلة","خلل"]},
    {"q":"شلون الsecurity؟","kw":["كيف","security","أمان","حماية","يعمل"]},
    {"q":"وايد safe التطبيق","kw":["آمن","safe","تطبيق","جداً","محمي"]},
    {"q":"الحين نعمل deploy جديد","kw":["نشر","deploy","جديد","الآن","نقوم"]},
    {"q":"في latency عالي","kw":["تأخير","latency","عالي","مشكلة","بطيء"]},
    {"q":"خلنا schedule meeting الحين","kw":["اجتماع","meeting","schedule","الآن","نرتب"]},
]

def has_cjk(text):
    return any('\u4e00' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7a3' for c in text)

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
    if has_cjk(resp): return False  # hard fail CJK
    if len(resp.strip()) < 10: return False  # too short
    return any(k.lower() in resp.lower() for k in kws)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue

    tok, model = load_model(ckpt)

    did_ok=0; norm_ok=0; cs_ok=0
    cjk_count=0; details=[]

    prefix_did  = "حدد إذا كانت هذه الجملة لهجة خليجية بحرينية وأذكر السبب: "
    prefix_norm = ""  # already has instruction
    prefix_cs   = ""  # already has instruction

    for item in tqdm(DIALECT_ID, desc="Dialect ID"):
        resp = gen(tok, model, prefix_did + item["q"])
        cjk = has_cjk(resp)
        ok  = score(resp, item["kw"])
        if cjk: cjk_count+=1
        if ok: did_ok+=1
        details.append({"cat":"dialect_id","q":item["q"],"resp":resp[:150],"ok":ok,"cjk":cjk})

    for item in tqdm(NORMALIZE, desc="Normalize"):
        resp = gen(tok, model, item["q"])
        cjk = has_cjk(resp)
        ok  = score(resp, item["kw"])
        if cjk: cjk_count+=1
        if ok: norm_ok+=1
        details.append({"cat":"normalize","q":item["q"],"resp":resp[:150],"ok":ok,"cjk":cjk})

    for item in tqdm(CODE_SWITCH, desc="Code-Switch"):
        resp = gen(tok, model, item["q"])
        cjk = has_cjk(resp)
        ok  = score(resp, item["kw"])
        if cjk: cjk_count+=1
        if ok: cs_ok+=1
        details.append({"cat":"code_switch","q":item["q"],"resp":resp[:150],"ok":ok,"cjk":cjk})

    n_did=len(DIALECT_ID); n_norm=len(NORMALIZE); n_cs=len(CODE_SWITCH)
    total=n_did+n_norm+n_cs

    did_pct  = round(did_ok/n_did*100,1)
    norm_pct = round(norm_ok/n_norm*100,1)
    cs_pct   = round(cs_ok/n_cs*100,1)
    overall  = round((did_ok+norm_ok+cs_ok)/total*100,1)

    print(f"\n  Dialect ID:   {did_pct}%  ({did_ok}/{n_did})")
    print(f"  Normalize:    {norm_pct}%  ({norm_ok}/{n_norm})")
    print(f"  Code-Switch:  {cs_pct}%  ({cs_ok}/{n_cs})")
    print(f"  Overall:      {overall}%")
    print(f"  CJK failures: {cjk_count}/{total}")

    all_results[model_name] = {
        "dialect_id": did_pct, "normalize": norm_pct,
        "code_switch": cs_pct, "overall": overall,
        "cjk_count": cjk_count, "details": details
    }
    del model; torch.cuda.empty_cache()

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nTABLE 4.2 — BAHRAINI DIALECT EVALUATION (100 prompts, CJK-filtered):")
print(f"{'Model':<16} {'Dialect ID%':>12} {'Normalize%':>11} {'Code-Switch%':>13} {'Overall%':>9} {'CJK fails':>10}")
print("-"*75)
for name,res in all_results.items():
    star=" ★" if name=="selectkd_7b" else ""
    print(f"{name+star:<18} {str(res['dialect_id'])+'%':>12} {str(res['normalize'])+'%':>11} {str(res['code_switch'])+'%':>13} {str(res['overall'])+'%':>9} {res['cjk_count']:>10}")
PYEOF
