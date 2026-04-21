#!/bin/bash
#SBATCH --job-name=kd_eval_phd
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/eval_phd_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/eval_phd_%j.err
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

python - << 'PYEOF'
import os, json, torch, random, math
import torch.nn.functional as F
from scipy import stats as scipy_stats
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/phd_eval_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":      None,
    "sft_7b":       "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":    "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
}

# 50 parallel pairs — rigorous evaluation
PAIRS = [
    # Factual — easy
    {"en":"What is the capital of France?","ar":"ما هي عاصمة فرنسا؟","kw_en":["paris"],"kw_ar":["باريس","paris"]},
    {"en":"What is the capital of Australia?","ar":"ما هي عاصمة أستراليا؟","kw_en":["canberra"],"kw_ar":["كانبرا","canberra"]},
    {"en":"What is the capital of Japan?","ar":"ما هي عاصمة اليابان؟","kw_en":["tokyo"],"kw_ar":["طوكيو","tokyo"]},
    {"en":"What is the capital of Germany?","ar":"ما هي عاصمة ألمانيا؟","kw_en":["berlin"],"kw_ar":["برلين","berlin"]},
    {"en":"What is the capital of Egypt?","ar":"ما هي عاصمة مصر؟","kw_en":["cairo"],"kw_ar":["القاهرة","cairo"]},
    {"en":"What is the largest ocean?","ar":"ما هو أكبر محيط في العالم؟","kw_en":["pacific"],"kw_ar":["هادئ","pacific"]},
    {"en":"What is the largest continent?","ar":"ما هي أكبر قارة؟","kw_en":["asia"],"kw_ar":["آسيا","asia"]},
    {"en":"How many days in a week?","ar":"كم عدد أيام الأسبوع؟","kw_en":["7","seven"],"kw_ar":["7","سبعة"]},
    {"en":"How many months in a year?","ar":"كم عدد أشهر السنة؟","kw_en":["12","twelve"],"kw_ar":["12","اثنا عشر"]},
    {"en":"What color is the sky?","ar":"ما لون السماء؟","kw_en":["blue"],"kw_ar":["أزرق","blue"]},
    # Math
    {"en":"What is 15 multiplied by 7?","ar":"ما هو ناتج ضرب 15 في 7؟","kw_en":["105"],"kw_ar":["105"]},
    {"en":"What is 17 multiplied by 13?","ar":"ما هو ناتج ضرب 17 في 13؟","kw_en":["221"],"kw_ar":["221"]},
    {"en":"What is the square root of 144?","ar":"ما هو الجذر التربيعي لـ 144؟","kw_en":["12"],"kw_ar":["12"]},
    {"en":"What is the square root of 169?","ar":"ما هو الجذر التربيعي لـ 169؟","kw_en":["13"],"kw_ar":["13"]},
    {"en":"What is 25 percent of 200?","ar":"ما هو 25 بالمئة من 200؟","kw_en":["50"],"kw_ar":["50"]},
    {"en":"What is 8 squared?","ar":"ما هو مربع العدد 8؟","kw_en":["64"],"kw_ar":["64"]},
    {"en":"What is 120 divided by 6?","ar":"ما هو ناتج قسمة 120 على 6؟","kw_en":["20"],"kw_ar":["20"]},
    # Science
    {"en":"What is the chemical formula for water?","ar":"ما هي الصيغة الكيميائية للماء؟","kw_en":["h2o"],"kw_ar":["h2o","هـ2أو"]},
    {"en":"What is the boiling point of water in Celsius?","ar":"ما درجة غليان الماء بالمئوية؟","kw_en":["100"],"kw_ar":["100"]},
    {"en":"How many bones in adult human body?","ar":"كم عدد عظام جسم الإنسان البالغ؟","kw_en":["206"],"kw_ar":["206"]},
    {"en":"What planet is closest to the sun?","ar":"ما أقرب كوكب إلى الشمس؟","kw_en":["mercury"],"kw_ar":["عطارد","mercury"]},
    {"en":"What gas do plants absorb?","ar":"ما الغاز الذي تمتصه النباتات؟","kw_en":["carbon","co2"],"kw_ar":["كربون","co2","ثاني أكسيد"]},
    {"en":"What is the speed of light approximately?","ar":"ما هي سرعة الضوء تقريباً؟","kw_en":["300000","3×10"],"kw_ar":["300000","3×10"]},
    # History
    {"en":"In what year did World War II end?","ar":"في أي عام انتهت الحرب العالمية الثانية؟","kw_en":["1945"],"kw_ar":["1945"]},
    {"en":"In what year did the Berlin Wall fall?","ar":"في أي عام سقط جدار برلين؟","kw_en":["1989"],"kw_ar":["1989"]},
    {"en":"Who wrote Romeo and Juliet?","ar":"من كتب مسرحية روميو وجولييت؟","kw_en":["shakespeare"],"kw_ar":["شكسبير","shakespeare"]},
    {"en":"Who invented the telephone?","ar":"من اخترع الهاتف؟","kw_en":["bell"],"kw_ar":["بيل","bell"]},
    {"en":"In which continent is Egypt?","ar":"في أي قارة تقع مصر؟","kw_en":["africa"],"kw_ar":["أفريقيا","africa"]},
    # Reasoning
    {"en":"If a train travels 120km in 2 hours what is its speed?","ar":"إذا سافر قطار 120كم في ساعتين ما سرعته؟","kw_en":["60"],"kw_ar":["60"]},
    {"en":"What comes next: 2 4 8 16 ?","ar":"ما الرقم التالي: 2 4 8 16 ؟","kw_en":["32"],"kw_ar":["32"]},
    {"en":"What comes next: 1 1 2 3 5 ?","ar":"ما الرقم التالي: 1 1 2 3 5 ؟","kw_en":["8"],"kw_ar":["8"]},
    {"en":"If today is Monday what day is it in 3 days?","ar":"إذا كان اليوم الاثنين ما هو اليوم بعد 3 أيام؟","kw_en":["thursday"],"kw_ar":["خميس","thursday"]},
    # GCC Domain
    {"en":"What does CBB stand for in Bahrain?","ar":"ما اختصار CBB في البحرين؟","kw_en":["central bank"],"kw_ar":["مركزي","central"]},
    {"en":"What is IBAN used for in banking?","ar":"ما استخدام رقم IBAN في البنوك؟","kw_en":["bank","transfer","account"],"kw_ar":["حساب","تحويل","مصرف"]},
    {"en":"What is artificial intelligence?","ar":"ما هو الذكاء الاصطناعي؟","kw_en":["machine","learn","computer"],"kw_ar":["آلة","تعلم","حاسب"]},
    {"en":"What is machine learning?","ar":"ما هو تعلم الآلة؟","kw_en":["data","learn","pattern"],"kw_ar":["بيانات","تعلم","نمط"]},
    {"en":"What is deep learning?","ar":"ما هو التعلم العميق؟","kw_en":["neural","network","deep"],"kw_ar":["عصبي","شبكة","طبقات"]},
    {"en":"What is a neural network?","ar":"ما هي الشبكة العصبية؟","kw_en":["neuron","layer","brain"],"kw_ar":["عصبون","طبقة","دماغ"]},
    # Bahraini Dialect
    {"en":"What does wayyid mean in Bahraini dialect?","ar":"ما معنى وايد في اللهجة البحرينية؟","kw_en":["much","lot","very"],"kw_ar":["كثير","جدا","وفير"]},
    {"en":"What does shloon mean in Gulf Arabic?","ar":"ما معنى شلون في اللهجة الخليجية؟","kw_en":["how","what"],"kw_ar":["كيف","حال"]},
    {"en":"What does zain mean in Bahraini?","ar":"ما معنى زين في اللهجة البحرينية؟","kw_en":["good","nice","ok"],"kw_ar":["جيد","حسن","جميل"]},
    # Normalization
    {"en":"Translate to MSA Arabic: ana wayyid mabsoot","ar":"حول للفصحى: أنا وايد مبسوط","kw_en":["happy","pleased","very"],"kw_ar":["سعيد","مسرور","جداً"]},
    {"en":"What is the MSA equivalent of ma agdar?","ar":"ما مقابل 'ما أقدر' في الفصحى؟","kw_en":["cannot","unable","not able"],"kw_ar":["لا أستطيع","لا أقدر","عاجز"]},
    # Mixed language understanding  
    {"en":"What is Knowledge Distillation in AI?","ar":"ما هو تقطير المعرفة في الذكاء الاصطناعي؟","kw_en":["teacher","student","compress","transfer"],"kw_ar":["معلم","طالب","ضغط","نقل"]},
    {"en":"What is LoRA in machine learning?","ar":"ما هو LoRA في تعلم الآلة؟","kw_en":["rank","adapt","fine","lora"],"kw_ar":["رتبة","تكيف","ضبط","lora"]},
    {"en":"What is temperature in language models?","ar":"ما هي درجة الحرارة في نماذج اللغة؟","kw_en":["random","softmax","distribut","generat"],"kw_ar":["عشوائي","توزيع","توليد","softmax"]},
    {"en":"What is the difference between SFT and KD?","ar":"ما الفرق بين SFT و KD؟","kw_en":["supervised","distill","teacher","label"],"kw_ar":["خاضع","معلم","تقطير","تدريب"]},
    {"en":"What is quantization in neural networks?","ar":"ما هو التكميم في الشبكات العصبية؟","kw_en":["bit","compress","4bit","int"],"kw_ar":["بت","ضغط","4bit","تكميم"]},
    {"en":"What is fine-tuning in LLMs?","ar":"ما هو الضبط الدقيق في نماذج اللغة الكبيرة؟","kw_en":["train","adapt","task","fine"],"kw_ar":["تدريب","تكيف","مهمة","ضبط"]},
    {"en":"What is the Bahrain Vision 2030?","ar":"ما هي رؤية البحرين 2030؟","kw_en":["economy","develop","future","bahrain"],"kw_ar":["اقتصاد","تطوير","مستقبل","بحرين"]},
]

SEEDS = [42, 123, 777]

def compute_ece(confs, accs, n_bins=10):
    bins = torch.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs>=bins[i])&(confs<bins[i+1])
        if mask.sum()==0: continue
        ece += (mask.sum().item()/len(confs))*abs(accs[mask].float().mean().item()-confs[mask].mean().item())
    return round(ece*100,2)

def cohen_d(a,b):
    na,nb=len(a),len(b)
    pooled=math.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return abs(np.mean(a)-np.mean(b))/pooled if pooled>0 else 0

def load_model(ckpt):
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
    tok=AutoTokenizer.from_pretrained(BASE,token=TOKEN)
    base=AutoModelForCausalLM.from_pretrained(BASE,quantization_config=bnb,device_map={"":0},token=TOKEN)
    model=PeftModel.from_pretrained(base,ckpt) if ckpt else base
    model.eval()
    return tok,model

def generate_conf(tok,model,q,seed=42,max_new=150):
    torch.manual_seed(seed)
    msgs=[{"role":"user","content":q}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt",truncation=True,max_length=512).to("cuda:0")
    with torch.no_grad():
        out=model.generate(**inputs,max_new_tokens=max_new,do_sample=False,
                           pad_token_id=tok.eos_token_id,output_scores=True,
                           return_dict_in_generate=True)
    resp=tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    conf=sum(F.softmax(s,dim=-1).max().item() for s in out.scores)/max(len(out.scores),1) if out.scores else 0.5
    return resp,conf

def score(resp,kws):
    r=resp.lower()
    return any(k.lower() in r for k in kws)

all_results = {}

for model_name,ckpt in CKPTS.items():
    print(f"\n{'='*55}\nEvaluating: {model_name}\n{'='*55}")
    tok,model=load_model(ckpt)

    seed_en_accs,seed_ar_accs,seed_lpgs,seed_clcs=[],[],[],[]
    seed_ece_en,seed_ece_ar,seed_conf_gaps=[],[],[]

    for seed in SEEDS:
        print(f"  Seed {seed}...")
        en_ok,ar_ok=[],[]
        en_confs,ar_confs=[],[]
        clcs_n=0

        for pair in PAIRS:
            er,ec=generate_conf(tok,model,pair["en"],seed)
            ar_r,ac=generate_conf(tok,model,pair["ar"],seed)
            e_ok=score(er,pair["kw_en"])
            a_ok=score(ar_r,pair["kw_ar"])
            en_ok.append(e_ok); ar_ok.append(a_ok)
            en_confs.append(ec); ar_confs.append(ac)
            if e_ok and a_ok: clcs_n+=1

        n=len(PAIRS)
        en_a=sum(en_ok)/n*100; ar_a=sum(ar_ok)/n*100
        lpg=abs(en_a-ar_a); clcs=clcs_n/n*100

        ec_t=torch.tensor(en_confs); ac_t=torch.tensor(ar_confs)
        eo_t=torch.tensor(en_ok); ao_t=torch.tensor(ar_ok)
        ece_en=compute_ece(ec_t,eo_t); ece_ar=compute_ece(ac_t,ao_t)
        conf_gap=sum(en_confs)/n*100-sum(ar_confs)/n*100

        seed_en_accs.append(en_a); seed_ar_accs.append(ar_a)
        seed_lpgs.append(lpg); seed_clcs.append(clcs)
        seed_ece_en.append(ece_en); seed_ece_ar.append(ece_ar)
        seed_conf_gaps.append(conf_gap)

        print(f"    EN={en_a:.1f}% AR={ar_a:.1f}% LPG={lpg:.1f}% CLCS={clcs:.1f}% conf_gap={conf_gap:.1f}%")

    # Mean ± std
    summary={
        "en_accuracy":  f"{np.mean(seed_en_accs):.1f} ± {np.std(seed_en_accs):.1f}",
        "ar_accuracy":  f"{np.mean(seed_ar_accs):.1f} ± {np.std(seed_ar_accs):.1f}",
        "LPG":          f"{np.mean(seed_lpgs):.1f} ± {np.std(seed_lpgs):.1f}",
        "CLCS":         f"{np.mean(seed_clcs):.1f} ± {np.std(seed_clcs):.1f}",
        "ECE_english":  f"{np.mean(seed_ece_en):.2f} ± {np.std(seed_ece_en):.2f}",
        "ECE_arabic":   f"{np.mean(seed_ece_ar):.2f} ± {np.std(seed_ece_ar):.2f}",
        "conf_gap":     f"{np.mean(seed_conf_gaps):.1f} ± {np.std(seed_conf_gaps):.1f}",
        "raw_en":seed_en_accs,"raw_ar":seed_ar_accs,"raw_lpg":seed_lpgs,
    }

    print(f"\n  EN:  {summary['en_accuracy']}%")
    print(f"  AR:  {summary['ar_accuracy']}%")
    print(f"  LPG: {summary['LPG']}%")
    print(f"  CLCS:{summary['CLCS']}%")

    all_results[model_name]={"summary":summary}
    del model; torch.cuda.empty_cache()

# Statistical tests: base vs token_kd
print("\n\nSTATISTICAL SIGNIFICANCE TESTS:")
if "base_7b" in all_results and "token_kd_7b" in all_results:
    base_en=all_results["base_7b"]["summary"]["raw_en"]
    kd_en  =all_results["token_kd_7b"]["summary"]["raw_en"]
    base_ar=all_results["base_7b"]["summary"]["raw_ar"]
    kd_ar  =all_results["token_kd_7b"]["summary"]["raw_ar"]

    t_en,p_en=scipy_stats.ttest_rel(base_en,kd_en)
    t_ar,p_ar=scipy_stats.ttest_rel(base_ar,kd_ar)
    d_en=cohen_d(base_en,kd_en)
    d_ar=cohen_d(base_ar,kd_ar)

    print(f"  base vs token_kd — English: t={t_en:.3f} p={p_en:.3f} Cohen's d={d_en:.3f}")
    print(f"  base vs token_kd — Arabic:  t={t_ar:.3f} p={p_ar:.3f} Cohen's d={d_ar:.3f}")
    all_results["stats"]={
        "base_vs_tokenKD_EN":{"t":t_en,"p":p_en,"cohen_d":d_en},
        "base_vs_tokenKD_AR":{"t":t_ar,"p":p_ar,"cohen_d":d_ar},
    }

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(all_results,f,ensure_ascii=False,indent=2)

print(f"\nFINAL PhD-LEVEL TABLE (mean ± std, n=3 seeds, 50 pairs):")
print(f"Model            EN%              AR%              LPG%             CLCS%")
print("-"*75)
for name,res in all_results.items():
    if name=="stats": continue
    s=res["summary"]
    print(f"{name:<16} {s['en_accuracy']:>14}  {s['ar_accuracy']:>14}  {s['LPG']:>14}  {s['CLCS']:>14}")
print(f"\nResults saved to {OUT}")
PYEOF
