#!/bin/bash
#SBATCH --job-name=eval_sft_seq
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/eval_sft_seq_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/eval_sft_seq_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""
export PYTORCH_ALLOC_CONF=expandable_segments:True

python - << 'PYEOF'
import os, json, torch, math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/sft_seqkd_eval.json"

CKPTS = {
    "sft_7b":    "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
}

PAIRS = [
    ("What is the capital of France?","ما هي عاصمة فرنسا؟",["paris"],["باريس"]),
    ("What is 15 multiplied by 7?","ما ناتج 15 في 7؟",["105"],["105"]),
    ("What is the largest ocean?","ما أكبر محيط؟",["pacific"],["هادئ"]),
    ("How many bones in human body?","كم عدد عظام الإنسان؟",["206"],["206"]),
    ("What year did Berlin Wall fall?","متى سقط جدار برلين؟",["1989"],["1989"]),
    ("Who wrote Romeo and Juliet?","من كتب روميو وجولييت؟",["shakespeare"],["شكسبير"]),
    ("What is the boiling point of water?","ما درجة غليان الماء؟",["100"],["100"]),
    ("What is machine learning?","ما هو تعلم الآلة؟",["data","learn"],["بيانات","تعلم"]),
    ("What does CBB stand for in Bahrain?","ما اختصار CBB في البحرين؟",["central bank"],["مركزي"]),
    ("What is the capital of Japan?","ما عاصمة اليابان؟",["tokyo"],["طوكيو"]),
    ("What is 17 multiplied by 13?","ما ناتج 17 في 13؟",["221"],["221"]),
    ("What is artificial intelligence?","ما هو الذكاء الاصطناعي؟",["machine","learn"],["آلة","تعلم"]),
    ("What does wayyid mean in Bahraini?","ما معنى وايد في البحريني؟",["much","very","lot"],["كثير","جدا"]),
    ("What is LoRA?","ما هو LoRA؟",["rank","adapt"],["رتبة","تكيف"]),
    ("What is the capital of Saudi Arabia?","ما عاصمة المملكة العربية السعودية؟",["riyadh"],["الرياض"]),
    ("How many days in a week?","كم عدد أيام الأسبوع؟",["7","seven"],["7","سبعة"]),
    ("What is Bahrain Vision 2030?","ما هي رؤية البحرين 2030؟",["economy","develop"],["اقتصاد","تطوير"]),
    ("What is 8 multiplied by 9?","ما ناتج 8 في 9؟",["72"],["72"]),
    ("What is photosynthesis?","ما هو التركيب الضوئي؟",["plant","sun"],["نبات","شمس"]),
    ("What is the capital of UAE?","ما عاصمة الإمارات؟",["abu dhabi"],["أبوظبي"]),
    ("How many continents are there?","كم عدد القارات؟",["7","seven"],["7","سبع"]),
    ("What is GDP?","ما هو الناتج المحلي الإجمالي؟",["gross","product"],["اقتصاد","ناتج"]),
    ("What is 100 divided by 4?","ما ناتج 100 ÷ 4؟",["25"],["25"]),
    ("What is the largest country by area?","ما أكبر دولة مساحة؟",["russia"],["روسيا"]),
    ("What is Python programming?","ما هو برمجة بايثون؟",["program","code"],["برمجة","لغة"]),
    ("Who invented the telephone?","من اخترع الهاتف؟",["bell"],["بيل"]),
    ("What is the speed of light?","ما سرعة الضوء؟",["300"],["300"]),
    ("What language is spoken in Bahrain?","ما اللغة المستخدمة في البحرين؟",["arabic"],["عربية"]),
    ("What is the Quran?","ما هو القرآن؟",["islam","holy"],["إسلام","كتاب"]),
    ("What is Knowledge Distillation?","ما هو تقطير المعرفة؟",["teacher","student"],["معلم","طالب"]),
]

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()
    return tok, model

def gen(tok, model, q):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score(resp, kws):
    return any(k.lower() in resp.lower() for k in kws)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    tok, model = load_model(ckpt)
    en_ok=0; ar_ok=0; clcs=0

    for en_q, ar_q, en_kw, ar_kw in tqdm(PAIRS, desc="30-pair eval"):
        er  = gen(tok, model, en_q)
        arr = gen(tok, model, ar_q)
        e_ok = score(er, en_kw)
        a_ok = score(arr, ar_kw)
        if e_ok: en_ok+=1
        if a_ok: ar_ok+=1
        if e_ok and a_ok: clcs+=1

    n = len(PAIRS)
    en_pct  = round(en_ok/n*100,1)
    ar_pct  = round(ar_ok/n*100,1)
    lpg     = round(abs(en_pct-ar_pct),1)
    clcs_pct= round(clcs/n*100,1)

    print(f"EN={en_pct}% AR={ar_pct}% LPG={lpg}% CLCS={clcs_pct}%")
    all_results[model_name] = {"EN":en_pct,"AR":ar_pct,"LPG":lpg,"CLCS":clcs_pct}
    del model; torch.cuda.empty_cache()

with open(OUT,"w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: {OUT}")
print("\nRESULTS:")
for name,res in all_results.items():
    print(f"{name}: EN={res['EN']}% AR={res['AR']}% LPG={res['LPG']}% CLCS={res['CLCS']}%")
PYEOF
