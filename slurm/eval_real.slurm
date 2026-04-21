#!/bin/bash
#SBATCH --job-name=kd_eval_real
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/eval_real_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/eval_real_%j.err
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

echo "Starting REAL evaluation at: $(date)"

python - << 'PYEOF'
import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/real_eval_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":      None,
    "sft_7b":       "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":    "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
}

# ── Harder, more discriminating test set ─────────────────────
# Each item has: question, expected keywords, language, difficulty
PROMPTS = [
    # English factual — harder
    {"q": "What is the capital of Australia?", "kw": ["canberra"], "lang": "en", "diff": "medium"},
    {"q": "What is the square root of 169?", "kw": ["13"], "lang": "en", "diff": "easy"},
    {"q": "Who wrote 1984?", "kw": ["orwell"], "lang": "en", "diff": "medium"},
    {"q": "What is the chemical formula for water?", "kw": ["h2o", "h₂o"], "lang": "en", "diff": "easy"},
    {"q": "In which continent is Egypt located?", "kw": ["africa"], "lang": "en", "diff": "easy"},
    {"q": "What year did the Berlin Wall fall?", "kw": ["1989"], "lang": "en", "diff": "medium"},
    {"q": "What is the largest ocean on Earth?", "kw": ["pacific"], "lang": "en", "diff": "easy"},
    {"q": "How many bones are in the adult human body?", "kw": ["206"], "lang": "en", "diff": "hard"},
    {"q": "What is the powerhouse of the cell?", "kw": ["mitochondria", "mitochondrion"], "lang": "en", "diff": "medium"},
    {"q": "What is 17 multiplied by 13?", "kw": ["221"], "lang": "en", "diff": "medium"},
    # English reasoning
    {"q": "If a train travels 120km in 2 hours, what is its speed in km/h?", "kw": ["60"], "lang": "en", "diff": "medium"},
    {"q": "What comes next in the sequence: 2, 4, 8, 16, ?", "kw": ["32"], "lang": "en", "diff": "easy"},
    {"q": "If all roses are flowers and some flowers fade, can we conclude all roses fade?", "kw": ["no", "cannot", "not necessarily"], "lang": "en", "diff": "hard"},

    # Arabic factual — harder
    {"q": "ما هي عاصمة أستراليا؟", "kw": ["كانبرا", "canberra"], "lang": "ar", "diff": "medium"},
    {"q": "ما هو الجذر التربيعي للعدد 169؟", "kw": ["13"], "lang": "ar", "diff": "easy"},
    {"q": "من كتب رواية 1984؟", "kw": ["أورويل", "orwell"], "lang": "ar", "diff": "medium"},
    {"q": "ما هي الصيغة الكيميائية للماء؟", "kw": ["h2o", "h₂o", "هـ2أو"], "lang": "ar", "diff": "easy"},
    {"q": "في أي قارة تقع مصر؟", "kw": ["أفريقيا", "africa", "إفريقيا"], "lang": "ar", "diff": "easy"},
    {"q": "في أي عام سقط جدار برلين؟", "kw": ["1989"], "lang": "ar", "diff": "medium"},
    {"q": "ما هو أكبر محيط في العالم؟", "kw": ["الهادئ", "pacific", "الهادي"], "lang": "ar", "diff": "easy"},
    {"q": "كم عدد عظام جسم الإنسان البالغ؟", "kw": ["206"], "lang": "ar", "diff": "hard"},
    {"q": "ما هو ناتج ضرب 17 في 13؟", "kw": ["221"], "lang": "ar", "diff": "medium"},
    # Arabic reasoning
    {"q": "إذا سافر قطار 120 كم في ساعتين، ما سرعته بالكيلومتر في الساعة؟", "kw": ["60"], "lang": "ar", "diff": "medium"},
    {"q": "ما الرقم التالي في المتتالية: 2، 4، 8، 16، ؟", "kw": ["32"], "lang": "ar", "diff": "easy"},

    # GCC domain — tests domain knowledge
    {"q": "What does CBB stand for in Bahrain?", "kw": ["central bank", "bahrain"], "lang": "en", "diff": "domain"},
    {"q": "What is IBAN used for?", "kw": ["bank", "account", "transfer", "international"], "lang": "en", "diff": "domain"},
    {"q": "ما اختصار CBB في البحرين؟", "kw": ["مركزي", "البحرين", "central"], "lang": "ar", "diff": "domain"},
    {"q": "ما استخدام رقم IBAN؟", "kw": ["حساب", "تحويل", "bank", "مصرفي"], "lang": "ar", "diff": "domain"},

    # Bahraini dialect — tests dialect knowledge
    {"q": "ما معنى كلمة 'وايد' في اللهجة البحرينية؟", "kw": ["كثير", "جدا", "much", "lot"], "lang": "ar", "diff": "dialect"},
    {"q": "ترجم 'شلونك' للعربية الفصحى", "kw": ["كيف حالك", "كيف", "حال"], "lang": "ar", "diff": "dialect"},
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

def generate(tok, model, question, max_new=200):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score(response, keywords):
    resp_lower = response.lower()
    return any(kw.lower() in resp_lower for kw in keywords)

def is_arabic_response(response):
    return sum(1 for c in response if '\u0600' <= c <= '\u06FF') > len(response) * 0.15

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    tok, model = load_model(ckpt)

    results = []
    en_correct, ar_correct, en_total, ar_total = 0, 0, 0, 0
    lang_correct = 0
    by_difficulty = {}
    by_category = {}

    for item in tqdm(PROMPTS, desc=model_name):
        resp = generate(tok, model, item["q"])
        correct = score(resp, item["kw"])
        lang = item["lang"]

        # Check response language
        resp_is_arabic = is_arabic_response(resp)
        correct_lang = (lang == "ar" and resp_is_arabic) or (lang == "en" and not resp_is_arabic)

        if correct_lang:
            lang_correct += 1

        if lang == "en":
            en_total += 1
            if correct: en_correct += 1
        else:
            ar_total += 1
            if correct: ar_correct += 1

        diff = item.get("diff", "medium")
        by_difficulty[diff] = by_difficulty.get(diff, {"correct":0,"total":0})
        by_difficulty[diff]["total"] += 1
        if correct: by_difficulty[diff]["correct"] += 1

        results.append({
            "question": item["q"],
            "response": resp[:300],
            "expected_keywords": item["kw"],
            "correct": correct,
            "language": lang,
            "correct_language": correct_lang,
            "difficulty": diff
        })

    en_acc = round(en_correct/en_total*100, 1) if en_total > 0 else 0
    ar_acc = round(ar_correct/ar_total*100, 1) if ar_total > 0 else 0
    lpg    = round(abs(en_acc - ar_acc), 1)
    lang_pct = round(lang_correct/len(PROMPTS)*100, 1)

    diff_summary = {k: f"{v['correct']}/{v['total']} ({round(v['correct']/v['total']*100)}%)"
                    for k,v in by_difficulty.items()}

    summary = {
        "english_accuracy": en_acc,
        "arabic_accuracy": ar_acc,
        "LPG": lpg,
        "language_direction": lang_pct,
        "by_difficulty": diff_summary,
        "total_prompts": len(PROMPTS),
    }

    print(f"\n  English accuracy:   {en_acc}% ({en_correct}/{en_total})")
    print(f"  Arabic accuracy:    {ar_acc}% ({ar_correct}/{ar_total})")
    print(f"  LPG:                {lpg}%")
    print(f"  Language direction: {lang_pct}%")
    print(f"  By difficulty: {diff_summary}")

    all_results[model_name] = {"summary": summary, "details": results}

    del model
    torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nDONE! Results saved to {OUT}")
print(f"\n{'='*60}")
print(f"REAL COMPARISON TABLE:")
print(f"{'Model':<20} {'EN':>6} {'AR':>6} {'LPG':>6} {'Lang%':>7} {'Easy':>8} {'Hard':>8}")
print("-"*65)
for name, res in all_results.items():
    s = res["summary"]
    d = s["by_difficulty"]
    easy = d.get("easy","—")
    hard = d.get("hard","—")
    print(f"{name:<20} {s['english_accuracy']:>5}% {s['arabic_accuracy']:>5}% {s['LPG']:>5}% {s['language_direction']:>6}% {easy:>8} {hard:>8}")
PYEOF
