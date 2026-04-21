#!/bin/bash
#SBATCH --job-name=kd_lpg_ece
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/lpg_ece_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/lpg_ece_%j.err
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
import os, json, torch, math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/lpg_ece_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":      None,
    "sft_7b":       "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":    "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
}

# Parallel EN/AR prompt pairs — same question in both languages
# This is the CORE of LPG analysis
PARALLEL_PAIRS = [
    {"en": "What is the capital of Australia?",   "ar": "ما هي عاصمة أستراليا؟",             "kw_en": ["canberra"], "kw_ar": ["كانبرا","canberra"]},
    {"en": "What is the square root of 169?",     "ar": "ما هو الجذر التربيعي للعدد 169؟",   "kw_en": ["13"],       "kw_ar": ["13"]},
    {"en": "Who wrote the novel 1984?",           "ar": "من كتب رواية 1984؟",                 "kw_en": ["orwell"],   "kw_ar": ["أورويل","orwell"]},
    {"en": "In which continent is Egypt?",        "ar": "في أي قارة تقع مصر؟",               "kw_en": ["africa"],   "kw_ar": ["أفريقيا","افريقيا"]},
    {"en": "What year did Berlin Wall fall?",     "ar": "في أي عام سقط جدار برلين؟",         "kw_en": ["1989"],     "kw_ar": ["1989"]},
    {"en": "What is the largest ocean?",          "ar": "ما هو أكبر محيط في العالم؟",         "kw_en": ["pacific"],  "kw_ar": ["هادئ","pacific"]},
    {"en": "How many bones in human body?",       "ar": "كم عدد عظام جسم الإنسان؟",          "kw_en": ["206"],      "kw_ar": ["206"]},
    {"en": "What is 17 multiplied by 13?",        "ar": "ما هو ناتج ضرب 17 في 13؟",          "kw_en": ["221"],      "kw_ar": ["221"]},
    {"en": "What is the chemical formula of water?", "ar": "ما الصيغة الكيميائية للماء؟",    "kw_en": ["h2o"],      "kw_ar": ["h2o","هـ2أو"]},
    {"en": "Speed of a train going 120km in 2h?", "ar": "سرعة قطار يقطع 120كم في ساعتين؟",  "kw_en": ["60"],       "kw_ar": ["60"]},
    # GCC domain pairs
    {"en": "What does CBB stand for in Bahrain?", "ar": "ما اختصار CBB في البحرين؟",         "kw_en": ["central bank"], "kw_ar": ["مركزي","central"]},
    {"en": "What is IBAN used for?",              "ar": "ما استخدام رقم IBAN؟",               "kw_en": ["bank","transfer"], "kw_ar": ["حساب","تحويل"]},
    {"en": "What is machine learning?",           "ar": "ما هو تعلم الآلة؟",                 "kw_en": ["data","learn"], "kw_ar": ["بيانات","تعلم"]},
    {"en": "What is artificial intelligence?",   "ar": "ما هو الذكاء الاصطناعي؟",           "kw_en": ["machine","computer"], "kw_ar": ["آلة","حاسب","ذكاء"]},
    # Bahraini dialect (Arabic only — tests dialect knowledge)
    {"en": "What does 'wayyid' mean in Bahraini?", "ar": "ما معنى كلمة وايد في اللهجة البحرينية؟", "kw_en": ["much","lot","very"], "kw_ar": ["كثير","جدا"]},
]

def compute_ece(confidences, accuracies, n_bins=10):
    """Expected Calibration Error"""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].float().mean().item()
        bin_conf = confidences[mask].mean().item()
        bin_size = mask.sum().item() / len(confidences)
        ece += bin_size * abs(bin_acc - bin_conf)
    return round(ece * 100, 2)

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def generate_with_confidence(tok, model, question, max_new=150):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id, output_scores=True, return_dict_in_generate=True
        )
    response = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Average max confidence across generated tokens
    if out.scores:
        confs = [F.softmax(s, dim=-1).max().item() for s in out.scores]
        avg_conf = sum(confs) / len(confs)
    else:
        avg_conf = 0.5
    return response.strip(), avg_conf

def score(response, keywords):
    r = response.lower()
    return any(kw.lower() in r for kw in keywords)

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*55}")
    print(f"Evaluating: {model_name}")
    print('='*55)
    tok, model = load_model(ckpt)

    en_correct, ar_correct = 0, 0
    en_confs, ar_confs = [], []
    en_accs,  ar_accs  = [], []
    clcs_both_correct  = 0  # Cross-Lingual Consistency Score
    details = []

    for pair in tqdm(PARALLEL_PAIRS, desc=model_name):
        # English
        en_resp, en_conf = generate_with_confidence(tok, model, pair["en"])
        en_ok = score(en_resp, pair["kw_en"])

        # Arabic
        ar_resp, ar_conf = generate_with_confidence(tok, model, pair["ar"])
        ar_ok = score(ar_resp, pair["kw_ar"])

        if en_ok: en_correct += 1
        if ar_ok: ar_correct += 1
        if en_ok and ar_ok: clcs_both_correct += 1

        en_confs.append(en_conf)
        ar_confs.append(ar_conf)
        en_accs.append(en_ok)
        ar_accs.append(ar_ok)

        details.append({
            "en_q": pair["en"], "ar_q": pair["ar"],
            "en_response": en_resp[:200], "ar_response": ar_resp[:200],
            "en_correct": en_ok, "ar_correct": ar_ok,
            "en_confidence": round(en_conf, 4), "ar_confidence": round(ar_conf, 4),
            "consistent": en_ok and ar_ok
        })

    n = len(PARALLEL_PAIRS)
    en_acc  = round(en_correct / n * 100, 1)
    ar_acc  = round(ar_correct / n * 100, 1)
    lpg     = round(abs(en_acc - ar_acc), 1)
    clcs    = round(clcs_both_correct / n * 100, 1)

    # ECE
    en_conf_t = torch.tensor(en_confs)
    ar_conf_t = torch.tensor(ar_confs)
    en_acc_t  = torch.tensor(en_accs)
    ar_acc_t  = torch.tensor(ar_accs)
    ece_en    = compute_ece(en_conf_t, en_acc_t)
    ece_ar    = compute_ece(ar_conf_t, ar_acc_t)

    # Teacher confidence gap — KEY metric
    # Lower AR confidence = Arabic tokens are harder for model
    avg_en_conf = round(sum(en_confs)/len(en_confs)*100, 1)
    avg_ar_conf = round(sum(ar_confs)/len(ar_confs)*100, 1)
    conf_gap    = round(avg_en_conf - avg_ar_conf, 1)

    summary = {
        "en_accuracy":      en_acc,
        "ar_accuracy":      ar_acc,
        "LPG":              lpg,
        "CLCS":             clcs,
        "ECE_english":      ece_en,
        "ECE_arabic":       ece_ar,
        "avg_conf_english": avg_en_conf,
        "avg_conf_arabic":  avg_ar_conf,
        "confidence_gap":   conf_gap,
        "n_pairs":          n,
    }

    print(f"\n  English accuracy:      {en_acc}%")
    print(f"  Arabic accuracy:       {ar_acc}%")
    print(f"  LPG:                   {lpg}%")
    print(f"  CLCS:                  {clcs}%")
    print(f"  ECE English:           {ece_en}%")
    print(f"  ECE Arabic:            {ece_ar}%")
    print(f"  Avg confidence EN:     {avg_en_conf}%")
    print(f"  Avg confidence AR:     {avg_ar_conf}%")
    print(f"  Confidence gap EN-AR:  {conf_gap}%")

    all_results[model_name] = {"summary": summary, "details": details}
    del model
    torch.cuda.empty_cache()

# Save
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\n{'='*65}")
print("FINAL TABLE — LPG + ECE + CLCS + Confidence Analysis:")
print(f"{'Model':<16} {'EN%':>5} {'AR%':>5} {'LPG':>5} {'CLCS':>6} {'ECE_EN':>7} {'ECE_AR':>7} {'Conf_Gap':>9}")
print("-"*65)
for name, res in all_results.items():
    s = res["summary"]
    print(f"{name:<16} {s['en_accuracy']:>4}% {s['ar_accuracy']:>4}% {s['LPG']:>4}% {s['CLCS']:>5}% {s['ECE_english']:>6}% {s['ECE_arabic']:>6}% {s['confidence_gap']:>8}%")

print(f"\nKey insight — Confidence Gap shows why Arabic is harder:")
print(f"Higher confidence_gap = model less certain on Arabic tokens")
print(f"This motivates SelecTKD: filter low-confidence Arabic tokens")
print(f"\nResults saved to {OUT}")
PYEOF
