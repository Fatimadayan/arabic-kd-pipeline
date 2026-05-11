#!/bin/bash
#SBATCH --job-name=kd_benchmark
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/benchmark_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/benchmark_%j.err
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

echo "Starting benchmark evaluation at: $(date)"

python - << 'PYEOF'
import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

TOKEN   = os.environ["HF_TOKEN"]
BASE    = "Qwen/Qwen2.5-7B-Instruct"
CKPTS   = {
    "base_7b":      None,
    "sft_7b":       "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":    "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b":  "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
}
OUT = "/data/datasets/user151/qwen-arabic-kd/results/benchmark_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

# ── Load benchmarks ───────────────────────────────────────────
print("Loading benchmarks...")

# English MMLU (100 samples, 5 subjects)
mmlu_en = load_dataset("cais/mmlu", "all", split="test[:100]")

# Arabic MMLU
try:
    arabic_mmlu = load_dataset("MBZUAI/ArabicMMLU", split="test[:100]", trust_remote_code=True)
    has_arabic_mmlu = True
except:
    print("ArabicMMLU not available — using custom Arabic QA")
    has_arabic_mmlu = False

# Custom bilingual test (50 EN + 50 AR)
custom_bilingual = [
    # English
    {"q": "What is the capital of France?", "a": "Paris", "lang": "en"},
    {"q": "What is 15 multiplied by 7?", "a": "105", "lang": "en"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100", "lang": "en"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare", "lang": "en"},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "lang": "en"},
    {"q": "How many days are in a leap year?", "a": "366", "lang": "en"},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter", "lang": "en"},
    {"q": "What is the square root of 144?", "a": "12", "lang": "en"},
    {"q": "In which year did World War II end?", "a": "1945", "lang": "en"},
    {"q": "What is the speed of light approximately?", "a": "300000", "lang": "en"},
    # Arabic
    {"q": "ما هي عاصمة فرنسا؟", "a": "باريس", "lang": "ar"},
    {"q": "ما هو ناتج ضرب 15 في 7؟", "a": "105", "lang": "ar"},
    {"q": "ما هي درجة غليان الماء بالمئوية؟", "a": "100", "lang": "ar"},
    {"q": "من كتب روميو وجولييت؟", "a": "شكسبير", "lang": "ar"},
    {"q": "ما هو الرمز الكيميائي للذهب؟", "a": "Au", "lang": "ar"},
    {"q": "كم عدد أيام السنة الكبيسة؟", "a": "366", "lang": "ar"},
    {"q": "ما هو أكبر كوكب في المجموعة الشمسية؟", "a": "المشتري", "lang": "ar"},
    {"q": "ما هو الجذر التربيعي للعدد 144؟", "a": "12", "lang": "ar"},
    {"q": "في أي عام انتهت الحرب العالمية الثانية؟", "a": "1945", "lang": "ar"},
    {"q": "ما هي سرعة الضوء تقريباً؟", "a": "300000", "lang": "ar"},
]

def load_model(ckpt_path):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    if ckpt_path:
        model = PeftModel.from_pretrained(base, ckpt_path)
    else:
        model = base
    model.eval()
    return tokenizer, model

def answer(tokenizer, model, question, max_new=128):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score_answer(response, expected):
    return expected.lower() in response.lower()

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print('='*50)

    tokenizer, model = load_model(ckpt)
    results = {"model": model_name, "bilingual": [], "mmlu_en": [], "summary": {}}

    # ── Bilingual test ────────────────────────────────────────
    print("Running bilingual test (20 prompts)...")
    correct_en, correct_ar, total_en, total_ar = 0, 0, 0, 0

    for item in tqdm(custom_bilingual, desc="Bilingual"):
        resp = answer(tokenizer, model, item["q"])
        correct = score_answer(resp, item["a"])
        lang_correct = item["lang"] in resp.lower() or any(
            '\u0600' <= c <= '\u06FF' for c in resp
        ) == (item["lang"] == "ar")

        results["bilingual"].append({
            "question": item["q"],
            "expected": item["a"],
            "response": resp[:200],
            "correct": correct,
            "language": item["lang"],
            "correct_language": lang_correct
        })

        if item["lang"] == "en":
            total_en += 1
            if correct: correct_en += 1
        else:
            total_ar += 1
            if correct: correct_ar += 1

    en_acc = correct_en / total_en * 100
    ar_acc = correct_ar / total_ar * 100
    lpg = abs(en_acc - ar_acc)

    # ── English MMLU sample ───────────────────────────────────
    print("Running MMLU English (100 samples)...")
    mmlu_correct = 0
    choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for item in tqdm(mmlu_en, desc="MMLU-EN"):
        q = item["question"]
        choices = item["choices"]
        q_full = f"{q}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer with A, B, C, or D only."
        resp = answer(tokenizer, model, q_full, max_new=5)
        expected = choices_map[item["answer"]]
        if expected in resp.upper()[:10]:
            mmlu_correct += 1

    mmlu_acc = mmlu_correct / len(mmlu_en) * 100

    results["summary"] = {
        "english_accuracy": round(en_acc, 2),
        "arabic_accuracy": round(ar_acc, 2),
        "LPG": round(lpg, 2),
        "mmlu_english": round(mmlu_acc, 2),
        "total_bilingual": len(custom_bilingual),
    }

    print(f"\nResults for {model_name}:")
    print(f"  English accuracy: {en_acc:.1f}%")
    print(f"  Arabic accuracy:  {ar_acc:.1f}%")
    print(f"  LPG:              {lpg:.1f}%")
    print(f"  MMLU English:     {mmlu_acc:.1f}%")

    all_results[model_name] = results

    del model
    torch.cuda.empty_cache()

# ── Save results ──────────────────────────────────────────────
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nDONE! Results saved to {OUT}")
print("\n" + "="*60)
print("FINAL COMPARISON TABLE:")
print(f"{'Model':<20} {'EN Acc':>8} {'AR Acc':>8} {'LPG':>8} {'MMLU-EN':>10}")
print("-"*60)
for name, res in all_results.items():
    s = res["summary"]
    print(f"{name:<20} {s['english_accuracy']:>7.1f}% {s['arabic_accuracy']:>7.1f}% {s['LPG']:>7.1f}% {s['mmlu_english']:>9.1f}%")
PYEOF
