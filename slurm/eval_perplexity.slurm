#!/bin/bash
#SBATCH --job-name=kd_ppl
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/ppl_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/ppl_%j.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

python - << 'PYEOF'
import os, json, torch, math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
DATA  = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/perplexity_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

CKPTS = {
    "base_7b":     None,
    "sft_7b":      "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

# Load test data — split by language
with open(DATA) as f:
    rows = [json.loads(l) for l in f]

ar_rows = [r for r in rows if r.get("language","en")=="ar"][:50]
en_rows = [r for r in rows if r.get("language","en")=="en"][:50]
print(f"AR test: {len(ar_rows)}, EN test: {len(en_rows)}")

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def compute_perplexity(tok, model, rows, max_len=256):
    """Compute perplexity on a set of instruction-response pairs."""
    total_loss = 0.0
    total_tokens = 0
    
    for row in tqdm(rows, desc="PPL", leave=False):
        text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
        inputs = tok(text, return_tensors="pt", truncation=True,
                    max_length=max_len, padding=False).to("cuda:0")
        
        labels = inputs["input_ids"].clone()
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
        
        n_tokens = inputs["input_ids"].shape[1]
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return round(ppl, 2), round(avg_loss, 4)

# Task performance: use GSM8K-style numeric extraction
TASK_PROMPTS = {
    "factual_ar": [
        {"q":"ما هي عاصمة فرنسا؟", "a":"باريس"},
        {"q":"كم عدد أيام السنة؟", "a":"365"},
        {"q":"ما عاصمة البحرين؟", "a":"المنامة"},
        {"q":"ما هو مجموع 15 + 27؟", "a":"42"},
        {"q":"في أي سنة تأسست جامعة البحرين؟", "a":"1986"},
    ],
    "factual_en": [
        {"q":"What is the capital of France?", "a":"paris"},
        {"q":"How many days in a year?", "a":"365"},
        {"q":"What is 15 plus 27?", "a":"42"},
        {"q":"What is the capital of Bahrain?", "a":"manama"},
        {"q":"What is machine learning?", "a":"data"},
    ],
    "reasoning_ar": [
        {"q":"إذا كان لدى أحمد 10 تفاحات وأعطى 3 لصديقه، كم تبقى معه؟", "a":"7"},
        {"q":"ما هو الرقم الأكبر: 15 أم 9؟", "a":"15"},
        {"q":"اذا كانت الساعة 3 وأضفنا ساعتين، كم ستكون الساعة؟", "a":"5"},
    ],
    "reasoning_en": [
        {"q":"If Ahmed has 10 apples and gives 3 to his friend, how many remain?", "a":"7"},
        {"q":"Which number is larger: 15 or 9?", "a":"15"},
        {"q":"If it is 3 o'clock and we add 2 hours, what time is it?", "a":"5"},
    ]
}

def gen(tok, model, q, max_new=50):
    msgs = [{"role":"user","content":q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

all_results = {}

for model_name, ckpt in CKPTS.items():
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("SKIP"); continue

    tok, model = load_model(ckpt)
    torch.cuda.empty_cache()

    # Perplexity
    print("Computing perplexity...")
    ppl_ar, loss_ar = compute_perplexity(tok, model, ar_rows)
    ppl_en, loss_en = compute_perplexity(tok, model, en_rows)
    print(f"  PPL Arabic:  {ppl_ar} (loss={loss_ar})")
    print(f"  PPL English: {ppl_en} (loss={loss_en})")

    # Task performance
    print("Computing task performance...")
    task_results = {}
    for task_name, prompts in TASK_PROMPTS.items():
        ok = 0
        for item in prompts:
            resp = gen(tok, model, item["q"])
            if item["a"].lower() in resp.lower():
                ok += 1
        score = round(ok/len(prompts)*100, 1)
        task_results[task_name] = score
        print(f"  {task_name}: {score}%")

    all_results[model_name] = {
        "perplexity_ar": ppl_ar,
        "perplexity_en": ppl_en,
        "ppl_ratio": round(ppl_ar/ppl_en, 2),
        "task_factual_ar": task_results["factual_ar"],
        "task_factual_en": task_results["factual_en"],
        "task_reasoning_ar": task_results["reasoning_ar"],
        "task_reasoning_en": task_results["reasoning_en"],
        "task_avg": round(sum(task_results.values())/len(task_results),1)
    }
    del model; torch.cuda.empty_cache()

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nPERPLEXITY AND TASK PERFORMANCE TABLE:")
print(f"{'Model':<16} {'PPL_AR':>8} {'PPL_EN':>8} {'Ratio':>7} {'Fact_AR':>9} {'Fact_EN':>9} {'Reas_AR':>9} {'Reas_EN':>9}")
print("-"*80)
for name, res in all_results.items():
    star=" ★" if "selectkd" in name else ""
    print(f"{name+star:<18} {res['perplexity_ar']:>8.1f} {res['perplexity_en']:>8.1f} {res['ppl_ratio']:>7.2f}× {res['task_factual_ar']:>8.1f}% {res['task_factual_en']:>8.1f}% {res['task_reasoning_ar']:>8.1f}% {res['task_reasoning_en']:>8.1f}%")

print(f"\nSaved to {OUT}")
PYEOF
