#!/bin/bash
cat > /data/datasets/user151/qwen-arabic-kd/slurm/final_eval_FIXED.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=kd_final_big_fixed
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/big_fixed_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/big_fixed_%j.err
#SBATCH --partition=gpu
#SBATCH --time=08:00:00  # Increased for 2000+ samples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

python3 - << 'PYEOF'
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
OUT   = "/data/datasets/user151/qwen-arabic-kd/results/big_eval_fixed.json"

CKPTS = {
    "base_7b": None,
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
    "selectkd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b",
}

def load_model(ckpt):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, 
                                               device_map={"":0}, token=TOKEN)
    model = PeftModel.from_pretrained(base, ckpt) if ckpt else base
    model.eval()
    return tok, model

def gen(tok, model, prompt, max_new=64):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                            temperature=0.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ================= DATASETS =================
print("Loading AraDiCE family...")
qatar_culture = load_dataset("QCRI/AraDiCE-Culture", "Qatar", split="test")  # N=30

# Subsample larger datasets for 8hr runtime
truthful = load_dataset("QCRI/AraDiCE-TruthfulQA", "TruthfulQA-msa", split="test").select(range(200))
piqa    = load_dataset("QCRI/AraDiCE-PIQA", split="test").select(range(200))
hella   = load_dataset("QCRI/AraDICE-HellaSwag", "HellaSwag-msa", split="validation").select(range(200))

print(f"Culture: {len(qatar_culture)}, Truthful: {len(truthful)}, PIQA: {len(piqa)}, Hella: {len(hella)}")

# ================= EVAL FUNCTIONS =================
def eval_culture(data, tok, model):  # MCQ, Option A correct
    correct = 0
    for item in tqdm(data, desc="Culture"):
        q, A, B, C = item["Question"], item["Option A"], item["Option B"], item["Option C"]
        prompt = f"{q}\nأ) {A}\nب) {B}\nج) {C}\nأجب: أ/ب/ج فقط"
        resp = gen(tok, model, prompt)
        if "أ" in resp[:5] or resp.strip() == "A":
            correct += 1
    return round(correct/len(data)*100, 1)

def eval_truthful(data, tok, model):  # Semantic match
    correct = 0
    for item in tqdm(data, desc="Truthful"):
        q = item["question"]
        ref = item["correct_answers"][0]  # First correct answer
        prompt = f"الإجابة الصحيحة لـ: {q}"
        resp = gen(tok, model, prompt)
        if any(word in resp.lower() for word in ref.lower().split()):
            correct += 1
    return round(correct/len(data)*100, 1)

def eval_piqa(data, tok, model):  # Best solution selection
    correct = 0
    for item in tqdm(data, desc="PIQA"):
        goal, sol1, sol2 = item["goal"], item["sol1"], item["sol2"]
        prompt = f"لتحقيق '{goal}'، أي حل أفضل؟ 1) {sol1} 2) {sol2}"
        resp = gen(tok, model, prompt)
        # Simple heuristic: check for "1" or sol1 keywords
        if "1" in resp[:10] or any(w in resp.lower() for w in sol1.lower().split()[:3]):
            correct += 1
    return round(correct/len(data)*100, 1)

# ================= MAIN LOOP =================
results = {}
for name, ckpt in CKPTS.items():
    print(f"\n===== {name.upper()} =====")
    if ckpt and not os.path.exists(ckpt+"/adapter_config.json"):
        print("Skipping missing CKPT")
        continue
    
    tok, model = load_model(ckpt)
    
    culture_score = eval_culture(qatar_culture, tok, model)
    truthful_score = eval_truthful(truthful, tok, model)
    piqa_score = eval_piqa(piqa, tok, model)
    
    results[name] = {
        "AraDiCE_Culture_N30": culture_score,
        "TruthfulQA_N200": truthful_score,
        "PIQA_N200": piqa_score,
        "Total_N": 30+200+200
    }
    
    print(f"CULTURE: {culture_score}% | TRUTH: {truthful_score}% | PIQA: {piqa_score}%")
    
    del model, tok
    torch.cuda.empty_cache()

# ================= OUTPUT =================
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n THESIS TABLE (Copy-Paste Ready)")
print(f"{'Model':<12} {'Culture(N=30)':>14} {'TruthQA(N=200)':>14} {'PIQA(N=200)':>12}")
print("-"*60)
for k,v in results.items():
    print(f"{k:<12} {v['AraDiCE_Culture_N30']:>6.1f}%{'':>8} {v['TruthfulQA_N200']:>6.1f}%{'':>8} {v['PIQA_N200']:>6.1f}%")

print(f"\n Saved: {OUT}")
PYEOF
EOF