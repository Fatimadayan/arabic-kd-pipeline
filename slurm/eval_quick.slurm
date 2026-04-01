#!/bin/bash
#SBATCH --job-name=kd_eval
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/eval_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=HF_TOKEN_REMOVED

nvidia-smi
echo "Starting evaluation at: $(date)"

python - << 'PYEOF'
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

TOKEN = os.environ["HF_TOKEN"]
BASE  = "Qwen/Qwen2.5-7B-Instruct"
CKPTS = {
    "sft_7b":      "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b",
    "seq_kd_7b":   "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b",
    "token_kd_7b": "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b",
}
OUT = "/data/datasets/user151/qwen-arabic-kd/results/eval_results.json"
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

# Bilingual test prompts
PROMPTS = [
    {"id": "en1", "lang": "en", "text": "What is the capital of France?"},
    {"id": "en2", "lang": "en", "text": "Explain the water cycle in simple terms."},
    {"id": "en3", "lang": "en", "text": "What is 15 multiplied by 7?"},
    {"id": "ar1", "lang": "ar", "text": "ما هي عاصمة فرنسا؟"},
    {"id": "ar2", "lang": "ar", "text": "اشرح دورة الماء بعبارات بسيطة."},
    {"id": "ar3", "lang": "ar", "text": "ما هو ناتج ضرب 15 في 7؟"},
    {"id": "ar4", "lang": "ar", "text": "ما هي أهمية اللغة العربية في العالم؟"},
    {"id": "en4", "lang": "en", "text": "What is the importance of the Arabic language in the world?"},
]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
all_results = {}

for model_name, ckpt_path in CKPTS.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print('='*50)

    tokenizer = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.eval()

    results = []
    for p in PROMPTS:
        inputs = tokenizer(p["text"], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150,
                do_sample=False, temperature=1.0
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "id": p["id"],
            "lang": p["lang"],
            "prompt": p["text"],
            "response": response
        })
        print(f"\n[{p['id']}] {p['text'][:50]}")
        print(f"→ {response[:150]}")

    all_results[model_name] = results
    del model, base_model
    torch.cuda.empty_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n\nDONE! Results saved to {OUT}")
print("\n" + "="*50)
print("SUMMARY — Arabic response check:")
for model_name, results in all_results.items():
    ar = [r for r in results if r["lang"] == "ar"]
    print(f"\n{model_name}:")
    for r in ar:
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in r["response"])
        print(f"  {r['id']}: {'✅ Arabic' if has_arabic else '❌ English only'}")
PYEOF
