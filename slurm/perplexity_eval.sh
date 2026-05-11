#!/bin/bash
#SBATCH --job-name=perplexity_eval
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/perplexity_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/perplexity_%j.err
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

python3 << 'PYEOF'
import os, json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

TOKEN = os.environ["HF_TOKEN"]
BASE = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = "/data/datasets/user151/qwen-arabic-kd/results"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "base_7b": (BASE, None),
    "token_kd_7b": (BASE, "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b"),
    "selectkd_7b": (BASE, "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b"),
}

results = {}

print("="*80)
print("PERPLEXITY EVALUATION ON WIKITEXT-2")
print("="*80)

# Load WikiText-2
print("\nLoading WikiText-2...")
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_text = "\n\n".join(test_dataset["text"])
print(f"✓ Loaded {len(test_text)} characters")

for model_name, (base_model, adapter_path) in MODELS.items():
    print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
    
    if adapter_path and not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"⚠️  Adapter not found, skipping"); continue
    
    # Load model
    print("Loading...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto", token=TOKEN)
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    
    # Tokenize
    encodings = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=2048)
    
    # Get max length
    try:
        max_length = model.config.max_position_embeddings
    except:
        max_length = 2048
    
    # Compute perplexity
    stride, seq_len = 2048, encodings.input_ids.size(1)
    nll_sum, n_tokens, prev_end_loc = 0.0, 0, 0
    
    print(f"Computing perplexity ({seq_len} tokens)...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        num_valid_tokens = (target_ids != -100).sum().item()
        num_loss_tokens = num_valid_tokens - target_ids.size(0)
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens
        prev_end_loc = end_loc
        if end_loc == seq_len: break
    
    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll).item()
    
    results[model_name] = {"perplexity": round(ppl, 2), "avg_nll": round(avg_nll.item(), 4)}
    print(f"✓ Perplexity: {ppl:.2f}")
    
    del model, tokenizer
    torch.cuda.empty_cache()

# Save
with open(os.path.join(OUT_DIR, "perplexity_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Print
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
for model_name, info in sorted(results.items()):
    print(f"{model_name:20s}: {info['perplexity']:6.2f}")
print(f"\n✅ Saved: {OUT_DIR}/perplexity_results.json")
PYEOF
