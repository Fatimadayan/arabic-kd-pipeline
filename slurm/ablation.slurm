#!/bin/bash
#SBATCH --job-name=kd_ablation
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/ablation_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

echo "Starting ablation study at: $(date)"

python - << 'PYEOF'
import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

TOKEN   = os.environ["HF_TOKEN"]
BASE    = "Qwen/Qwen2.5-7B-Instruct"
SFT_CKPT = "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b"
DATA    = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
OUT_DIR = "/data/datasets/user151/qwen-arabic-kd/checkpoints/ablation"
RESULTS = "/data/datasets/user151/qwen-arabic-kd/results/ablation_results.json"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

# Ablation grid
CONFIGS = [
    {"T": 1, "alpha": 0.3},
    {"T": 1, "alpha": 0.7},
    {"T": 2, "alpha": 0.3},
    {"T": 2, "alpha": 0.7},
    {"T": 4, "alpha": 0.3},
    {"T": 4, "alpha": 0.5},
    {"T": 4, "alpha": 0.7},
]
# Note: T=2, alpha=0.5 already done (main experiment)

print("Loading data...")
with open(DATA) as f:
    rows = [json.loads(l) for l in f]
print(f"Samples: {len(rows)}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def encode(row):
    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

print("Encoding dataset...")
encodings = [encode(r) for r in rows[:500]]  # Use 500 samples for ablation speed

class KDDataset(TorchDataset):
    def __init__(self, enc): self.enc = enc
    def __len__(self): return len(self.enc)
    def __getitem__(self, i): return {k: v.squeeze(0) for k, v in self.enc[i].items()}

dataset = KDDataset(encodings)
loader  = DataLoader(dataset, batch_size=2, shuffle=True)

all_results = {}

# Add the already-done run
all_results["T2_a0.5"] = {
    "T": 2, "alpha": 0.5,
    "final_loss": 13.75,
    "epochs": 3,
    "note": "Main experiment — already complete"
}

for cfg in CONFIGS:
    T     = cfg["T"]
    alpha = cfg["alpha"]
    tag   = f"T{T}_a{alpha}"
    print(f"\n{'='*40}")
    print(f"Running ablation: T={T}, alpha={alpha}")
    print('='*40)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    # Load teacher
    teacher = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    teacher.eval()

    # Load student from SFT checkpoint
    student_base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    student = PeftModel.from_pretrained(student_base, SFT_CKPT, is_trainable=True)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)

    epoch_losses = []
    for epoch in range(2):  # 2 epochs for ablation speed
        total_loss = 0
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = input_ids.clone()

            with torch.no_grad():
                t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            shift_s = s_logits[..., :-1, :].contiguous()
            shift_l = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_s.view(-1, shift_s.size(-1)),
                shift_l.view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            shift_t = t_logits[..., :-1, :].contiguous()
            kl_loss = F.kl_div(
                F.log_softmax(shift_s / T, dim=-1),
                F.softmax(shift_t / T, dim=-1),
                reduction="batchmean"
            ) * (T ** 2)

            loss = alpha * ce_loss + (1 - alpha) * kl_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        epoch_losses.append(round(avg, 4))
        print(f"  Epoch {epoch+1} loss: {avg:.4f}")

    # Save checkpoint
    ckpt_path = f"{OUT_DIR}/{tag}"
    student.save_pretrained(ckpt_path)

    all_results[tag] = {
        "T": T,
        "alpha": alpha,
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1],
        "checkpoint": ckpt_path
    }

    del teacher, student, student_base
    torch.cuda.empty_cache()
    print(f"Saved: {tag}")

# Save all results
with open(RESULTS, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nABLATION COMPLETE!")
print(f"\n{'Config':<12} {'T':>4} {'Alpha':>6} {'Final Loss':>12}")
print("-"*40)
for tag, res in sorted(all_results.items()):
    print(f"{tag:<12} {res['T']:>4} {res['alpha']:>6} {res['final_loss']:>12.4f}")
print(f"\nResults saved to {RESULTS}")
PYEOF
