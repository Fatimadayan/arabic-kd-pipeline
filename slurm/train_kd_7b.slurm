#!/bin/bash
#SBATCH --job-name=kd_token_kd
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/token_kd_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/token_kd_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=HF_TOKEN_REMOVED

nvidia-smi
echo "Starting Token-KD at: $(date)"

python - << 'PYEOF'
import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

TOKEN    = os.environ["HF_TOKEN"]
DATA     = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
SEQ_CKPT = "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b"
SFT_CKPT = "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b"
OUT      = "/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b"
os.makedirs(OUT, exist_ok=True)

# KD hyperparameters
T     = 2.0   # temperature
ALPHA = 0.5   # CE weight
LR    = 1e-4
EPOCHS = 3
BATCH  = 2

print("Loading data...")
with open(DATA) as f:
    rows = [json.loads(l) for l in f]
print(f"Samples: {len(rows)}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def encode(row):
    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

print("Encoding dataset...")
encodings = [encode(r) for r in rows]

class KDDataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, i):
        return {k: v.squeeze(0) for k, v in self.encodings[i].items()}

dataset = KDDataset(encodings)
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

print("Loading teacher model in 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map={"": 0},
    token=TOKEN
)
teacher.eval()
print("Teacher loaded on cuda:0")

print("Loading student model...")
student_base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map={"": 0},
    token=TOKEN
)

# Load seq-kd weights if available, else sft weights
ckpt = SEQ_CKPT if os.path.exists(SEQ_CKPT+"/adapter_config.json") else SFT_CKPT
print(f"Loading LoRA from: {ckpt}")
student = PeftModel.from_pretrained(student_base, ckpt, is_trainable=True)
student.print_trainable_parameters()

optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad], lr=LR
)

print(f"Starting Token-KD: T={T}, alpha={ALPHA}, epochs={EPOCHS}")
for epoch in range(EPOCHS):
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        input_ids      = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")
        labels         = input_ids.clone()

        # Teacher logits (no grad)
        with torch.no_grad():
            t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            t_logits = t_out.logits

        # Student logits
        s_out    = student(input_ids=input_ids, attention_mask=attention_mask)
        s_logits = s_out.logits

        # CE loss
        shift_s = s_logits[..., :-1, :].contiguous()
        shift_l = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_s.view(-1, shift_s.size(-1)),
            shift_l.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # KL divergence loss (Hinton)
        shift_t = t_logits[..., :-1, :].contiguous()
        kl_loss = F.kl_div(
            F.log_softmax(shift_s / T, dim=-1),
            F.softmax(shift_t / T, dim=-1),
            reduction="batchmean"
        ) * (T ** 2)

        loss = ALPHA * ce_loss + (1 - ALPHA) * kl_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if (step+1) % 10 == 0:
            print(f"Epoch {epoch+1} step {step+1} loss: {total_loss/(step+1):.4f}")

    print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    # Save checkpoint each epoch
    ckpt_path = f"{OUT}/epoch_{epoch+1}"
    student.save_pretrained(ckpt_path)
    print(f"Saved epoch {epoch+1} to {ckpt_path}")

# Save final
student.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print(f"DONE! Token-KD model saved to {OUT}")
PYEOF
