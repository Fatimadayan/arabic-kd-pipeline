#!/bin/bash
#SBATCH --job-name=kd_selectkd
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/selectkd_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/selectkd_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

echo "Starting SelecTKD at: $(date)"

python - << 'PYEOF'
import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

TOKEN    = os.environ["HF_TOKEN"]
BASE     = "Qwen/Qwen2.5-7B-Instruct"
SFT_CKPT = "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b"
DATA     = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
OUT      = "/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b"
os.makedirs(OUT, exist_ok=True)

T     = 2.0
ALPHA = 0.5
THETA = 0.7   # ← SelecTKD confidence threshold
LR    = 1e-4
EPOCHS = 3
BATCH  = 2

print(f"SelecTKD config: T={T}, alpha={ALPHA}, theta={THETA}")
print("Hypothesis: filtering low-confidence tokens improves Arabic more than English")

with open(DATA) as f:
    rows = [json.loads(l) for l in f]
print(f"Samples: {len(rows)}")

tokenizer = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def encode(row):
    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

encodings = [encode(r) for r in rows]
langs     = [r.get("language", "en") for r in rows]

class KDDataset(TorchDataset):
    def __init__(self, enc, langs):
        self.enc = enc
        self.langs = langs
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        item = {k: v.squeeze(0) for k, v in self.enc[i].items()}
        item["lang"] = self.langs[i]
        return item

dataset = KDDataset(encodings, langs)
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

teacher = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
)
teacher.eval()

student_base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
)
student = PeftModel.from_pretrained(student_base, SFT_CKPT, is_trainable=True)
student.print_trainable_parameters()

optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad], lr=LR
)

# Track SelecTKD statistics per language
stats = {"ar": {"tokens_total":0,"tokens_selected":0,"correct":0},
         "en": {"tokens_total":0,"tokens_selected":0,"correct":0}}

print(f"\nStarting SelecTKD training...")
for epoch in range(EPOCHS):
    total_loss, total_ce, total_kd = 0, 0, 0
    masked_tokens, total_tokens = 0, 0

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")
        labels = input_ids.clone()
        batch_langs = batch["lang"]

        with torch.no_grad():
            t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_s = s_logits[..., :-1, :].contiguous()
        shift_t = t_logits[..., :-1, :].contiguous()
        shift_l = labels[..., 1:].contiguous()

        # CE loss (unchanged)
        ce_loss = F.cross_entropy(
            shift_s.view(-1, shift_s.size(-1)),
            shift_l.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # SelecTKD: confidence mask
        # confidence = max softmax probability of teacher at each token
        teacher_conf = F.softmax(shift_t / T, dim=-1).max(dim=-1).values
        # Mask: only distill where teacher is confident
        confidence_mask = (teacher_conf > THETA).float()

        total_tokens   += confidence_mask.numel()
        masked_tokens  += (confidence_mask == 0).sum().item()

        # Track per language
        for bi, lang in enumerate(batch_langs):
            lang_key = "ar" if lang == "ar" else "en"
            stats[lang_key]["tokens_total"]    += confidence_mask[bi].numel()
            stats[lang_key]["tokens_selected"] += (confidence_mask[bi] > 0).sum().item()

        # KL loss with confidence mask applied
        kl_per_token = F.kl_div(
            F.log_softmax(shift_s / T, dim=-1),
            F.softmax(shift_t / T, dim=-1),
            reduction="none"
        ).sum(dim=-1)  # [B, seq_len]

        # Apply mask: zero out low-confidence positions
        masked_kl = kl_per_token * confidence_mask
        # Normalize by number of selected tokens (avoid empty batch)
        n_selected = confidence_mask.sum().clamp(min=1)
        kl_loss = (masked_kl.sum() / n_selected) * (T ** 2)

        loss = ALPHA * ce_loss + (1 - ALPHA) * kl_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_ce   += ce_loss.item()
        total_kd   += kl_loss.item()

        if (step+1) % 50 == 0:
            mask_rate = masked_tokens / max(total_tokens, 1) * 100
            print(f"  Step {step+1}: loss={total_loss/(step+1):.4f} "
                  f"ce={total_ce/(step+1):.4f} kd={total_kd/(step+1):.4f} "
                  f"masked={mask_rate:.1f}%")

    avg_loss = total_loss / len(loader)
    mask_pct = masked_tokens / max(total_tokens, 1) * 100
    print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")
    print(f"Tokens masked (low confidence): {mask_pct:.1f}%")

    # Language-specific filtering stats
    for lang in ["en", "ar"]:
        s = stats[lang]
        if s["tokens_total"] > 0:
            sel_rate = s["tokens_selected"] / s["tokens_total"] * 100
            print(f"  {lang.upper()} token selection rate: {sel_rate:.1f}%")

    ckpt_path = f"{OUT}/epoch_{epoch+1}"
    student.save_pretrained(ckpt_path)
    print(f"Saved epoch {epoch+1}")

student.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

# Save stats
stats_file = "/data/datasets/user151/qwen-arabic-kd/results/selectkd_stats.json"
with open(stats_file, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nDONE! SelecTKD model saved to {OUT}")
print(f"\nKey finding: Compare AR vs EN token selection rates")
print(f"If AR rate < EN rate → Arabic tokens are less confident → SelecTKD filters more Arabic noise")
print(f"This motivates: SelecTKD should improve Arabic accuracy more than English")
print(f"Stats saved to {stats_file}")
PYEOF
