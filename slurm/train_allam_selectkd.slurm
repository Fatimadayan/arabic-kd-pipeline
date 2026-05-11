#!/bin/bash
#SBATCH --job-name=allam_selectkd
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/allam_selectkd_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/allam_selectkd_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""

pip install tiktoken --quiet
echo "Starting ALLaM SelecTKD at: $(date)"

python - << 'PYEOF'
import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

TOKEN    = os.environ["HF_TOKEN"]
# ALLaM as student — Arabic-native architecture
ALLAM    = "humain-ai/ALLaM-7B-Instruct-preview"
# Qwen32B as teacher (same teacher as your main experiment)
TEACHER  = "Qwen/Qwen2.5-32B-Instruct"  # use 7B as teacher for ALLaM comparison
DATA     = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
OUT      = "/data/datasets/user151/qwen-arabic-kd/checkpoints/allam_selectkd"
STATS    = "/data/datasets/user151/qwen-arabic-kd/results/allam_stats.json"
os.makedirs(OUT, exist_ok=True)
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

T=2.0; ALPHA=0.5; THETA=0.7; LR=1e-4; EPOCHS=3; BATCH=2

print(f"ALLaM SelecTKD: T={T}, alpha={ALPHA}, theta={THETA}")
print("Student: ALLaM-7B (Arabic-native)")
print("Teacher: Qwen2.5-7B (for comparison)")

with open(DATA) as f:
    rows = [json.loads(l) for l in f]
print(f"Samples: {len(rows)}")

# Load ALLaM tokenizer
print("Loading ALLaM tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ALLAM, token=TOKEN, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def encode(row):
    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

print("Encoding dataset...")
encodings = [encode(r) for r in rows]
langs     = [r.get("language","en") for r in rows]

class KDDataset(TorchDataset):
    def __init__(self, enc, langs): self.enc=enc; self.langs=langs
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        item = {k:v.squeeze(0) for k,v in self.enc[i].items()}
        item["lang"] = self.langs[i]
        return item

dataset = KDDataset(encodings, langs)
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

# Load teacher (Qwen 7B for fair comparison)
print("Loading teacher model...")
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, quantization_config=bnb, device_map={"":0}, token=TOKEN
)
teacher.eval()

# Load ALLaM student + LoRA
print("Loading ALLaM student...")
allam_base = AutoModelForCausalLM.from_pretrained(
    ALLAM, quantization_config=bnb, device_map={"":0}, token=TOKEN
)
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
student = get_peft_model(allam_base, lora_config)
student.print_trainable_parameters()

optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad], lr=LR
)

stats = {"ar":{"tokens_total":0,"tokens_selected":0},
         "en":{"tokens_total":0,"tokens_selected":0}}

print("\nStarting ALLaM SelecTKD training...")
for epoch in range(EPOCHS):
    total_loss=0; masked=0; total=0

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        input_ids     = batch["input_ids"].to("cuda:0")
        attention_mask= batch["attention_mask"].to("cuda:0")
        labels        = input_ids.clone()
        batch_langs   = batch["lang"]

        with torch.no_grad():
            t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_s = s_logits[...,:-1,:].contiguous()
        shift_t = t_logits[...,:-1,:].contiguous()
        shift_l = labels[...,1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_s.view(-1, shift_s.size(-1)),
            shift_l.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # SelecTKD confidence mask
        teacher_conf  = F.softmax(shift_t/T, dim=-1).max(dim=-1).values
        conf_mask     = (teacher_conf > THETA).float()

        total  += conf_mask.numel()
        masked += (conf_mask==0).sum().item()

        for bi, lang in enumerate(batch_langs):
            lk = "ar" if lang=="ar" else "en"
            stats[lk]["tokens_total"]    += conf_mask[bi].numel()
            stats[lk]["tokens_selected"] += (conf_mask[bi]>0).sum().item()

        kl_per = F.kl_div(
            F.log_softmax(shift_s/T, dim=-1),
            F.softmax(shift_t/T, dim=-1),
            reduction="none"
        ).sum(dim=-1)

        masked_kl = kl_per * conf_mask
        n_sel     = conf_mask.sum().clamp(min=1)
        kl_loss   = (masked_kl.sum()/n_sel) * (T**2)

        loss = ALPHA*ce_loss + (1-ALPHA)*kl_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg = total_loss/len(loader)
    mask_pct = masked/max(total,1)*100
    print(f"\nEpoch {epoch+1} loss: {avg:.4f}, masked: {mask_pct:.1f}%")

    for lang in ["en","ar"]:
        s = stats[lang]
        if s["tokens_total"]>0:
            sel = s["tokens_selected"]/s["tokens_total"]*100
            print(f"  {lang.upper()} token selection: {sel:.1f}%")

student.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

with open(STATS,"w") as f:
    json.dump(stats, f, indent=2)

print(f"\nDONE! ALLaM SelecTKD saved to {OUT}")
print("\nKey comparison:")
for lang in ["en","ar"]:
    s=stats[lang]
    if s["tokens_total"]>0:
        sel=s["tokens_selected"]/s["tokens_total"]*100
        print(f"  {lang.upper()} selection rate: {sel:.1f}%")
PYEOF
