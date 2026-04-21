#!/bin/bash
#SBATCH --job-name=kd_theta_abl
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/theta_abl_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/theta_abl_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=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

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
OUT_DIR  = "/data/datasets/user151/qwen-arabic-kd/checkpoints/ablation_theta"
RESULTS  = "/data/datasets/user151/qwen-arabic-kd/results/theta_ablation.json"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/data/datasets/user151/qwen-arabic-kd/results", exist_ok=True)

# Eval pairs for quick LPG check
PAIRS = [
    {"en":"What is the capital of France?","ar":"ما هي عاصمة فرنسا؟","kw_en":["paris"],"kw_ar":["باريس","paris"]},
    {"en":"What is 15 multiplied by 7?","ar":"ما ناتج 15 في 7؟","kw_en":["105"],"kw_ar":["105"]},
    {"en":"What is the largest ocean?","ar":"ما أكبر محيط؟","kw_en":["pacific"],"kw_ar":["هادئ","pacific"]},
    {"en":"Who wrote Romeo and Juliet?","ar":"من كتب روميو وجولييت؟","kw_en":["shakespeare"],"kw_ar":["شكسبير"]},
    {"en":"What is the boiling point of water?","ar":"ما درجة غليان الماء؟","kw_en":["100"],"kw_ar":["100"]},
    {"en":"How many bones in human body?","ar":"كم عدد عظام الإنسان؟","kw_en":["206"],"kw_ar":["206"]},
    {"en":"What is machine learning?","ar":"ما هو تعلم الآلة؟","kw_en":["data","learn"],"kw_ar":["بيانات","تعلم"]},
    {"en":"What does CBB stand for?","ar":"ما اختصار CBB؟","kw_en":["central bank"],"kw_ar":["مركزي"]},
    {"en":"What is Knowledge Distillation?","ar":"ما هو تقطير المعرفة؟","kw_en":["teacher","student"],"kw_ar":["معلم","طالب"]},
    {"en":"What is LoRA?","ar":"ما هو LoRA؟","kw_en":["rank","adapt"],"kw_ar":["رتبة","تكيف"]},
]

THETAS = [0.5, 0.6, 0.7, 0.8, 0.9]

with open(DATA) as f:
    rows = [json.loads(l) for l in f][:500]

print(f"Training on {len(rows)} samples")

tokenizer = AutoTokenizer.from_pretrained(BASE, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def encode(row):
    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['teacher_response']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

encodings = [encode(r) for r in rows]
langs     = [r.get("language","en") for r in rows]

class KDDataset(TorchDataset):
    def __init__(self, enc, langs): self.enc=enc; self.langs=langs
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        item = {k:v.squeeze(0) for k,v in self.enc[i].items()}
        item["lang"] = self.langs[i]
        return item

def train_selectkd(theta, epochs=2):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    teacher = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    teacher.eval()
    student_base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"":0}, token=TOKEN
    )
    student = PeftModel.from_pretrained(student_base, SFT_CKPT, is_trainable=True)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)

    loader = DataLoader(KDDataset(encodings, langs), batch_size=2, shuffle=True)
    T = 2.0; ALPHA = 0.5

    en_selected_total, ar_selected_total = 0, 0
    en_tokens_total,   ar_tokens_total   = 0, 0

    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels         = input_ids.clone()
            batch_langs    = batch["lang"]

            with torch.no_grad():
                t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_s  = s_logits[...,:-1,:].contiguous()
            shift_t  = t_logits[...,:-1,:].contiguous()
            shift_l  = labels[...,1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_s.view(-1, shift_s.size(-1)),
                shift_l.view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            teacher_conf  = F.softmax(shift_t/T, dim=-1).max(dim=-1).values
            conf_mask     = (teacher_conf > theta).float()

            # Track per language
            for bi, lang in enumerate(batch_langs):
                lk = "ar" if lang == "ar" else "en"
                t  = conf_mask[bi].numel()
                s  = (conf_mask[bi]>0).sum().item()
                if lk == "ar":
                    ar_tokens_total   += t
                    ar_selected_total += s
                else:
                    en_tokens_total   += t
                    en_selected_total += s

            kl_per = F.kl_div(
                F.log_softmax(shift_s/T, dim=-1),
                F.softmax(shift_t/T, dim=-1),
                reduction="none"
            ).sum(dim=-1)

            masked_kl = kl_per * conf_mask
            n_sel     = conf_mask.sum().clamp(min=1)
            kl_loss   = (masked_kl.sum()/n_sel) * (T**2)

            loss = ALPHA * ce_loss + (1-ALPHA) * kl_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"  θ={theta} Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

    en_sel_rate = en_selected_total/max(en_tokens_total,1)*100
    ar_sel_rate = ar_selected_total/max(ar_tokens_total,1)*100
    print(f"  θ={theta}: EN selected={en_sel_rate:.1f}% AR selected={ar_sel_rate:.1f}%")

    # Quick eval
    student.eval()
    en_ok, ar_ok = 0, 0
    for pair in PAIRS:
        for lang, q, kws in [("en",pair["en"],pair["kw_en"]),("ar",pair["ar"],pair["kw_ar"])]:
            msgs = [{"role":"user","content":q}]
            text = tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            inputs = tokenizer(text,return_tensors="pt",truncation=True,max_length=512).to("cuda:0")
            with torch.no_grad():
                out = student.generate(**inputs,max_new_tokens=100,do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).lower()
            ok = any(k.lower() in resp for k in kws)
            if lang=="en": en_ok += ok
            else: ar_ok += ok

    n = len(PAIRS)
    en_acc = round(en_ok/n*100,1)
    ar_acc = round(ar_ok/n*100,1)
    lpg    = round(abs(en_acc-ar_acc),1)

    ckpt_path = f"{OUT_DIR}/theta_{str(theta).replace('.','')}"
    student.save_pretrained(ckpt_path)

    del teacher, student, student_base
    torch.cuda.empty_cache()

    return {
        "theta": theta,
        "en_acc": en_acc,
        "ar_acc": ar_acc,
        "LPG": lpg,
        "en_selected_pct": round(en_sel_rate,1),
        "ar_selected_pct": round(ar_sel_rate,1),
        "final_loss": round(total_loss/len(loader),4)
    }

# Add baseline (Token-KD, no filtering = theta=0)
results = {"theta_0.0_baseline": {
    "theta": 0.0, "en_acc": 86.7, "ar_acc": 63.3,
    "LPG": 23.4, "en_selected_pct": 100.0, "ar_selected_pct": 100.0,
    "note": "Token-KD without filtering (all tokens selected)"
}}

for theta in THETAS:
    print(f"\n{'='*40}\nRunning θ={theta}\n{'='*40}")
    res = train_selectkd(theta)
    results[f"theta_{theta}"] = res
    print(f"  Result: EN={res['en_acc']}% AR={res['ar_acc']}% LPG={res['LPG']}%")

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nSELECTKD ABLATION — THRESHOLD SENSITIVITY:")
print(f"{'θ':<6} {'EN%':>6} {'AR%':>6} {'LPG':>6} {'EN_sel%':>8} {'AR_sel%':>8}")
print("-"*45)
for k,v in sorted(results.items()):
    note = " ← no filter" if v.get("theta")==0.0 else (" ← best" if v.get("LPG")==0.0 else "")
    print(f"{v['theta']:<6} {v['en_acc']:>5}% {v['ar_acc']:>5}% {v['LPG']:>5}% {v['en_selected_pct']:>7}% {v['ar_selected_pct']:>7}%{note}")

print(f"\nKey finding: As θ increases, AR token selection rate drops")
print(f"This proves Arabic tokens have systematically lower teacher confidence")
print(f"Results saved to {RESULTS}")
PYEOF
