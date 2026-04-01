#!/bin/bash
#SBATCH --job-name=kd_seq_kd
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/seq_kd_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/seq_kd_%j.err
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
echo "Starting Sequence KD at: $(date)"

python - << 'PYEOF'
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import Dataset

TOKEN = os.environ["HF_TOKEN"]
DATA  = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
SFT_CKPT = "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b"
OUT   = "/data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b"
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
with open(DATA) as f:
    rows = [json.loads(l) for l in f]
dataset = Dataset.from_list([
    {"text": f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['teacher_response']}"}
    for r in rows
])
print(f"Samples: {len(dataset)}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map={"":0},
    token=TOKEN
)

print("Loading SFT LoRA weights...")
model = PeftModel.from_pretrained(model, SFT_CKPT, is_trainable=True)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    bf16=True,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    processing_class=tokenizer,
)

print("Starting Sequence KD training...")
trainer.train()
trainer.save_model(OUT)
print(f"DONE! Seq-KD model saved to {OUT}")
PYEOF
