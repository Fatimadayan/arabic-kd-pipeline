#!/bin/bash
#SBATCH --job-name=kd_sft_7b
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/sft_7b_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/sft_7b_%j.err
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
echo "Starting SFT training at: $(date)"

python - << 'PYEOF'
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

TOKEN = os.environ["HF_TOKEN"]
DATA  = "/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl"
OUT   = "/data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b"
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

print("Loading model in 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    token=TOKEN
)

lora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
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

print("Starting training...")
trainer.train()
trainer.save_model(OUT)
print(f"DONE! Model saved to {OUT}")
PYEOF
