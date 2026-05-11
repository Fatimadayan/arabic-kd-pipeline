# SelecTKD: Confidence-Based Token Filtering for Bilingual Arabic-English Knowledge Distillation

**University of Bahrain — Senior Capstone Project 2026**
**Student:** 202200527 | **Supervisor:** Dr. Abdullah Ebrahim Subah
**Cluster:** Hayrat Benefit Lab — NVIDIA A100-PCIE-40GB × 2

---

## Overview

This repository implements **SelecTKD**, a confidence-based token filtering method for bilingual Arabic-English Knowledge Distillation. We compress Qwen2.5-32B-Instruct → Qwen2.5-7B-Instruct for GCC browser deployment while preserving Arabic-English bilingual balance.

### Key Result
| Metric | Base-7B | Token-KD | SelecTKD |
|--------|---------|---------|---------|
| LPG ↓ | 23.4% | 23.4% | **0.0%** ★ |
| MMLU-EN | 73.8% | 73.5% | **73.4%** (no change) |
| PPL Ratio (AR/EN) ↓ | 3.00× | 2.96× | **1.20×** ★ |
| CLCS ↑ | 60.0% | 60.0% | **80.0%** ★ |

---

## Repository Structure

```
/data/datasets/user151/qwen-arabic-kd/
├── data/
│   ├── teacher_32b_responses.jsonl     # 4,927 bilingual samples (32B teacher)
│   ├── banking_train_10k.jsonl         # 10,000 GCC banking samples
│   ├── final_combined.jsonl            # 14,927 total training samples
│   ├── habash_corpus.tsv               # Habash Bahrain Corpus (53,529 sentences)
│   └── habash_prompts.json             # 300 evaluation prompts (100/task)
│
├── checkpoints/
│   ├── sft_7b/                         # Stage 1: Supervised Fine-Tuning
│   ├── seq_kd_7b/                      # Stage 2: Sequence-Level KD
│   ├── token_kd_7b/                    # Stage 3: Token-Level KD
│   ├── selectkd_7b/                    # Stage 4: SelecTKD ★ MAIN CONTRIBUTION
│   └── selectkd_dialect_ft/            # Stage 5: Habash Fine-Tuned (FT-V2)
│
├── results/
│   ├── final_eval.json                 # Layer 3 behavioral metrics
│   ├── habash_eval_results.json        # Layer 2b Habash corpus results
│   ├── perplexity_results.json         # PPL AR/EN per model
│   ├── aradice_final.json              # AraDiCE-Qatar results
│   ├── confidence_analysis_T2.json     # Confidence mechanism evidence
│   ├── seeds_results.json              # Seed reproducibility
│   └── harness/                        # lm-eval-harness MMLU results
│
├── slurm/
│   ├── eval_habash_clean.slurm         # Habash 300-prompt evaluation
│   ├── eval_perplexity.slurm           # PPL AR/EN evaluation
│   ├── eval_finetuned_full.slurm       # Full eval on FT models
│   ├── check_responses.slurm           # Qualitative response inspection
│   └── finetune_selectkd_dialect.slurm # Habash fine-tuning (FT-V2)
│
└── logs/                               # SLURM job logs
```

---

## Pipeline

### Stage 1 — Supervised Fine-Tuning (SFT)
```python
# Loss: CE(student, teacher_responses)
# Result: LPG = 6.7%
Checkpoint: /data/datasets/user151/qwen-arabic-kd/checkpoints/sft_7b/
```

### Stage 2 — Sequence-Level KD
```python
# Loss: CE(student, teacher_sequences)
# Result: LPG = 6.6%
Checkpoint: /data/datasets/user151/qwen-arabic-kd/checkpoints/seq_kd_7b/
```

### Stage 3 — Token-Level KD
```python
# Loss: α·CE + (1-α)·T²·KL(student‖teacher), T=2, α=0.5
# Result: LPG = 23.4% — FAILS (same as base)
Checkpoint: /data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b/
```

### Stage 4 — SelecTKD ★ (Main Contribution)
```python
# conf(t) = max(softmax(teacher_logits(t) / T=2))
# mask(t) = 1 if conf(t) > θ=0.7  else 0
# Loss: α·CE + (1-α)·T²·Σ[KL×mask]/Σmask
# Result: LPG = 0.0% ★
Checkpoint: /data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b/
```

### Stage 5 — Dialect Fine-Tuning (FT-V2)
```python
# Base: selectkd_7b + new LoRA r=8 on Habash Corpus (500 sentences)
# Result: Habash Dialect 78%, Normalize 72%, Code-Switch 75%
Checkpoint: /data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_dialect_ft/
```

---

## Model Configuration

| Component | Model | Parameters | Config |
|-----------|-------|-----------|--------|
| Teacher | Qwen/Qwen2.5-32B-Instruct | 32B | 4-bit NF4, inference only |
| Student | Qwen/Qwen2.5-7B-Instruct + LoRA | 5M trainable / 7.6B total | r=16, α=32, q_proj+v_proj |

---

## Training Data

| Source | Size | Language |
|--------|------|----------|
| opus-100 + alpaca (32B teacher) | 4,927 | Arabic + English |
| GCC Banking (generated) | 10,000 | Arabic + English |
| **Total** | **14,927** | **Bilingual** |

---

## Evaluation

### Run All Evaluations

```bash
# Activate environment
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface

# Layer 2b — Habash Corpus (300 prompts)
sbatch /data/datasets/user151/qwen-arabic-kd/slurm/eval_habash_clean.slurm

# Perplexity AR/EN
sbatch /data/datasets/user151/qwen-arabic-kd/slurm/eval_perplexity.slurm

# Layer 1 — MMLU via lm-eval-harness
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,load_in_4bit=True \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size 4 \
  --output_path /data/datasets/user151/qwen-arabic-kd/results/harness/

# Check results
squeue -u user151
cat /data/datasets/user151/qwen-arabic-kd/results/habash_eval_results.json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for name,res in d.items():
    print(f'{name}: DID={res["dialect_id"]}% NORM={res["normalize"]}% CS={res["code_switch"]}% Overall={res["overall"]}%')
"
```

### Run SFT and Seq-KD Evaluations (Missing from Table)

```bash
# Submit full eval for SFT and Seq-KD checkpoints
cat > /data/datasets/user151/qwen-arabic-kd/slurm/eval_sft_seqkd.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=eval_sft_seq
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/eval_sft_seq_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/eval_sft_seq_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=$HF_TOKEN

python /data/datasets/user151/qwen-arabic-kd/scripts/eval_layer3.py \
  --checkpoints sft_7b,seq_kd_7b \
  --output /data/datasets/user151/qwen-arabic-kd/results/sft_seqkd_eval.json
EOF

sbatch /data/datasets/user151/qwen-arabic-kd/slurm/eval_sft_seqkd.slurm
```

---

## Complete Results Table

| Layer | Metric | Base-7B | SFT-KD | Seq-KD | Token-KD | SelecTKD ★ | FT-V1 | FT-V2 |
|-------|--------|---------|--------|--------|---------|-----------|-------|-------|
| L1 | MMLU-EN | 73.8±0.6 | — | — | 73.5±0.6 | 73.4±0.6 | 59.5 | TBD |
| L1 | ArabicMMLU | 63.0 | 61.0 | 60.0 | 63.0 | 58.0 | 58.0 | TBD |
| L1 | GSM8K | 18.0 | 52.0 | 32.0 | 18.0 | 26.0 | 20.0 | TBD |
| L2a | AraDiCE-Qatar | 53.3 | — | — | 56.7 | 50.0 | 60.0 | TBD |
| L2b | Habash Overall | 77.7 | — | — | 77.7 | 70.7 | 4.7 | TBD |
| L2b | Dialect ID | 94.0 | — | — | 95.0 | 80.0 | 7.0 | 78.0 |
| L2b | Normalise | 68.0 | — | — | 70.0 | 53.0 | 1.0 | 72.0 |
| L2b | Code-Switch | 71.0 | — | — | 68.0 | 79.0 | 6.0 | 75.0 |
| L3 | EN% | 86.7 | 86.7 | 83.3 | 86.7 | 86.7 | 86.7 | 75.0 |
| L3 | AR% | 63.3 | 80.0 | 76.7 | 63.3 | 86.7 | 80.0 | TBD |
| L3 | LPG ↓ | 23.4 | 6.7 | 6.6 | 23.4 | **0.0** | 6.7 | TBD |
| L3 | CLCS ↑ | 60.0 | 73.3 | 66.7 | 60.0 | **80.0** | 66.7 | TBD |
| L3 | ECE-AR ↓ | 19.78 | 3.60 | 16.16 | 19.53 | **5.24** | 9.93 | TBD |
| PPL | PPL ratio | 3.00× | 1.32× | — | 2.96× | **1.20×** | 3.79× | TBD |

---

## Infrastructure

```
Cluster:    hayrat.uob.edu.bh
GPUs:       NVIDIA A100-PCIE-40GB × 2 (gpu01, gpu02)
Scheduler:  SLURM
Conda env:  /data/datasets/user151/conda-envs/kd_env (Python 3.10)
HF cache:   /data/datasets/user151/.cache/huggingface/
Home quota: 5GB — keep clean! Use /data/datasets/user151/ for all work
```

### Quick Commands
```bash
# Check disk usage
df -kh ~
du -hd1 /data/datasets/user151/qwen-arabic-kd/

# Check GPU
nvidia-smi

# Check jobs
squeue -u user151

# Watch job output live
tail -f /data/datasets/user151/qwen-arabic-kd/logs/<job_name>_<jobid>.out

# Kill a job
scancel <JOBID>
```

---

## Key Equations

```
LPG = |ACC_EN − ACC_AR|

conf(t) = max(softmax(teacher_logits(t) / T=2))
mask(t) = 1 if conf(t) > 0.7  else  0

L_SelecTKD = α × CE(student, labels)
           + (1−α) × T² × Σ[KL(s‖t) × mask(t)] / Σ mask(t)

Confidence ratio at T=2: EN=33.88% / AR=7.41% = 4.57×
MMLU verification: |73.8 − 73.4| = 0.4% < SE 0.6% → NOT significant
```

---

## References

- Hinton et al. (2015) — Knowledge Distillation. arXiv:1503.02531
- Hu et al. (2021) — LoRA. arXiv:2106.09685
- Dettmers et al. (2023) — QLoRA / 4-bit NF4. NeurIPS 2023
- Al-Ajmi et al. (2022) — Habash Bahrain Corpus. LREC 2022
- Mousa et al. (2025) — AraDiCE. NAACL 2025
- Koto et al. (2024) — ArabicMMLU. ACL 2024
- Wu et al. (2025) — SelecTKD. arXiv preprint

---

**Supervisor:** Dr. Abdullah Ebrahim Subah | **University of Bahrain | 2026**
