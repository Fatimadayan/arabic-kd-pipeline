# Arabic/Bilingual Knowledge Distillation Pipeline
### University of Bahrain — Benefit Lab | Senior project 2026


---

## Research Question
> *"Can Knowledge Distillation from a large multilingual teacher preserve Arabic + English bilingual instruction-following in smaller student models, and how does compression affect cross-lingual symmetry?"*

---

## What This Does
This project compresses a large AI model (teacher) into smaller models (students) while keeping Arabic + English capabilities. Like a professor teaching students — the students learn to respond like the professor.
```
Teacher: Qwen2.5-7B-Instruct (4-bit quantized)
    │
    ├── Stage 1: SFT        → student learns from teacher answers
    ├── Stage 2: Sequence KD → student copies teacher response style  
    └── Stage 3: Token KD   → student matches teacher word-by-word (MAIN CONTRIBUTION)
                              L = α·CE + (1-α)·T²·KL(student/T ‖ teacher/T)
                              T=2.0, α=0.5
```

---

## Results

| Stage | Model | Final Loss | Token Accuracy | Arabic Preserved |
|---|---|---|---|---|
| Baseline SFT | sft_7b | 0.400 | 90.08% | ✅ 100% |
| Sequence KD | seq_kd_7b | 0.384 | 90.90% | ✅ 100% |
| **Token KD** | **token_kd_7b** | **12.5*** | **—** | **✅ 100%** |

*Token-KD loss is KL-divergence scale — not comparable to CE loss. Consistent decrease across epochs confirms learning.

### Key Finding
**Arabic bilingual capability is 100% preserved across all 3 KD training stages.**

All models respond correctly in Arabic when prompted in Arabic, and in English when prompted in English.

---

## Sample Outputs (Token-KD Model)

| Prompt | Response |
|---|---|
| ما هي عاصمة فرنسا؟ | عاصمة فرنسا هي باريس. باريس هي أكبر مدينة في فرنسا... ✅ |
| ما هو ناتج ضرب 15 في 7؟ | ناتج ضرب 15 في 7 هو 105 ✅ |
| What is the capital of France? | The capital of France is Paris... ✅ |
| Explain the water cycle | The water cycle is a natural process... ✅ |

---

## Training Data

| Dataset | Samples | Language |
|---|---|---|
| tatsu-lab/alpaca | 500 | English |
| Helsinki-NLP/opus-100 (ar-en) | 250 Arabic + 250 English | Bilingual |

---

## Checkpoints (Saved on Cluster)

| Model | Size | Location |
|---|---|---|
| sft_7b | 101MB | checkpoints/sft_7b/ |
| seq_kd_7b | 61MB | checkpoints/seq_kd_7b/ |
| token_kd_7b | 88MB | checkpoints/token_kd_7b/ |

---

## Infrastructure
- **GPU:** NVIDIA A100-PCIE-40GB (×2)
- **Cluster:** Benefit Lab, University of Bahrain
- **Scheduler:** SLURM (partition: gpu)
- **Framework:** HuggingFace Transformers + PEFT (LoRA) + TRL
- **Total training time:** ~35 minutes for all 3 stages

---

## Project Structure
```
qwen-arabic-kd/
├── configs/          # Model, data, eval settings
├── scripts/          # Training + evaluation scripts
├── slurm/            # SLURM job submission scripts
├── data/             # Teacher-generated training data
├── checkpoints/      # Trained model checkpoints
├── results/          # Evaluation results JSON
└── README.md
```

---

## Next Steps
- [ ] Upgrade teacher to Qwen2.5-32B (fits on A100 in 4-bit)
- [ ] Generate 5,000+ Arabic+English samples from 32B teacher
- [ ] Add TinyLlama-1.1B as Student-S
- [ ] Full 4-pillar evaluation (MMLU, ArabicMMLU, cross-lingual)
- [ ] Compute LPG, CLCS, IFS, RDI metrics
- [ ] Ablation: T∈{1,2,4}, α∈{0.3,0.5,0.7}

---

## References
- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
- Qwen Team (2024). *Qwen2.5 Technical Report*
