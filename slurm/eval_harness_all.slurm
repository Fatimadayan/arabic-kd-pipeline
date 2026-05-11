#!/bin/bash
#SBATCH --job-name=kd_harness_all
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/harness_all_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/harness_all_%j.err
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""
export OUT=/data/datasets/user151/qwen-arabic-kd/results/harness
mkdir -p $OUT

pip install lm-eval --quiet

MODELS=(
  "base_7b|Qwen/Qwen2.5-7B-Instruct|"
  "token_kd|Qwen/Qwen2.5-7B-Instruct|/data/datasets/user151/qwen-arabic-kd/checkpoints/token_kd_7b"
  "selectkd|Qwen/Qwen2.5-7B-Instruct|/data/datasets/user151/qwen-arabic-kd/checkpoints/selectkd_7b"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name base peft <<< "$entry"
  echo "=== Running $name ==="
  if [ -z "$peft" ]; then
    ARGS="pretrained=$base,dtype=bfloat16,token=$HF_TOKEN"
  else
    ARGS="pretrained=$base,peft=$peft,dtype=bfloat16,token=$HF_TOKEN"
  fi
  lm_eval \
    --model hf \
    --model_args "$ARGS" \
    --tasks mmlu \
    --num_fewshot 0 \
    --limit 100 \
    --output_path $OUT/${name}_mmlu \
    --batch_size 1
  echo "Done: $name"
done

echo "All harness runs complete!"
ls -lh $OUT/
