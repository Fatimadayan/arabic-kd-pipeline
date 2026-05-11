#!/bin/bash
#SBATCH --job-name=kd_harness
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/harness_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/harness_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=""
export OUT=/data/datasets/user151/qwen-arabic-kd/results/harness
mkdir -p $OUT

echo "Installing lm-eval in kd_env..."
pip install lm-eval --quiet

echo "Testing lm-eval install..."
python -c "import lm_eval; print('lm_eval version:', lm_eval.__version__)"

# Run MMLU on base model only first (test)
echo "Running MMLU on base_7b via harness..."
lm_eval \
  --model hf \
  --model_args "pretrained=Qwen/Qwen2.5-7B-Instruct,load_in_4bit=True,dtype=bfloat16,token=$HF_TOKEN" \
  --tasks mmlu \
  --num_fewshot 0 \
  --limit 100 \
  --output_path $OUT/base_7b_mmlu \
  --batch_size 1

echo "MMLU harness done for base_7b"
echo "Results in $OUT/base_7b_mmlu"
ls -lh $OUT/
