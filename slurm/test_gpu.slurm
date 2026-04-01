#!/bin/bash
#SBATCH --job-name=kd_test
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/test_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/test_%j.err
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

export HF_HOME=/data/datasets/user151/.cache/huggingface
export HF_TOKEN=$(grep HF_TOKEN /home/nfs/202200527/.bashrc | head -1 | cut -d'=' -f2)

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
echo "Environment OK"
