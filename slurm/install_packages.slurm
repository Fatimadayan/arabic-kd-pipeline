#!/bin/bash
#SBATCH --job-name=kd_install
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/install_%j.out
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env
cd /data/datasets/user151/qwen-arabic-kd
pip install pyyaml -q
pip install -r requirements.txt -q
echo "INSTALL DONE"
