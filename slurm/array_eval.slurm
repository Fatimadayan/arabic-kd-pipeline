#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --array=0-499  # 500 jobs!
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/array_%A_%a.out

# Process 1 sample per job
SAMPLE_ID=$SLURM_ARRAY_TASK_ID
# Your eval code for sample $SAMPLE_ID
