#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=goldgpt_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load miniconda
eval "$(conda shell.bash hook)"
conda activate env1

# Huggingface cache (isteğe bağlı)
export TRANSFORMERS_CACHE=/tmp/m.ozu_temp/hf_cache

pip install -U transformers --quiet

# Artık pip install yok, çünkü paketler zaten kurulu
python Intel.py
