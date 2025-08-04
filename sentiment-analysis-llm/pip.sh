#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=install_accelerate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:15:00

module load miniconda
eval "$(conda shell.bash hook)"
conda activate env1

echo "📦 accelerate yükleniyor..."
pip install accelerate --quiet

echo "✅ accelerate kurulumu tamamlandı."
