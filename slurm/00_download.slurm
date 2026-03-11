#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_download
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH -o logs/out_%j.log
#SBATCH -e logs/err_%j.log

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate master_env

mkdir -p logs

cd "$(dirname "$0")/.."   # Sube al root del proyecto
echo "Directorio de trabajo: $(pwd)"

python src/preprocessing/00_download_dataset.py
