#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_04_segment
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH -o logs/out_%j.log
#SBATCH -e logs/err_%j.log
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate master_env

mkdir -p logs

cd "$(dirname "$0")/.."
echo "Directorio de trabajo: $(pwd)"

python src/preprocessing/04_segment.py
