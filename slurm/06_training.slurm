#!/bin/bash
#SBATCH -p gpu
#SBATCH -J boneage_06_training
#SBATCH -n 1
#SBATCH -t 4-00:00:00
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
mkdir -p logs
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/06_training.py --experiment ${1:-30}
