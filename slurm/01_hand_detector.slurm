#!/bin/bash
#SBATCH -p gpu
#SBATCH -J boneage_01_hand_detector
#SBATCH -n 1
#SBATCH -t 1-00:00
#SBATCH -o /lustre/home/mlozano/boneage-predictor/logs/out_%j.log
#SBATCH -e /lustre/home/mlozano/boneage-predictor/logs/err_%j.log
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate master_env
mkdir -p logs
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/preprocessing/01_train_hand_detector.py
