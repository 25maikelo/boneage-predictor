#!/bin/bash
#SBATCH -p gpu
#SBATCH -J test_gpu
#SBATCH -n 1
#SBATCH -t 0:05:00
#SBATCH -o /lustre/home/mlozano/boneage-predictor/logs/test_gpu_%j.log
#SBATCH -e /lustre/home/mlozano/boneage-predictor/logs/test_gpu_err_%j.log
#SBATCH --gres=gpu:1

nvidia-smi
module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
python /lustre/home/mlozano/boneage-predictor/scripts/test_gpu.py
