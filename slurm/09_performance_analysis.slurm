#!/bin/bash
#SBATCH -p gpu
#SBATCH -J boneage_09_performance_analysis
#SBATCH -n 1
#SBATCH -t 1-00:00
#SBATCH -o /lustre/home/mlozano/boneage-predictor/logs/out_%j.log
#SBATCH -e /lustre/home/mlozano/boneage-predictor/logs/err_%j.log
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
mkdir -p logs
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/09_performance_analysis.py --experiment ${1:-30}
