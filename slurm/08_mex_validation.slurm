#!/bin/bash
#SBATCH -p gpu
#SBATCH -J boneage_08_mex_validation
#SBATCH -n 1
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
mkdir -p logs
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/08_mex_validation.py --experiment ${1:-30}
