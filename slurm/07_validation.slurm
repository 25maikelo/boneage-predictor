#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_07_validation
#SBATCH -n 1
#SBATCH -t 1-00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
mkdir -p logs
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/07_validation.py --experiment ${1:-30}
