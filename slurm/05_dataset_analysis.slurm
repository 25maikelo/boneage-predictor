#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_05_dataset_analysis
#SBATCH -n 1
#SBATCH -t 1-00:00

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate master_env
mkdir -p logs
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/05_dataset_analysis.py --experiment ${1:-30}
