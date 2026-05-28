#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_10_age_analysis
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/10_age_range_analysis.py --experiment ${1:-34} --bin-size ${2:-12} --dataset ${3:-both}
