#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_analysis
#SBATCH -n 1
#SBATCH -t 1-16:00
#SBATCH -o logs/out_%j.log
#SBATCH -e logs/err_%j.log
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate master_env

mkdir -p logs

cd "$(dirname "$0")/.."
echo "Directorio de trabajo: $(pwd)"

EXPERIMENT=${1:-26}
python src/09_performance_analysis.py --experiment $EXPERIMENT
