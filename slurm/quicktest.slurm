#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_quicktest
#SBATCH -n 1
#SBATCH -t 1:00:00

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu

cd /lustre/home/mlozano/boneage-predictor
mkdir -p logs

EXP=${1:-99}

bash scripts/quicktest.sh $EXP
