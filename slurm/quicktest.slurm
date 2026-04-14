#!/bin/bash
#SBATCH -p gpu
#SBATCH -J boneage_quicktest
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH -o /lustre/home/mlozano/boneage-predictor/logs/quicktest_out_%j.log
#SBATCH -e /lustre/home/mlozano/boneage-predictor/logs/quicktest_err_%j.log
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu

cd /lustre/home/mlozano/boneage-predictor
mkdir -p logs

EXP=${1:-99}

bash scripts/quicktest.sh $EXP
