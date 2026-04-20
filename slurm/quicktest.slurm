#!/bin/bash
#SBATCH -p q1
#SBATCH -J boneage_quicktest
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -o /lustre/home/mlozano/boneage-predictor/experiments/99/slurm_out_%j.log
#SBATCH -e /lustre/home/mlozano/boneage-predictor/experiments/99/slurm_err_%j.log

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu

cd /lustre/home/mlozano/boneage-predictor
mkdir -p logs

EXP=${1:-99}

bash scripts/quicktest.sh $EXP
