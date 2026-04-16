#!/bin/bash
#SBATCH -p gpu
#SBATCH -J boneage_01_hand_detector
#SBATCH -n 1
#SBATCH -t 1-00:00
#SBATCH -o /lustre/home/mlozano/boneage-predictor/models/hand-detector/slurm_out_%j.log
#SBATCH -e /lustre/home/mlozano/boneage-predictor/models/hand-detector/slurm_err_%j.log
#SBATCH --gres=gpu:1

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
mkdir -p logs
cd /lustre/home/mlozano/boneage-predictor
echo "Directorio de trabajo: $(pwd)"
python src/preprocessing/01_train_hand_detector.py

# Mover logs SLURM a la carpeta del run recién creado
LATEST=$(ls -dt models/hand-detector/hand-detector_[0-9][0-9] 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    mv models/hand-detector/slurm_out_${SLURM_JOB_ID}.log "$LATEST/" 2>/dev/null
    mv models/hand-detector/slurm_err_${SLURM_JOB_ID}.log "$LATEST/" 2>/dev/null
    echo "Logs movidos a: $LATEST"
fi
