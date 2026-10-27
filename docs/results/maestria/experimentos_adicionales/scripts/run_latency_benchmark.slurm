#!/bin/bash
#SBATCH -p gpu
#SBATCH -J boneage_latency_bench
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre/home/mlozano/boneage-predictor/review2/results_json/latency_benchmark.log
#SBATCH --error=/lustre/home/mlozano/boneage-predictor/review2/results_json/latency_benchmark.log

module load anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate boneage_gpu
cd /lustre/home/mlozano/boneage-predictor

rm -f review2/results_json/latency_benchmark.jsonl

for exp in 23 26 27 28; do
  echo "=== Experimento $exp ==="
  python3 review2/scripts/latency_benchmark.py --experiment $exp
  echo
done
