import os, sys, time, json, argparse
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from config.experiment import load_experiment_config, get_experiment_output_dir
from src.utils.losses import LOSS_MAP

N_WARMUP = 3
N_RUNS = 20

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=int, required=True)
args = parser.parse_args()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.reset_memory_stats(gpus[0].name.replace("/physical_device:", ""))
    print(f"GPU disponible: {[g.name for g in gpus]}")
else:
    print("Sin GPU, corriendo en CPU.")

cfg = load_experiment_config(args.experiment)
exp_dir = get_experiment_output_dir(args.experiment)
model_path = os.path.join(exp_dir, "models", "fusion_model")

t0 = time.time()
model = tf.keras.models.load_model(model_path, custom_objects=LOSS_MAP, compile=False)
load_time = time.time() - t0

input_shapes = [inp.shape for inp in model.inputs]
batch = []
for shp in input_shapes:
    dims = [1 if d is None else d for d in shp]
    batch.append(np.random.rand(*dims).astype("float32"))

for _ in range(N_WARMUP):
    model(batch, training=False)

if gpus:
    tf.config.experimental.reset_memory_stats(gpus[0].name.replace("/physical_device:", ""))

times = []
for _ in range(N_RUNS):
    t0 = time.time()
    model(batch, training=False)
    times.append((time.time() - t0) * 1000.0)

peak_mem_mb = None
if gpus:
    try:
        info = tf.config.experimental.get_memory_info(gpus[0].name.replace("/physical_device:", ""))
        peak_mem_mb = round(info["peak"] / (1024 * 1024), 1)
    except Exception as e:
        print(f"No se pudo leer memoria GPU: {e}")

result = {
    "experiment": args.experiment,
    "backbone": cfg.BASE_MODEL_CHOICE,
    "n_params": int(model.count_params()),
    "load_time_s": round(load_time, 3),
    "device": "GPU" if gpus else "CPU",
    "inference_latency_ms_mean": round(float(np.mean(times)), 3),
    "inference_latency_ms_std": round(float(np.std(times)), 3),
    "peak_gpu_memory_mb": peak_mem_mb,
}
print(json.dumps(result, indent=2))

out_path = os.path.join(PROJECT_ROOT, "review2/results_json/latency_benchmark.jsonl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "a") as f:
    f.write(json.dumps(result) + "\n")
