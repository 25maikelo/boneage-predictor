import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/lustre/home/mlozano/boneage-predictor")
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from config.paths import EQUALIZED_IMAGES_DIR, SEGMENTED_IMAGES_DIR
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.utils.losses import LOSS_MAP

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"

cfg27 = load_experiment_config(27)
cfg60 = load_experiment_config(60)

df = pd.read_csv(cfg27.DATASET_PATH)
df = df[(df["boneage"] >= cfg27.AGE_RANGE[0]) & (df["boneage"] <= cfg27.AGE_RANGE[1])]
df["gender"] = df["male"].astype(float)
train_df, val_df = train_test_split(df, test_size=cfg27.TEST_SPLIT, random_state=42)
print(f"val split n={len(val_df)}")

IMAGE_SIZE = cfg27.IMAGE_SIZE  # same for both configs, (112,112)

def load_img(folder, pid, size=IMAGE_SIZE):
    img = cv2.imread(os.path.join(folder, f"{pid}.png"))
    if img is None:
        return np.zeros((*size, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

ages = val_df["boneage"].to_numpy(dtype="float32")
genders = val_df["gender"].to_numpy(dtype="float32").reshape(-1, 1)

# ---------- Whole-hand (exp 60) ----------
wh_path = os.path.join(get_experiment_output_dir(60), "models", "whole_hand_model")
wh_model = load_model(wh_path, custom_objects=LOSS_MAP)
wh_imgs = np.stack([load_img(EQUALIZED_IMAGES_DIR, pid) for pid in val_df["id"]])
wh_preds = wh_model.predict((wh_imgs, genders), batch_size=32, verbose=0).flatten()
wh_mae = float(np.mean(np.abs(wh_preds - ages)))
print(f"Whole-hand (exp 60) val MAE = {wh_mae:.3f} months  (n={len(ages)})")

# ---------- Fusion (exp 27) ----------
fusion_path = os.path.join(get_experiment_output_dir(27), "models", "fusion_model")
fusion_model = load_model(fusion_path, custom_objects=LOSS_MAP)
seg_inputs = []
for seg in cfg27.SEGMENTS_ORDER:
    seg_inputs.append(np.stack([load_img(os.path.join(SEGMENTED_IMAGES_DIR, seg), pid) for pid in val_df["id"]]))
fusion_inputs = tuple(seg_inputs) + (genders,)
fusion_preds = fusion_model.predict(fusion_inputs, batch_size=32, verbose=0).flatten()
fusion_mae = float(np.mean(np.abs(fusion_preds - ages)))
print(f"Fusion F-DenseNet121 (exp 27) val MAE = {fusion_mae:.3f} months  (n={len(ages)})")

diff = wh_mae - fusion_mae
print(f"\nDelta (whole-hand - fusion) = {diff:+.3f} months")

import json
out = {
    "n_val": int(len(ages)),
    "whole_hand_mae": wh_mae,
    "fusion_densenet121_mae": fusion_mae,
    "delta_whole_hand_minus_fusion": diff,
}
with open(os.path.join(PROJECT_ROOT, "review2", "results_json", "whole_hand_vs_fusion.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to review2/results_json/whole_hand_vs_fusion.json")
