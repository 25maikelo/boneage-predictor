import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/lustre/home/mlozano/boneage-predictor")
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from config.paths import VALIDATION_CSV, VALIDATION_IMAGES_DIR, MEX_CSV, MEX_IMAGES_DIR
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.utils.losses import LOSS_MAP

import importlib.util
spec = importlib.util.spec_from_file_location("val_mod", "/lustre/home/mlozano/boneage-predictor/src/07_validation.py")
val_mod = importlib.util.module_from_spec(spec)
sys.modules["val_mod"] = val_mod
spec.loader.exec_module(val_mod)

spec2 = importlib.util.spec_from_file_location("mex_mod", "/lustre/home/mlozano/boneage-predictor/src/08_mex_validation.py")
mex_mod = importlib.util.module_from_spec(spec2)
sys.modules["mex_mod"] = mex_mod
spec2.loader.exec_module(mex_mod)

cfg = load_experiment_config(60)
IMAGE_SIZE = cfg.IMAGE_SIZE

wh_path = os.path.join(get_experiment_output_dir(60), "models", "whole_hand_model")
model = load_model(wh_path, custom_objects=LOSS_MAP)


def normalize_image(img):
    return img.astype("float32") / 255.0


def run_rsna():
    df = pd.read_csv(VALIDATION_CSV)
    df.reset_index(drop=True, inplace=True)
    df["boneage"] = df["boneage"].apply(val_mod.parse_age_to_months)

    preds, trues, failed = [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        sid = str(row["id"])
        real_age = float(row["boneage"])
        img_path = os.path.join(VALIDATION_IMAGES_DIR, f"{sid}.png")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            failed.append((sid, "not_found")); continue
        try:
            zoomed = val_mod.frame_and_zoom(gray)
            eq = val_mod.clahe_equalize(zoomed)
            img = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            failed.append((sid, f"preprocess: {e}")); continue
        try:
            inp = tf.expand_dims(normalize_image(img), 0)
            gender_in = tf.constant([[float(row.get("male", 0))]], tf.float32)
            pred_age = float(model.predict((inp, gender_in), verbose=0).flatten()[0])
        except Exception as e:
            failed.append((sid, f"predict: {e}")); continue
        preds.append(pred_age); trues.append(real_age)
        if i % 200 == 0:
            print(f"[RSNA] {i}/{len(df)} ok={len(preds)} failed={len(failed)}", flush=True)

    mae = float(np.mean(np.abs(np.array(preds) - np.array(trues)))) if preds else None
    print(f"[RSNA] TOTAL processed={len(preds)} failed={len(failed)} MAE={mae:.3f}")
    return {"processed": len(preds), "failed": len(failed), "mae": mae,
            "failed_detail": failed}


def run_mex():
    df = pd.read_csv(MEX_CSV)
    df.reset_index(drop=True, inplace=True)
    df["bone_age"] = df["bone_age"].apply(mex_mod.parse_age_to_months)

    preds, trues, failed = [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        sid = str(row["ID"])
        bone_age_true = float(row["bone_age"])
        img_path = os.path.join(MEX_IMAGES_DIR, f"{sid}.png")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            failed.append((sid, "not_found")); continue
        try:
            zoomed = mex_mod.frame_and_zoom(gray)
            eq = mex_mod.clahe_equalize(zoomed)
            img = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            failed.append((sid, f"preprocess: {e}")); continue
        try:
            inp = tf.expand_dims(normalize_image(img), 0)
            gender_in = tf.constant([[1.0 if row.get("gender") == "M" else 0.0]], tf.float32)
            pred_age = float(model.predict((inp, gender_in), verbose=0).flatten()[0])
        except Exception as e:
            failed.append((sid, f"predict: {e}")); continue
        if not np.isnan(pred_age) and not np.isnan(bone_age_true):
            preds.append(pred_age); trues.append(bone_age_true)
        else:
            failed.append((sid, "nan"))

    mae = float(np.mean(np.abs(np.array(preds) - np.array(trues)))) if preds else None
    print(f"[MEX] TOTAL processed={len(preds)} failed={len(failed)} MAE={mae:.3f}")
    return {"processed": len(preds), "failed": len(failed), "mae": mae,
            "failed_detail": failed}


rsna_res = run_rsna()
mex_res = run_mex()

import json
out = {"rsna_external": rsna_res, "mex_external": mex_res}
with open("/lustre/home/mlozano/boneage-predictor/review2/results_json/whole_hand_external_eval.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved to review2/results_json/whole_hand_external_eval.json")
