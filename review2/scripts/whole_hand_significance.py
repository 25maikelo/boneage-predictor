import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/lustre/home/mlozano/boneage-predictor")
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from config.paths import (EQUALIZED_IMAGES_DIR, SEGMENTED_IMAGES_DIR,
                           VALIDATION_CSV, VALIDATION_IMAGES_DIR, MEX_CSV, MEX_IMAGES_DIR)
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.utils.losses import LOSS_MAP

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"

import importlib.util
spec = importlib.util.spec_from_file_location("val_mod", f"{PROJECT_ROOT}/src/07_validation.py")
val_mod = importlib.util.module_from_spec(spec)
sys.modules["val_mod"] = val_mod
spec.loader.exec_module(val_mod)

spec2 = importlib.util.spec_from_file_location("mex_mod", f"{PROJECT_ROOT}/src/08_mex_validation.py")
mex_mod = importlib.util.module_from_spec(spec2)
sys.modules["mex_mod"] = mex_mod
spec2.loader.exec_module(mex_mod)

cfg27 = load_experiment_config(27)
cfg60 = load_experiment_config(60)
IMAGE_SIZE = cfg60.IMAGE_SIZE


def normalize_image(img):
    return img.astype("float32") / 255.0


def load_img(folder, pid, size=IMAGE_SIZE):
    img = cv2.imread(os.path.join(folder, f"{pid}.png"))
    if img is None:
        return np.zeros((*size, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


wh_model = load_model(os.path.join(get_experiment_output_dir(60), "models", "whole_hand_model"),
                       custom_objects=LOSS_MAP)
fusion_model = load_model(os.path.join(get_experiment_output_dir(27), "models", "fusion_model"),
                           custom_objects=LOSS_MAP)

abs_err = {"internal": {}, "rsna_external": {}, "mex_external": {}}

# ---------------- Internal validation split (same for both, n=2357) ----------------
df = pd.read_csv(cfg27.DATASET_PATH)
df = df[(df["boneage"] >= cfg27.AGE_RANGE[0]) & (df["boneage"] <= cfg27.AGE_RANGE[1])]
df["gender"] = df["male"].astype(float)
_, val_df = train_test_split(df, test_size=cfg27.TEST_SPLIT, random_state=42)
ids = [str(i) for i in val_df["id"]]
ages = val_df["boneage"].to_numpy(dtype="float32")
genders = val_df["gender"].to_numpy(dtype="float32").reshape(-1, 1)

wh_imgs = np.stack([load_img(EQUALIZED_IMAGES_DIR, pid) for pid in val_df["id"]])
wh_preds = wh_model.predict((wh_imgs, genders), batch_size=32, verbose=0).flatten()

seg_inputs = tuple(
    np.stack([load_img(os.path.join(SEGMENTED_IMAGES_DIR, seg), pid) for pid in val_df["id"]])
    for seg in cfg27.SEGMENTS_ORDER
) + (genders,)
fusion_preds = fusion_model.predict(seg_inputs, batch_size=32, verbose=0).flatten()

abs_err["internal"]["whole_hand"] = {"ids": ids, "abs_err": np.abs(wh_preds - ages).tolist()}
abs_err["internal"]["fusion"] = {"ids": ids, "abs_err": np.abs(fusion_preds - ages).tolist()}
print(f"[internal] n={len(ids)} wh_MAE={np.mean(np.abs(wh_preds-ages)):.3f} "
      f"fusion_MAE={np.mean(np.abs(fusion_preds-ages)):.3f}", flush=True)

# ---------------- RSNA external (whole-hand: rerun with id tracking) ----------------
rdf = pd.read_csv(VALIDATION_CSV)
rdf.reset_index(drop=True, inplace=True)
rdf["boneage"] = rdf["boneage"].apply(val_mod.parse_age_to_months)

wh_ids, wh_ae = [], []
for i, (_, row) in enumerate(rdf.iterrows()):
    sid = str(row["id"])
    real_age = float(row["boneage"])
    gray = cv2.imread(os.path.join(VALIDATION_IMAGES_DIR, f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
    zoomed = val_mod.frame_and_zoom(gray)
    eq = val_mod.clahe_equalize(zoomed)
    img = cv2.resize(cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB), IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    inp = tf.expand_dims(normalize_image(img), 0)
    gender_in = tf.constant([[float(row.get("male", 0))]], tf.float32)
    pred_age = float(wh_model.predict((inp, gender_in), verbose=0).flatten()[0])
    wh_ids.append(sid); wh_ae.append(abs(pred_age - real_age))
    if i % 300 == 0:
        print(f"[RSNA whole-hand] {i}/{len(rdf)}", flush=True)

abs_err["rsna_external"]["whole_hand"] = {"ids": wh_ids, "abs_err": wh_ae}

with open(f"{PROJECT_ROOT}/experiments/27/validation/plot_data.json") as f:
    d = json.load(f)
f_ids = [str(i) for i in d["scatter"]["ids"]]
f_trues = np.array(d["scatter"]["trues"]); f_preds = np.array(d["scatter"]["preds"])
abs_err["rsna_external"]["fusion"] = {"ids": f_ids, "abs_err": np.abs(f_preds - f_trues).tolist()}
print(f"[RSNA external] wh n={len(wh_ids)} MAE={np.mean(wh_ae):.3f}  "
      f"fusion n={len(f_ids)} MAE={np.mean(np.abs(f_preds-f_trues)):.3f}", flush=True)

# ---------------- MEX external (whole-hand: rerun with id tracking) ----------------
mdf = pd.read_csv(MEX_CSV)
mdf.reset_index(drop=True, inplace=True)
mdf["bone_age"] = mdf["bone_age"].apply(mex_mod.parse_age_to_months)

wh_ids_m, wh_ae_m = [], []
for i, (_, row) in enumerate(mdf.iterrows()):
    sid = str(row["ID"])
    bone_age_true = float(row["bone_age"])
    gray = cv2.imread(os.path.join(MEX_IMAGES_DIR, f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
    zoomed = mex_mod.frame_and_zoom(gray)
    eq = mex_mod.clahe_equalize(zoomed)
    img = cv2.resize(cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB), IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    inp = tf.expand_dims(normalize_image(img), 0)
    gender_in = tf.constant([[1.0 if row.get("gender") == "M" else 0.0]], tf.float32)
    pred_age = float(wh_model.predict((inp, gender_in), verbose=0).flatten()[0])
    if not np.isnan(pred_age) and not np.isnan(bone_age_true):
        wh_ids_m.append(sid); wh_ae_m.append(abs(pred_age - bone_age_true))

abs_err["mex_external"]["whole_hand"] = {"ids": wh_ids_m, "abs_err": wh_ae_m}

with open(f"{PROJECT_ROOT}/experiments/27/mex-validation/plot_data.json") as f:
    d = json.load(f)
fm_ids = [str(i) for i in d["scatter"]["ids"]]
fm_trues = np.array(d["scatter"]["trues"]); fm_preds = np.array(d["scatter"]["preds"])
abs_err["mex_external"]["fusion"] = {"ids": fm_ids, "abs_err": np.abs(fm_preds - fm_trues).tolist()}
print(f"[MEX external] wh n={len(wh_ids_m)} MAE={np.mean(wh_ae_m):.3f}  "
      f"fusion n={len(fm_ids)} MAE={np.mean(np.abs(fm_preds-fm_trues)):.3f}", flush=True)


# ---------------- Paired significance tests (same methodology as extended_stats.py) ----------------
def rank_biserial(a, b):
    diffs = np.array(a) - np.array(b)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    n_pos = np.sum(diffs > 0); n_neg = np.sum(diffs < 0)
    return (n_pos - n_neg) / len(diffs)


def bootstrap_paired(err_a, err_b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(err_a)
    diffs = np.array(err_a) - np.array(err_b)
    obs = np.mean(diffs)
    boots = np.empty(n_boot)
    idx_all = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        boots[i] = np.mean(diffs[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    p = float(np.mean(boots <= 0) * 2) if obs > 0 else float(np.mean(boots >= 0) * 2)
    return float(obs), float(ci_lo), float(ci_hi), min(p, 1.0)


results = {}
for dataset in ["internal", "rsna_external", "mex_external"]:
    ids_wh = abs_err[dataset]["whole_hand"]["ids"]
    ids_f = abs_err[dataset]["fusion"]["ids"]
    common = sorted(set(ids_wh) & set(ids_f))
    map_wh = dict(zip(ids_wh, abs_err[dataset]["whole_hand"]["abs_err"]))
    map_f = dict(zip(ids_f, abs_err[dataset]["fusion"]["abs_err"]))
    err_wh = np.array([map_wh[i] for i in common])
    err_f = np.array([map_f[i] for i in common])
    stat, p_w = wilcoxon(err_wh, err_f) if np.any(err_wh != err_f) else (np.nan, 1.0)
    r = rank_biserial(err_wh, err_f)
    delta, ci_lo, ci_hi, p_b = bootstrap_paired(err_wh, err_f)
    results[dataset] = {
        "n_common": len(common), "whole_hand_mae": float(np.mean(err_wh)),
        "fusion_mae": float(np.mean(err_f)), "delta_mae": delta,
        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
        "wilcoxon_p": float(p_w), "bootstrap_p": p_b, "rank_biserial_r": r,
    }
    print(f"\n[{dataset}] n_common={len(common)} wh_MAE={np.mean(err_wh):.3f} "
          f"fusion_MAE={np.mean(err_f):.3f} dMAE={delta:+.3f} IC95=[{ci_lo:.3f},{ci_hi:.3f}] "
          f"p_wilcoxon={p_w:.4f} p_bootstrap={p_b:.4f} rank_biserial_r={r:.3f}")

out_path = f"{PROJECT_ROOT}/review2/results_json/whole_hand_significance.json"
with open(out_path, "w") as f:
    json.dump({"per_dataset_abs_err": abs_err, "paired_tests": results}, f, indent=2)
print(f"\nSaved to {out_path}")
