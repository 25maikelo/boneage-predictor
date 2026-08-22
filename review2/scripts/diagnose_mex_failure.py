import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/lustre/home/mlozano/boneage-predictor")
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from config.paths import MEX_CSV, MEX_IMAGES_DIR
from config.experiment import load_experiment_config, get_experiment_output_dir

import importlib.util
spec = importlib.util.spec_from_file_location("mex_mod", "/lustre/home/mlozano/boneage-predictor/src/08_mex_validation.py")
mex_mod = importlib.util.module_from_spec(spec)
sys.modules["mex_mod"] = mex_mod
spec.loader.exec_module(mex_mod)

cfg = load_experiment_config(27)
SEG_MODEL_PATH = "/lustre/home/mlozano/boneage-predictor/models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)

df = pd.read_csv(MEX_CSV)
df.reset_index(drop=True, inplace=True)
df["bone_age"] = df["bone_age"].apply(mex_mod.parse_age_to_months)

SEGMENTS_ORDER = cfg.SEGMENTS_ORDER

for i, (_, row) in enumerate(df.iterrows()):
    sid = str(row["ID"])
    bone_age_true = float(row["bone_age"])
    img_path = os.path.join(MEX_IMAGES_DIR, f"{sid}.png")
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(sid, "not_found", "bone_age=", bone_age_true); continue
    try:
        zoomed = mex_mod.frame_and_zoom(gray)
        eq = mex_mod.clahe_equalize(zoomed)
        segments = mex_mod.get_segments(eq, seg_model, SEGMENTS_ORDER, cfg)
        empty = [seg for seg, s in segments.items() if np.sum(s) == 0]
        if empty:
            print(sid, "empty_segment:", empty, "bone_age=", bone_age_true, "raw_bone_age=", row["bone_age"])
    except Exception as e:
        print(sid, "preprocess/segment exception:", e, "bone_age=", bone_age_true)
    if pd.isna(bone_age_true):
        print(sid, "NAN bone_age", "raw=", row["bone_age"])

print("done")
