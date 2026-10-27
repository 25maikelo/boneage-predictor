import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/lustre/home/mlozano/boneage-predictor")
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter

from config.paths import VALIDATION_CSV, VALIDATION_IMAGES_DIR

sys.path.insert(0, "/lustre/home/mlozano/boneage-predictor/src")
import importlib.util
spec = importlib.util.spec_from_file_location("val_mod", "/lustre/home/mlozano/boneage-predictor/src/07_validation.py")
val_mod = importlib.util.module_from_spec(spec)
# avoid running main() on import
import types
sys.modules["val_mod"] = val_mod
spec.loader.exec_module(val_mod)

SEG_MODEL_PATH = "/lustre/home/mlozano/boneage-predictor/models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)

df = pd.read_csv(VALIDATION_CSV)
df.reset_index(drop=True, inplace=True)
df["boneage"] = df["boneage"].apply(val_mod.parse_age_to_months)

SEGMENTS_ORDER = ["pinky", "middle", "thumb", "wrist"]

failed = []
ok = 0
for i, (_, row) in enumerate(df.iterrows()):
    sid = str(row["id"])
    img_path = os.path.join(VALIDATION_IMAGES_DIR, f"{sid}.png")
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        failed.append((sid, "not_found", row.get("boneage"))); continue
    try:
        zoomed = val_mod.frame_and_zoom(gray)
        eq = val_mod.clahe_equalize(zoomed)
    except Exception as e:
        failed.append((sid, f"preprocess: {e}", row.get("boneage"))); continue
    try:
        segments = val_mod.segment_spatial(eq, seg_model, SEGMENTS_ORDER)
        empty = [seg for seg, s in segments.items() if np.sum(s) == 0]
        if empty:
            failed.append((sid, f"empty_segment: {','.join(empty)}", row.get("boneage"))); continue
    except Exception as e:
        failed.append((sid, f"segment: {e}", row.get("boneage"))); continue
    ok += 1
    if i % 200 == 0:
        print(f"{i}/{len(df)} processed, ok={ok}, failed={len(failed)}", flush=True)

print(f"\nTOTAL: {len(df)}  ok={ok}  failed={len(failed)}")
reasons = Counter(f[1].split(":")[0] for f in failed)
print("Reason breakdown:", dict(reasons))

import json
with open("/lustre/home/mlozano/boneage-predictor/review2/results_json/val_failures_diagnosis.json", "w") as f:
    json.dump({
        "total": len(df), "ok": ok, "failed": len(failed),
        "reason_counts": dict(reasons),
        "failed_detail": [{"id": s, "reason": r, "boneage": (None if pd.isna(a) else float(a))} for s, r, a in failed],
    }, f, indent=2)
print("Saved to review2/results_json/val_failures_diagnosis.json")
