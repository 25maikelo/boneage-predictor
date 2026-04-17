#!/usr/bin/env python3
"""
Prueba rápida del modelo de segmentación sobre imágenes de validación.
Genera visualizaciones en scripts/seg_test_output/.

Uso:
    python scripts/test_segmentation.py
    python scripts/test_segmentation.py --n 10 --model models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model
from config.paths import VALIDATION_CSV, VALIDATION_IMAGES_DIR, SEGMENTATION_MODEL_PATH

SEGMENTS_ORDER = ["pinky", "middle", "thumb", "wrist"]
COLORS = {
    "pinky":  (255,  80,  80),
    "middle": ( 80, 200,  80),
    "thumb":  ( 80, 130, 255),
    "wrist":  (255, 200,  50),
}
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "seg_test_output")


def run(model_path, n):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Modelo : {model_path}")
    model = load_model(model_path, compile=False)
    print(f"Output shape: {model.output_shape}")

    df = pd.read_csv(VALIDATION_CSV).head(n)
    ok = empty = not_found = 0

    for _, row in df.iterrows():
        sid = str(row["id"])
        img_path = os.path.join(VALIDATION_IMAGES_DIR, f"{sid}.png")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[NOT FOUND] {sid}")
            not_found += 1
            continue

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        inp = cv2.resize(rgb, (224, 224)) / 255.0
        pred = model.predict(inp[np.newaxis, ...], verbose=0)[0]
        mask = np.argmax(pred, axis=-1).astype(np.uint8)
        mask_full = cv2.resize(mask, (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        unique = np.unique(mask)
        seg_counts = {name: int(np.sum(mask == cid))
                      for cid, name in enumerate(SEGMENTS_ORDER, start=1)}
        has_empty = any(v == 0 for v in seg_counts.values())

        if has_empty:
            empty += 1
        else:
            ok += 1

        # Visualización
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cid, name in enumerate(SEGMENTS_ORDER, start=1):
            m = (mask_full == cid)
            overlay[m] = (overlay[m].astype(float) * 0.5 +
                          np.array(COLORS[name]) * 0.5).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(mask_full, cmap="tab10", vmin=0, vmax=4)
        axes[1].set_title("Máscara"); axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Overlay"); axes[2].axis("off")

        status = "OK" if not has_empty else "EMPTY_SEG"
        label = (f"ID: {sid} | clases: {unique.tolist()} | {status}\n" +
                 " | ".join(f"{k}:{v}px" for k, v in seg_counts.items()))
        fig.suptitle(label, fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{sid}_{status}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=80)
        plt.close(fig)
        print(f"[{status}] {sid} -> clases={unique.tolist()} | {seg_counts}")

    print(f"\nResultado: OK={ok}  EMPTY_SEG={empty}  NOT_FOUND={not_found}  (n={n})")
    print(f"Imágenes guardadas en: {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Ruta al modelo de segmentación (.h5). "
                             "Default: hand-detector_00")
    parser.add_argument("--n", type=int, default=5,
                        help="Número de imágenes a probar (default: 5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model or os.path.join(
        PROJECT_ROOT,
        "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
    )
    run(model_path, args.n)
