#!/usr/bin/env python3
"""
Paso 4: Segmentación de imágenes en 4 regiones anatómicas de la mano.
Usa el modelo de segmentación pre-entrenado para generar máscaras por segmento.

Soporta dos estrategias de recorte, seleccionables con --mode:
  - spatial (default): mantiene la posición y tamaño original de la imagen,
    solo enmascara el fondo fuera del segmento (comportamiento histórico).
  - cropped: recorta al bounding box del segmento (+ padding), sin preservar
    la posición espacial original — el segmento ocupa la mayor parte del frame.
  - both: genera ambas variantes en una sola pasada (una sola inferencia del
    modelo de segmentación por imagen).

Entrada:  images/imagenes_ecualizadas/
Salida:   images/segmentos_spatial/{pinky,middle,thumb,wrist}/   (modo spatial)
          images/segmented_cropped/{pinky,middle,thumb,wrist}/  (modo cropped)

Uso:
    python src/preprocessing/04_segment_images.py
    python src/preprocessing/04_segment_images.py --mode cropped --padding 0.15
    python src/preprocessing/04_segment_images.py --mode both
"""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model

from config.paths import (
    EQUALIZED_IMAGES_DIR, MASKS_DIR,
    SEGMENTED_IMAGES_DIR, SEGMENTED_CROPPED_IMAGES_DIR, get_segmentation_model_path
)
from config.segmentation import HAND_DETECTOR_RUN
from src.utils.timing import report_timing, setup_logging

START_TIME = time.time()

SEGMENT_CLASSES = ["pinky", "middle", "thumb", "wrist"]

OUTPUT_DIRS = {
    "spatial": SEGMENTED_IMAGES_DIR,
    "cropped": SEGMENTED_CROPPED_IMAGES_DIR,
}


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {[g.name for g in gpus]}")
    else:
        print("ADVERTENCIA: No se detectó GPU. Ejecutando en CPU.")


def apply_mask_preserve_position(original_image, mask, background_color=(0, 0, 0)):
    """Enmascara el fondo fuera del segmento, manteniendo tamaño y posición original."""
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
    return np.where(mask_3ch == 255, original_image,
                    np.array(background_color, dtype=np.uint8)).astype(np.uint8)


def apply_mask_crop_bbox(original_image, mask, padding_ratio=0.15, background_color=(0, 0, 0)):
    """Enmascara el fondo fuera del segmento y recorta al bounding box de la
    máscara (+ padding relativo), sin preservar la posición espacial original.

    Si la máscara no tiene píxeles activos (segmento no detectado), devuelve
    una imagen mínima de 1x1 con el color de fondo.
    """
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.full((1, 1, 3), background_color, dtype=np.uint8)

    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = int(round((x1 - x0 + 1) * padding_ratio))
    pad_y = int(round((y1 - y0 + 1) * padding_ratio))
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)

    masked = apply_mask_preserve_position(original_image, mask, background_color)
    return masked[y0:y1 + 1, x0:x1 + 1]


def segment_image(img_gray, model, modes=("spatial",), padding_ratio=0.15):
    """Corre el modelo de segmentación una vez y genera, para cada modo
    solicitado, un dict {class_name: imagen} con la estrategia correspondiente.

    Devuelve (pred_mask, {modo: {class_name: imagen}}).
    """
    h, w = img_gray.shape
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(rgb, (224, 224)) / 255.0
    pred_mask = np.argmax(model.predict(inp[np.newaxis, ...], verbose=0)[0], axis=-1)
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    results = {mode: {} for mode in modes}
    for class_id, class_name in enumerate(SEGMENT_CLASSES, start=1):
        mask_class = (pred_mask == class_id).astype(np.uint8)
        if "spatial" in modes:
            results["spatial"][class_name] = apply_mask_preserve_position(img_gray, mask_class)
        if "cropped" in modes:
            results["cropped"][class_name] = apply_mask_crop_bbox(img_gray, mask_class, padding_ratio)
    return pred_mask, results


def setup_output_dirs(modes):
    os.makedirs(MASKS_DIR, exist_ok=True)
    for mode in modes:
        out_dir = OUTPUT_DIRS[mode]
        os.makedirs(out_dir, exist_ok=True)
        for cls in SEGMENT_CLASSES:
            os.makedirs(os.path.join(out_dir, cls), exist_ok=True)


def procesar_todas(model, modes, padding_ratio):
    files = [f for f in os.listdir(EQUALIZED_IMAGES_DIR) if f.lower().endswith(".png")]
    if not files:
        print(f"No se encontraron imágenes en {EQUALIZED_IMAGES_DIR}")
        return

    print(f"Segmentando {len(files)} imágenes (modos: {', '.join(modes)})...")
    for i, fname in enumerate(files, 1):
        image_id = os.path.splitext(fname)[0]
        src = os.path.join(EQUALIZED_IMAGES_DIR, fname)
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Error al cargar: {fname}")
            continue

        pred_mask, results = segment_image(img, model, modes=modes, padding_ratio=padding_ratio)

        # Guardar máscara (una sola vez, independiente del modo)
        max_val = pred_mask.max() if pred_mask.max() > 0 else 1
        cv2.imwrite(
            os.path.join(MASKS_DIR, f"{image_id}.png"),
            (pred_mask * 255 // max_val).astype(np.uint8)
        )

        # Guardar segmentos por cada modo solicitado, en su directorio independiente
        for mode in modes:
            out_dir = OUTPUT_DIRS[mode]
            for cls_name, seg_img in results[mode].items():
                cv2.imwrite(os.path.join(out_dir, cls_name, f"{image_id}.png"), seg_img)

        if i % 200 == 0:
            print(f"  {i}/{len(files)} procesadas")

    for mode in modes:
        print(f"Segmentos ({mode}) guardados en: {OUTPUT_DIRS[mode]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["spatial", "cropped", "both"], default="spatial",
        help="spatial=mantiene posición original (default, comportamiento histórico); "
             "cropped=recorta al bounding box del segmento; both=genera ambas variantes"
    )
    parser.add_argument(
        "--padding", type=float, default=0.15,
        help="Padding relativo al bounding box para el modo 'cropped' (default 0.15 = 15%%)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    modes = ["spatial", "cropped"] if args.mode == "both" else [args.mode]

    setup_logging("04_segment_images.py")
    setup_gpu()
    setup_output_dirs(modes)

    seg_model_path = get_segmentation_model_path(HAND_DETECTOR_RUN)
    print(f"Cargando modelo de segmentación: {seg_model_path}")
    seg_model = load_model(seg_model_path, compile=False)

    procesar_todas(seg_model, modes, args.padding)
    report_timing(START_TIME, "04_segment_images.py")
