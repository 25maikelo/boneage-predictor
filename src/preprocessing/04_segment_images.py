#!/usr/bin/env python3
"""
Paso 4: Segmentación de imágenes en 4 regiones anatómicas de la mano.
Usa el modelo de segmentación pre-entrenado para generar máscaras por segmento
conservando la posición espacial original.

Entrada:  images/imagenes_ecualizadas/
Salida:   images/segmentos_spatial/{pinky,middle,thumb,wrist}/

Uso:
    python src/preprocessing/04_segment_images.py
"""
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
    SEGMENTED_IMAGES_DIR, get_segmentation_model_path
)
from config.segmentation import HAND_DETECTOR_RUN
from src.utils.timing import report_timing, setup_logging

START_TIME = time.time()

SEGMENT_CLASSES = ["pinky", "middle", "thumb", "wrist"]


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {[g.name for g in gpus]}")
    else:
        print("ADVERTENCIA: No se detectó GPU. Ejecutando en CPU.")


def apply_mask_preserve_position(original_image, mask, background_color=(0, 0, 0)):
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
    return np.where(mask_3ch == 255, original_image,
                    np.array(background_color, dtype=np.uint8)).astype(np.uint8)


def segment_image(img_gray, model):
    """Genera un dict {class_name: segmented_image} para una imagen en escala de grises."""
    h, w = img_gray.shape
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(rgb, (224, 224)) / 255.0
    pred_mask = np.argmax(model.predict(inp[np.newaxis, ...], verbose=0)[0], axis=-1)
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    segments = {}
    for class_id, class_name in enumerate(SEGMENT_CLASSES, start=1):
        mask_class = (pred_mask == class_id).astype(np.uint8)
        segments[class_name] = apply_mask_preserve_position(img_gray, mask_class)
    return pred_mask, segments


def setup_output_dirs():
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(SEGMENTED_IMAGES_DIR, exist_ok=True)
    for cls in SEGMENT_CLASSES:
        os.makedirs(os.path.join(SEGMENTED_IMAGES_DIR, cls), exist_ok=True)


def procesar_todas(model):
    files = [f for f in os.listdir(EQUALIZED_IMAGES_DIR) if f.lower().endswith(".png")]
    if not files:
        print(f"No se encontraron imágenes en {EQUALIZED_IMAGES_DIR}")
        return

    print(f"Segmentando {len(files)} imágenes...")
    for i, fname in enumerate(files, 1):
        image_id = os.path.splitext(fname)[0]
        src = os.path.join(EQUALIZED_IMAGES_DIR, fname)
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Error al cargar: {fname}")
            continue

        pred_mask, segments = segment_image(img, model)

        # Guardar máscara
        max_val = pred_mask.max() if pred_mask.max() > 0 else 1
        cv2.imwrite(
            os.path.join(MASKS_DIR, f"{image_id}.png"),
            (pred_mask * 255 // max_val).astype(np.uint8)
        )

        # Guardar segmentos con posición espacial
        for cls_name, seg_img in segments.items():
            cv2.imwrite(os.path.join(SEGMENTED_IMAGES_DIR, cls_name, f"{image_id}.png"), seg_img)

        if i % 200 == 0:
            print(f"  {i}/{len(files)} procesadas")

    print(f"Segmentos guardados en: {SEGMENTED_IMAGES_DIR}")


if __name__ == "__main__":
    setup_logging("04_segment_images.py")
    setup_gpu()
    setup_output_dirs()

    seg_model_path = get_segmentation_model_path(HAND_DETECTOR_RUN)
    print(f"Cargando modelo de segmentación: {seg_model_path}")
    seg_model = load_model(seg_model_path, compile=False)

    procesar_todas(seg_model)
    report_timing(START_TIME, "04_segment_images.py")
