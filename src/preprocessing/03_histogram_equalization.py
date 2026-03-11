#!/usr/bin/env python3
"""
Paso 3: Ecualización adaptativa de histograma (CLAHE).
Mejora el contraste de las imágenes recortadas.

Entrada:  images/imagenes_recortadas/
Salida:   images/imagenes_ecualizadas/

Uso:
    python src/preprocessing/03_histogram_equalization.py
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2

from config.paths import CROPPED_IMAGES_DIR, EQUALIZED_IMAGES_DIR
from src.utils.timing import report_timing, setup_logging

START_TIME = time.time()


def ecualizacion_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    blended = cv2.addWeighted(img, 0.6, eq, 0.4, 0)
    return cv2.convertScaleAbs(blended, alpha=0.9, beta=-10)


def procesar_todas():
    os.makedirs(EQUALIZED_IMAGES_DIR, exist_ok=True)
    files = [f for f in os.listdir(CROPPED_IMAGES_DIR) if f.lower().endswith(".png")]
    if not files:
        print(f"No se encontraron imágenes en {CROPPED_IMAGES_DIR}")
        return

    print(f"Ecualizando {len(files)} imágenes...")
    for i, fname in enumerate(files, 1):
        src = os.path.join(CROPPED_IMAGES_DIR, fname)
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Error al cargar: {fname}")
            continue
        eq = ecualizacion_clahe(img)
        cv2.imwrite(os.path.join(EQUALIZED_IMAGES_DIR, fname), eq)
        if i % 500 == 0:
            print(f"  {i}/{len(files)} ecualizadas")

    print(f"Imágenes ecualizadas guardadas en: {EQUALIZED_IMAGES_DIR}")


if __name__ == "__main__":
    setup_logging("03_histogram_equalization.py")
    procesar_todas()
    report_timing(START_TIME, "03_histogram_equalization.py")
