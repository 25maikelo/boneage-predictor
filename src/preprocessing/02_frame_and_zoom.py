#!/usr/bin/env python3
"""
Paso 2: Rotación y recorte de imágenes de rayos X de mano.
Alinea horizontalmente el eje mayor de la mano y recorta el fondo.

Entrada:  images/boneage-training-dataset/
Salida:   images/imagenes_recortadas/

Uso:
    python src/preprocessing/02_frame_and_zoom.py
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from config.paths import RAW_IMAGES_DIR, CROPPED_IMAGES_DIR
from src.utils.timing import report_timing, setup_logging

START_TIME = time.time()


def verificar_inversion(thresh):
    h, w = thresh.shape
    bordes = (
        np.any(thresh[0, :] == 255) or np.any(thresh[-1, :] == 255) or
        np.any(thresh[:, 0] == 255) or np.any(thresh[:, -1] == 255)
    )
    if bordes:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def rotar_y_recortar(image_path, output_path, kernel_size=(9, 9), dilation_iters=4):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"No se pudo cargar: {image_path}")
        return

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)

    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilation_iters)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imwrite(output_path, img)
        return

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    angle = rect[-1]
    w_rect, h_rect = rect[1]

    if w_rect < h_rect:
        angle += 90
    if angle > 45:
        angle -= 90

    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    ones = np.ones((len(box), 1))
    box_h = np.hstack([box, ones])
    box_r = np.dot(M, box_h.T).T.astype(int)
    x0, y0 = box_r[:, 0].min(), box_r[:, 1].min()
    x1, y1 = box_r[:, 0].max(), box_r[:, 1].max()

    if x0 >= x1 or y0 >= y1:
        cv2.imwrite(output_path, img)
        return

    cropped = rotated[y0:y1, x0:x1]
    if cropped.size < 0.2 * img.size:
        cv2.imwrite(output_path, img)
        return

    cv2.imwrite(output_path, cropped)


def procesar_todas():
    os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)
    files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(".png")]
    if not files:
        print(f"No se encontraron imágenes en {RAW_IMAGES_DIR}")
        return

    print(f"Procesando {len(files)} imágenes...")
    for i, fname in enumerate(files, 1):
        src = os.path.join(RAW_IMAGES_DIR, fname)
        dst = os.path.join(CROPPED_IMAGES_DIR, fname)
        rotar_y_recortar(src, dst)
        if i % 500 == 0:
            print(f"  {i}/{len(files)} procesadas")

    print(f"Imágenes recortadas guardadas en: {CROPPED_IMAGES_DIR}")


if __name__ == "__main__":
    setup_logging("02_frame_and_zoom.py")
    procesar_todas()
    report_timing(START_TIME, "02_frame_and_zoom.py")
