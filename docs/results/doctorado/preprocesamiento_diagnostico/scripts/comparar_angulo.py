"""
Compara la logica de angulo anterior (bug de rotacion espuria de 90 grados,
hallazgo 5 de diagnostico.md) contra el fix aplicado en 02_frame_and_zoom.py,
sobre los 4 casos conocidos del hallazgo 4.

Uso:
    python docs/results/doctorado/preprocesamiento_diagnostico/scripts/comparar_angulo.py
"""
import os, sys

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/doctorado/preprocesamiento_diagnostico")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.paths import RAW_IMAGES_DIR

CASOS = ["1748.png", "12207.png", "6940.png", "2813.png"]


def verificar_inversion(thresh):
    h, w = thresh.shape
    bordes = (
        np.any(thresh[0, :] == 255) or np.any(thresh[-1, :] == 255) or
        np.any(thresh[:, 0] == 255) or np.any(thresh[:, -1] == 255)
    )
    if bordes:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def detect(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=4)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    return rect, box


def angle_anterior(rect):
    raw_angle = rect[-1]
    w_r, h_r = rect[1]
    angle = raw_angle
    if w_r < h_r:
        angle += 90
    if angle > 45:
        angle -= 90
    return angle


def angle_fix(rect, tol=1.0):
    raw_angle = rect[-1]
    w_r, h_r = rect[1]
    angle = raw_angle
    if w_r < h_r and abs(raw_angle - 90) > tol:
        angle += 90
    if angle > 45:
        angle -= 90
    return angle


def rot_crop(img, angle, box):
    h, w = img.shape
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((len(box), 1))
    box_h = np.hstack([box, ones])
    box_r = np.dot(M, box_h.T).T.astype(int)
    x0, y0 = max(0, box_r[:, 0].min()), max(0, box_r[:, 1].min())
    x1, y1 = min(new_w, box_r[:, 0].max()), min(new_h, box_r[:, 1].max())
    return rotated[y0:y1, x0:x1]


def main():
    fig, axes = plt.subplots(len(CASOS), 3, figsize=(10, 4 * len(CASOS)))
    for i, fn in enumerate(CASOS):
        img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
        rect, box = detect(img)
        a_ant = angle_anterior(rect)
        a_fix = angle_fix(rect)
        c_ant = rot_crop(img, a_ant, box)
        c_fix = rot_crop(img, a_fix, box)

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"{fn}\noriginal")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(c_ant, cmap="gray")
        axes[i, 1].set_title(f"logica anterior\nangulo={a_ant:.0f} grados", color="#c0392b")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(c_fix, cmap="gray")
        axes[i, 2].set_title(f"fix aplicado\nangulo={a_fix:.0f} grados", color="#27ae60")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "figures", "comparacion_angulo.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
