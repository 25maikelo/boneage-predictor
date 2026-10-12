"""
Valida sustraccion de fondo morfologica (top-hat / "rolling ball") como alternativa a
Otsu directo en 02_frame_and_zoom.py, sobre el dataset completo, sin modelo de
segmentacion. Ver docs/results/preprocesamiento_diagnostico/revision_literatura.md
seccion 3: es el metodo documentado para iluminacion/contraste desparejo (vs. la
propuesta mas simple de equalizeHist ya probada en validar_propuestas.py).

La imagen que se recorta sigue siendo la original sin alterar; la version con
sustraccion de fondo solo se usa para detectar mejor el contorno.

Uso:
    python docs/results/preprocesamiento_diagnostico/scripts/validar_background_subtraction.py
"""
import os, sys, json, time
from collections import Counter

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/preprocesamiento_diagnostico")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config.paths import RAW_IMAGES_DIR

# Kernel grande para estimar el "fondo" (debe ser mayor que la estructura de interes,
# es decir mayor que el grosor de los huesos/dedos en la imagen reescalada).
BG_KERNEL_SIZE = 51


def verificar_inversion(thresh):
    h, w = thresh.shape
    bordes = (
        np.any(thresh[0, :] == 255) or np.any(thresh[-1, :] == 255) or
        np.any(thresh[:, 0] == 255) or np.any(thresh[:, -1] == 255)
    )
    if bordes:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def background_subtract(img, kernel_size=BG_KERNEL_SIZE):
    """Top-hat morfologico: estima el fondo con una apertura de kernel grande y lo
    resta, aplanando iluminacion/contraste desparejo antes de umbralizar."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(img, background)
    return cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_crop(img_for_threshold, img_to_crop, kernel_size=(9, 9), dilation_iters=4):
    blur = cv2.GaussianBlur(img_for_threshold, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilation_iters)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_to_crop, "no_contour"

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    angle = rect[-1]
    w_rect, h_rect = rect[1]
    if w_rect < h_rect:
        angle += 90
    if angle > 45:
        angle -= 90
    h, w = img_for_threshold.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_to_crop, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((len(box), 1))
    box_h = np.hstack([box, ones])
    box_r = np.dot(M, box_h.T).T.astype(int)
    x0, y0 = box_r[:, 0].min(), box_r[:, 1].min()
    x1, y1 = box_r[:, 0].max(), box_r[:, 1].max()
    if x0 >= x1 or y0 >= y1:
        return img_to_crop, "degenerate"
    cropped = rotated[y0:y1, x0:x1]
    if cropped.size < 0.2 * img_for_threshold.size:
        return img_to_crop, "discard_area"
    return cropped, "ok"


def fill_ratio(crop_gray):
    blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    return float(np.mean(thresh > 0))


def main():
    files = sorted(os.listdir(RAW_IMAGES_DIR))
    n = len(files)
    print(f"Validando sustraccion de fondo morfologica sobre {n} imagenes")

    fill_actual, fill_bgsub = [], []
    estado_actual, estado_bgsub = [], []

    t0 = time.time()
    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_bgsub = background_subtract(img)

        crop_a, estado_a = compute_crop(img, img)
        crop_b, estado_b = compute_crop(img_bgsub, img)
        estado_actual.append(estado_a)
        estado_bgsub.append(estado_b)
        fill_actual.append(fill_ratio(crop_a))
        fill_bgsub.append(fill_ratio(crop_b))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n}  ({elapsed:.0f}s, ~{elapsed/(i+1)*n:.0f}s total estimado)", flush=True)

    fill_actual_arr = np.array(fill_actual)
    fill_bgsub_arr = np.array(fill_bgsub)
    peor_mask = fill_actual_arr < 0.3

    resumen = {
        "n": len(fill_actual),
        "kernel_size": BG_KERNEL_SIZE,
        "estados_actual": dict(Counter(estado_actual)),
        "estados_bgsub": dict(Counter(estado_bgsub)),
        "fill_ratio_actual_mean": float(fill_actual_arr.mean()),
        "fill_ratio_bgsub_mean": float(fill_bgsub_arr.mean()),
        "n_peores_actual_fillratio<0.3": int(peor_mask.sum()),
        "de_esos_peores_mejoran_con_bgsub_pct": float(
            np.mean(fill_bgsub_arr[peor_mask] > fill_actual_arr[peor_mask]) * 100
        ) if peor_mask.sum() > 0 else None,
        "fill_ratio_medio_de_peores_actual": float(fill_actual_arr[peor_mask].mean()) if peor_mask.sum() > 0 else None,
        "fill_ratio_medio_de_peores_bgsub": float(fill_bgsub_arr[peor_mask].mean()) if peor_mask.sum() > 0 else None,
        "pct_imagenes_que_empeoran_con_bgsub": float(np.mean(fill_bgsub_arr < fill_actual_arr - 0.01)) * 100,
    }
    with open(os.path.join(OUT_DIR, "results_json", "validacion_background_subtraction.json"), "w") as f:
        json.dump(resumen, f, indent=2)
    print("\nResultado:", json.dumps(resumen, indent=2))

    plt.figure(figsize=(7, 4.5))
    plt.hist(fill_actual_arr, bins=40, alpha=0.6, label="Otsu directo (actual)", color="#5b8ff9")
    plt.hist(fill_bgsub_arr, bins=40, alpha=0.6, label="Otsu + sustraccion de fondo previa", color="#2ca02c")
    plt.xlabel("Fill ratio"); plt.ylabel("Frecuencia"); plt.legend()
    plt.title(f"Fill ratio: Otsu directo vs. con sustraccion de fondo (n={n})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", "validacion_background_subtraction.png"), dpi=150)
    plt.close()
    print("Guardado: figures/validacion_background_subtraction.png")


if __name__ == "__main__":
    main()
