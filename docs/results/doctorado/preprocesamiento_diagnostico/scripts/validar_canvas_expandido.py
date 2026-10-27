"""
Valida el fix de "canvas expandido" para 02_frame_and_zoom.py: en vez de rotar
dentro del mismo tamano de canvas (w,h), que puede recortar contenido que gira
fuera del cuadro cuando la mano no esta centrada, se expande el canvas al
tamano minimo necesario para contener la imagen rotada completa antes de
calcular el recorte final.

Mide, sobre una muestra del dataset completo:
  - cuantas imagenes producen una coordenada de recorte negativa con el
    metodo actual (sintoma del bug: wraparound de indices negativos en numpy,
    produce un recorte chico/incorrecto en vez de un error)
  - el estado (ok/discard_area/etc.) actual vs. con el fix para esos casos
  - fill ratio actual vs. con el fix, en la muestra completa

Uso:
    python docs/results/doctorado/preprocesamiento_diagnostico/scripts/validar_canvas_expandido.py
"""
import os, sys, json, random, time
from collections import Counter

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/doctorado/preprocesamiento_diagnostico")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config.paths import RAW_IMAGES_DIR

SAMPLE_SIZE = 1500
RANDOM_SEED = 7


def verificar_inversion(thresh):
    h, w = thresh.shape
    bordes = (
        np.any(thresh[0, :] == 255) or np.any(thresh[-1, :] == 255) or
        np.any(thresh[:, 0] == 255) or np.any(thresh[:, -1] == 255)
    )
    if bordes:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def detect_box(img, kernel_size=(9, 9), dilation_iters=4):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilation_iters)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    angle = rect[-1]
    w_r, h_r = rect[1]
    if w_r < h_r:
        angle += 90
    if angle > 45:
        angle -= 90
    return box, angle


def crop_actual(img):
    detected = detect_box(img)
    if detected is None:
        return img, "no_contour"
    box, angle = detected
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((len(box), 1))
    box_h = np.hstack([box, ones])
    box_r = np.dot(M, box_h.T).T.astype(int)
    x0, y0 = box_r[:, 0].min(), box_r[:, 1].min()
    x1, y1 = box_r[:, 0].max(), box_r[:, 1].max()
    if x0 >= x1 or y0 >= y1:
        return img, "degenerate"
    cropped = rotated[y0:y1, x0:x1]
    if cropped.size < 0.2 * img.size:
        return img, "discard_area"
    return cropped, "ok", (x0, y0)


def crop_canvas_expandido(img):
    detected = detect_box(img)
    if detected is None:
        return img, "no_contour"
    box, angle = detected
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
    if x0 >= x1 or y0 >= y1:
        return img, "degenerate"
    cropped = rotated[y0:y1, x0:x1]
    if cropped.size < 0.2 * img.size:
        return img, "discard_area"
    return cropped, "ok"


def fill_ratio(crop):
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    return float(np.mean(thresh > 0))


def main():
    random.seed(RANDOM_SEED)
    files = random.sample(os.listdir(RAW_IMAGES_DIR), SAMPLE_SIZE)
    print(f"Muestra: {len(files)} imagenes (seed={RANDOM_SEED})")

    estados_actual, estados_fix = [], []
    fill_actual, fill_fix = [], []
    negative_cases = []

    t0 = time.time()
    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        r_actual = crop_actual(img)
        crop_a, estado_a = r_actual[0], r_actual[1]
        if len(r_actual) == 3:
            x0, y0 = r_actual[2]
            if x0 < 0 or y0 < 0:
                negative_cases.append(fn)
        estados_actual.append(estado_a)
        fill_actual.append(fill_ratio(crop_a))

        crop_f, estado_f = crop_canvas_expandido(img)
        estados_fix.append(estado_f)
        fill_fix.append(fill_ratio(crop_f))

        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

    fill_actual_arr = np.array(fill_actual)
    fill_fix_arr = np.array(fill_fix)

    resumen = {
        "n": len(fill_actual),
        "n_indice_negativo_metodo_actual": len(negative_cases),
        "pct_indice_negativo": 100 * len(negative_cases) / len(fill_actual),
        "estados_actual": dict(Counter(estados_actual)),
        "estados_fix": dict(Counter(estados_fix)),
        "fill_ratio_actual_mean": float(fill_actual_arr.mean()),
        "fill_ratio_fix_mean": float(fill_fix_arr.mean()),
        "pct_mejoran_con_fix": float(np.mean(fill_fix_arr > fill_actual_arr + 0.01)) * 100,
        "pct_empeoran_con_fix": float(np.mean(fill_fix_arr < fill_actual_arr - 0.01)) * 100,
        "casos_indice_negativo_que_pasan_a_ok_con_fix": sum(
            1 for fn in negative_cases
            if estados_fix[files.index(fn)] == "ok"
        ) if negative_cases else 0,
    }

    with open(os.path.join(OUT_DIR, "results_json", "validacion_canvas_expandido.json"), "w") as f:
        json.dump(resumen, f, indent=2)
    print("\nResultado:", json.dumps(resumen, indent=2))

    plt.figure(figsize=(7, 4.5))
    plt.hist(fill_actual_arr, bins=40, alpha=0.6, label="Actual (canvas fijo)", color="#5b8ff9")
    plt.hist(fill_fix_arr, bins=40, alpha=0.6, label="Canvas expandido (fix)", color="#2ca02c")
    plt.xlabel("Fill ratio"); plt.ylabel("Frecuencia"); plt.legend()
    plt.title(f"Fill ratio: canvas fijo vs. expandido (n={len(files)})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", "validacion_canvas_expandido.png"), dpi=150)
    plt.close()
    print("Guardado: figures/validacion_canvas_expandido.png")


if __name__ == "__main__":
    main()
