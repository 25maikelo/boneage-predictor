"""
Valida los dos proximos pasos propuestos en diagnostico.md, sobre el dataset completo
(12,813 imagenes raw), sin depender de ningun modelo de segmentacion:

1. clipLimit=4.0 como nuevo default de 03_histogram_equalization.py (vs. clipLimit=2.0 actual).
2. Normalizar contraste antes del umbral de Otsu en 02_frame_and_zoom.py (la imagen que se
   recorta sigue siendo la original sin ecualizar; el ecualizado solo se usa para detectar
   mejor el contorno).

Uso:
    python docs/results/preprocesamiento_diagnostico/scripts/validar_propuestas.py
"""
import os, sys, json, time

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/preprocesamiento_diagnostico")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config.paths import RAW_IMAGES_DIR


# ---------------------------------------------------------------------------
# Geometria de 02_frame_and_zoom.py (identica a auditoria_preprocesamiento.py)
# ---------------------------------------------------------------------------

def verificar_inversion(thresh):
    h, w = thresh.shape
    bordes = (
        np.any(thresh[0, :] == 255) or np.any(thresh[-1, :] == 255) or
        np.any(thresh[:, 0] == 255) or np.any(thresh[:, -1] == 255)
    )
    if bordes:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def compute_crop(img_for_threshold, img_to_crop, kernel_size=(9, 9), dilation_iters=4):
    """Calcula el umbral/contorno sobre `img_for_threshold` (puede ser una version
    realzada solo para deteccion) y aplica el recorte resultante sobre `img_to_crop`
    (la imagen real que se guarda, sin alterar). Si son la misma imagen, es el
    comportamiento actual de 02_frame_and_zoom.py."""
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


def clahe_variant(img, clip_limit, tile_grid=(8, 8), blend=0.4):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    eq = clahe.apply(img)
    blended = cv2.addWeighted(img, 1 - blend, eq, blend, 0)
    return cv2.convertScaleAbs(blended, alpha=0.9, beta=-10)


def image_quality_metrics(img):
    std = float(np.std(img))
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    p = hist / hist.sum()
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
    clip_rate = float(np.mean((img <= 1) | (img >= 254)))
    return {"std": std, "entropy": entropy, "clip_rate": clip_rate}


def main():
    files = sorted(os.listdir(RAW_IMAGES_DIR))
    n = len(files)
    print(f"Validando sobre el dataset completo: {n} imagenes")

    # ---- Propuesta 1: clipLimit=4.0 vs 2.0 (actual), pareado por imagen ----
    clahe_actual, clahe_propuesto, clahe_sin = [], [], []
    # ---- Propuesta 2: Otsu con y sin normalizacion de contraste previa ----
    fill_actual, fill_propuesto, estado_actual, estado_propuesto = [], [], [], []

    t0 = time.time()
    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Propuesta 1
        clahe_sin.append(image_quality_metrics(img))
        clahe_actual.append(image_quality_metrics(clahe_variant(img, 2.0)))
        clahe_propuesto.append(image_quality_metrics(clahe_variant(img, 4.0)))

        # Propuesta 2: version normalizada SOLO para detectar el contorno;
        # el recorte final sigue aplicandose sobre la imagen original `img`.
        img_norm = cv2.equalizeHist(img)

        crop_actual, estado_a = compute_crop(img, img)
        crop_propuesto, estado_p = compute_crop(img_norm, img)
        estado_actual.append(estado_a)
        estado_propuesto.append(estado_p)
        fill_actual.append(fill_ratio(crop_actual))
        fill_propuesto.append(fill_ratio(crop_propuesto))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n}  ({elapsed:.0f}s, ~{elapsed/(i+1)*n:.0f}s total estimado)")

    # ---- Resumen propuesta 1 ----
    def agg(metrics_list):
        arr = {k: np.array([m[k] for m in metrics_list]) for k in ["std", "entropy", "clip_rate"]}
        return {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in arr.items()}

    resumen_clahe = {
        "n": len(clahe_actual),
        "sin_clahe": agg(clahe_sin),
        "actual_clip2.0": agg(clahe_actual),
        "propuesto_clip4.0": agg(clahe_propuesto),
    }
    # Comparacion pareada (por imagen) de contraste y saturacion
    std_actual = np.array([m["std"] for m in clahe_actual])
    std_propuesto = np.array([m["std"] for m in clahe_propuesto])
    clip_actual = np.array([m["clip_rate"] for m in clahe_actual])
    clip_propuesto = np.array([m["clip_rate"] for m in clahe_propuesto])
    resumen_clahe["pareado"] = {
        "pct_imagenes_con_mas_contraste_en_propuesto": float(np.mean(std_propuesto > std_actual)) * 100,
        "pct_imagenes_con_menos_saturacion_en_propuesto": float(np.mean(clip_propuesto < clip_actual)) * 100,
        "delta_std_medio": float(np.mean(std_propuesto - std_actual)),
        "delta_clip_rate_medio": float(np.mean(clip_propuesto - clip_actual)),
    }

    with open(os.path.join(OUT_DIR, "results_json", "validacion_clahe_clip4.json"), "w") as f:
        json.dump(resumen_clahe, f, indent=2)
    print("\nPropuesta 1 (clipLimit=4.0):", json.dumps(resumen_clahe["pareado"], indent=2))

    # ---- Resumen propuesta 2 ----
    fill_actual_arr = np.array(fill_actual)
    fill_propuesto_arr = np.array(fill_propuesto)
    peor_mask = fill_actual_arr < 0.3  # el subconjunto de "recorte casi totalmente fallido"

    from collections import Counter
    resumen_otsu = {
        "n": len(fill_actual),
        "estados_actual": dict(Counter(estado_actual)),
        "estados_propuesto": dict(Counter(estado_propuesto)),
        "fill_ratio_actual_mean": float(fill_actual_arr.mean()),
        "fill_ratio_propuesto_mean": float(fill_propuesto_arr.mean()),
        "n_peores_actual_fillratio<0.3": int(peor_mask.sum()),
        "de_esos_peores_mejoran_con_propuesto_pct": float(
            np.mean(fill_propuesto_arr[peor_mask] > fill_actual_arr[peor_mask]) * 100
        ) if peor_mask.sum() > 0 else None,
        "fill_ratio_medio_de_peores_actual": float(fill_actual_arr[peor_mask].mean()) if peor_mask.sum() > 0 else None,
        "fill_ratio_medio_de_peores_propuesto": float(fill_propuesto_arr[peor_mask].mean()) if peor_mask.sum() > 0 else None,
        "pct_imagenes_que_empeoran_con_propuesto": float(np.mean(fill_propuesto_arr < fill_actual_arr - 0.01)) * 100,
    }
    with open(os.path.join(OUT_DIR, "results_json", "validacion_otsu_contraste_previo.json"), "w") as f:
        json.dump(resumen_otsu, f, indent=2)
    print("\nPropuesta 2 (normalizar antes de Otsu):", json.dumps(resumen_otsu, indent=2))

    # ---- Figuras ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].hist(std_actual, bins=40, alpha=0.6, label="clipLimit=2.0 (actual)", color="#5b8ff9")
    ax[0].hist(std_propuesto, bins=40, alpha=0.6, label="clipLimit=4.0 (propuesto)", color="#ff9900")
    ax[0].set_xlabel("Contraste (std)"); ax[0].set_ylabel("Frecuencia"); ax[0].legend()
    ax[0].set_title(f"Contraste: actual vs. propuesto (n={n})")

    ax[1].hist(fill_actual_arr, bins=40, alpha=0.6, label="Otsu directo (actual)", color="#5b8ff9")
    ax[1].hist(fill_propuesto_arr, bins=40, alpha=0.6, label="Otsu con equalizeHist previo", color="#ff9900")
    ax[1].set_xlabel("Fill ratio"); ax[1].set_ylabel("Frecuencia"); ax[1].legend()
    ax[1].set_title(f"Fill ratio: actual vs. propuesto (n={n})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", "validacion_propuestas.png"), dpi=150)
    plt.close()
    print("\nGuardado: figures/validacion_propuestas.png")


if __name__ == "__main__":
    main()
