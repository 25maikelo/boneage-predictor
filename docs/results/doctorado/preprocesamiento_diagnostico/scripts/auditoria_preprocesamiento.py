"""
Auditoria de calidad de preprocesamiento (02_frame_and_zoom.py, 03_histogram_equalization.py),
SIN depender del segmentador ni de ningun modelo entrenado. Metricas propias de la imagen.

02_frame_and_zoom.py:
  - Tasa de activacion de las salvaguardas (sin contorno, recorte degenerado, area<20%).
  - "Fill ratio" post-recorte: que fraccion del recorte final es mano (primer plano) vs.
    fondo/borde de placa residual, re-umbralizando el propio recorte. Es el proxy objetivo
    de "recorte flojo" (ej. se ve el borde de la placa) sin usar Dice de ningun modelo.

03_histogram_equalization.py:
  - Contraste (desviacion estandar), entropia y tasa de saturacion (clipping en 0/255) del
    resultado, comparando la mezcla actual (60/40, clipLimit=2.0, tileGridSize=8x8) contra
    variantes de clipLimit y tileGridSize.

Uso:
    python docs/results/doctorado/preprocesamiento_diagnostico/scripts/auditoria_preprocesamiento.py
"""
import os, sys, json, random
from collections import Counter

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/doctorado/preprocesamiento_diagnostico")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config.paths import RAW_IMAGES_DIR

RANDOM_SEED = 42
SAMPLE_SIZE = 800
N_WORST_SHOWN = 5


# ---------------------------------------------------------------------------
# 02_frame_and_zoom.py: misma logica geometrica, solo lectura (no escribe nada
# en data/images/, no toca el pipeline de produccion).
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


def compute_crop(img_gray, kernel_size=(9, 9), dilation_iters=4):
    """Replica rotar_y_recortar de 02_frame_and_zoom.py. Devuelve (crop, estado)."""
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilation_iters)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_gray, "no_contour"

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    angle = rect[-1]
    w_rect, h_rect = rect[1]
    if w_rect < h_rect:
        angle += 90
    if angle > 45:
        angle -= 90
    h, w = img_gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((len(box), 1))
    box_h = np.hstack([box, ones])
    box_r = np.dot(M, box_h.T).T.astype(int)
    x0, y0 = box_r[:, 0].min(), box_r[:, 1].min()
    x1, y1 = box_r[:, 0].max(), box_r[:, 1].max()
    if x0 >= x1 or y0 >= y1:
        return img_gray, "degenerate"
    cropped = rotated[y0:y1, x0:x1]
    if cropped.size < 0.2 * img_gray.size:
        return img_gray, "discard_area"
    return cropped, "ok"


def fill_ratio(crop_gray):
    """Fraccion de primer plano (mano) en el recorte final, re-umbralizando el
    propio recorte. Proxy objetivo de "recorte flojo" (fondo/borde residual),
    sin usar ningun modelo de segmentacion."""
    blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    return float(np.mean(thresh > 0))


def audit_frame_and_zoom(files):
    estados = Counter()
    fill_ratios = {}
    for fn in files:
        img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            estados["no_load"] += 1
            continue
        crop, estado = compute_crop(img)
        estados[estado] += 1
        fill_ratios[fn] = fill_ratio(crop)

    ratios = np.array(list(fill_ratios.values()))
    summary = {
        "n": len(files),
        "estados": dict(estados),
        "fill_ratio_mean": float(ratios.mean()),
        "fill_ratio_std": float(ratios.std()),
        "fill_ratio_p10": float(np.percentile(ratios, 10)),
        "fill_ratio_p50": float(np.percentile(ratios, 50)),
        "fill_ratio_p90": float(np.percentile(ratios, 90)),
    }
    worst = sorted(fill_ratios.items(), key=lambda kv: kv[1])[:N_WORST_SHOWN]
    summary["peores_fill_ratio"] = worst

    with open(os.path.join(OUT_DIR, "results_json", "auditoria_02_frame_and_zoom.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("02_frame_and_zoom.py:", json.dumps(summary, indent=2))

    # Histograma de fill ratio
    plt.figure(figsize=(7, 4))
    plt.hist(ratios, bins=30, color="#5b8ff9", edgecolor="k", alpha=0.85)
    plt.axvline(ratios.mean(), color="red", linestyle="--", label=f"media={ratios.mean():.3f}")
    plt.xlabel("Fill ratio (fraccion de mano en el recorte final)")
    plt.ylabel("Frecuencia")
    plt.title(f"Calidad de recorte: 02_frame_and_zoom.py (n={len(files)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", "fill_ratio_histograma.png"), dpi=150)
    plt.close()

    # Visualizar los peores casos (menor fill ratio = recorte mas flojo)
    for rank, (fn, ratio) in enumerate(worst, 1):
        img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
        crop, estado = compute_crop(img)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"Raw ({fn})"); ax[0].axis("off")
        ax[1].imshow(crop, cmap="gray")
        ax[1].set_title(f"Recorte (fill_ratio={ratio:.3f}, estado={estado})")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "figures", f"peor_recorte_{rank}_{fn}"), dpi=150)
        plt.close()

    return summary


# ---------------------------------------------------------------------------
# 03_histogram_equalization.py: metricas de contraste/entropia/saturacion
# ---------------------------------------------------------------------------

def clahe_variant(img, clip_limit, tile_grid, blend=0.4):
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


def audit_histogram_equalization(files):
    variants = {
        "sin_clahe": None,
        "actual_clip2.0_tile8": (2.0, (8, 8)),
        "clip1.0_tile8": (1.0, (8, 8)),
        "clip4.0_tile8": (4.0, (8, 8)),
        "clip2.0_tile4": (2.0, (4, 4)),
        "clip2.0_tile16": (2.0, (16, 16)),
    }
    results = {name: [] for name in variants}
    for fn in files:
        img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for name, params in variants.items():
            out = img if params is None else clahe_variant(img, *params)
            results[name].append(image_quality_metrics(out))

    summary = {}
    for name, metrics_list in results.items():
        arr = {k: np.array([m[k] for m in metrics_list]) for k in ["std", "entropy", "clip_rate"]}
        summary[name] = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in arr.items()}

    with open(os.path.join(OUT_DIR, "results_json", "auditoria_03_histogram_equalization.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("03_histogram_equalization.py:", json.dumps(summary, indent=2))

    # Grafico comparativo de contraste y saturacion por variante
    names = list(variants.keys())
    stds = [summary[n]["std"]["mean"] for n in names]
    clips = [summary[n]["clip_rate"]["mean"] * 100 for n in names]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    ax1.bar(x - 0.2, stds, width=0.4, color="#5b8ff9", label="Contraste (std)")
    ax1.set_ylabel("Desviacion estandar de intensidad")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, clips, width=0.4, color="#ff9900", label="% pixeles saturados")
    ax2.set_ylabel("% pixeles saturados (clip)")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_title(f"Contraste vs. saturacion por variante CLAHE (n={len(files)})")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", "clahe_comparativa.png"), dpi=150)
    plt.close()

    return summary


def main():
    random.seed(RANDOM_SEED)
    all_files = sorted(os.listdir(RAW_IMAGES_DIR))
    sample = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    print(f"Muestra: {len(sample)} de {len(all_files)} imagenes raw (seed={RANDOM_SEED})")

    audit_frame_and_zoom(sample)
    audit_histogram_equalization(sample)


if __name__ == "__main__":
    main()
