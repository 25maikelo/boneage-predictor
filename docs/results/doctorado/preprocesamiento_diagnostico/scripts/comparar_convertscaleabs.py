"""
Compara cv2.convertScaleAbs (bug, hallazgo 6 de diagnostico.md) contra el
clip explicito aplicado en 03_histogram_equalization.py, sobre una imagen
real.

Uso:
    python docs/results/doctorado/preprocesamiento_diagnostico/scripts/comparar_convertscaleabs.py
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

CASO = "5803.png"


def main():
    img = cv2.imread(os.path.join(RAW_IMAGES_DIR, CASO), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    blended = cv2.addWeighted(img, 0.6, eq, 0.4, 0)

    antes = cv2.convertScaleAbs(blended, alpha=0.9, beta=-10)
    despues = np.clip(blended.astype(np.float32) * 0.9 - 10, 0, 255).astype(np.uint8)

    diff = antes.astype(int) - despues.astype(int)
    print(f"Minimo real de blended: {blended.min()}")
    print(f"Fondo minimo antes (bug): {antes.min()}  despues (fix): {despues.min()}")
    print(f"Pixeles con diferencia >3 niveles: {int(np.sum(diff > 3))}/{antes.size} "
          f"({100*np.mean(diff > 3):.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(antes, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Antes (bug): fondo min={antes.min()}", color="#c0392b")
    axes[1].axis("off")
    axes[2].imshow(despues, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Despues (fix): fondo min={despues.min()}", color="#27ae60")
    axes[2].axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "figures", "fix_convertscaleabs.png")
    plt.savefig(out_path, dpi=120)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
