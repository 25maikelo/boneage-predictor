"""
Genera un PDF corto con el antes/despues del fix de canvas expandido en
02_frame_and_zoom.py (ver diagnostico.md, hallazgo 4).

Uso:
    python docs/results/preprocesamiento_diagnostico/scripts/generar_reporte_pdf.py
"""
import os, sys

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/preprocesamiento_diagnostico")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


def detect_box(img, kernel_size=(9, 9), dilation_iters=4):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = verificar_inversion(thresh)
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilation_iters)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    angle = rect[-1]
    w_r, h_r = rect[1]
    if w_r < h_r:
        angle += 90
    if angle > 45:
        angle -= 90
    return box, angle


def crop_antes(img):
    box, angle = detect_box(img)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((len(box), 1))
    box_h = np.hstack([box, ones])
    box_r = np.dot(M, box_h.T).T.astype(int)
    x0, y0 = box_r[:, 0].min(), box_r[:, 1].min()
    x1, y1 = box_r[:, 0].max(), box_r[:, 1].max()
    return rotated[y0:y1, x0:x1]


def crop_despues(img):
    box, angle = detect_box(img)
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
    out_path = os.path.join(OUT_DIR, "reporte_corrimiento_pixeles.pdf")
    with PdfPages(out_path) as pdf:
        # Portada
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.75, "Fix de recorte: canvas expandido",
                  ha="center", fontsize=20, weight="bold")
        fig.text(0.5, 0.70, "02_frame_and_zoom.py: antes y despues",
                  ha="center", fontsize=13, color="#444444")
        resumen = (
            "Problema: al rotar la imagen alrededor de su centro manteniendo el mismo\n"
            "tamano de canvas, la mano se puede salir del cuadro cuando esta lejos del\n"
            "centro. El recorte resultante pierde la mano por completo.\n\n"
            "Fix: expandir el canvas antes de rotar, para no perder contenido.\n\n"
            "Resultado validado sobre 1,500 imagenes:\n"
            "   Descartes (recorte fallido):  5.6% -> 2.1%  (63% menos)\n"
            "   Sin regresiones en el resto del dataset.\n\n"
            f"Casos mostrados en este reporte: {len(CASOS)}"
        )
        fig.text(0.1, 0.60, resumen, fontsize=11, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # Un caso por pagina
        for fn in CASOS:
            img = cv2.imread(os.path.join(RAW_IMAGES_DIR, fn), cv2.IMREAD_GRAYSCALE)
            antes = crop_antes(img)
            despues = crop_despues(img)

            fig, ax = plt.subplots(1, 3, figsize=(13, 5.5))
            ax[0].imshow(img, cmap="gray")
            ax[0].set_title("Original", fontsize=11)
            ax[0].axis("off")

            ax[1].imshow(antes, cmap="gray")
            ax[1].set_title(f"Antes (con bug)\n{antes.shape[1]}×{antes.shape[0]} px", fontsize=11, color="#c0392b")
            ax[1].axis("off")

            ax[2].imshow(despues, cmap="gray")
            ax[2].set_title(f"Despues (fix)\n{despues.shape[1]}×{despues.shape[0]} px", fontsize=11, color="#27ae60")
            ax[2].axis("off")

            fig.suptitle(f"Caso: {fn}", fontsize=13, weight="bold")
            plt.subplots_adjust(left=0.04, right=0.88, top=0.80, bottom=0.04, wspace=0.15)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Reporte guardado en: {out_path}")


if __name__ == "__main__":
    main()
