"""
Genera un PDF resumen corto de toda la etapa de auditoria de preprocesamiento
(02_frame_and_zoom.py y 03_histogram_equalization.py): hallazgos, fixes
aplicados, ejemplos de antes/despues y la nueva parametrizacion para
ablaciones. Pensado para presentar a los asesores, no para documentacion
tecnica exhaustiva (esa vive en diagnostico.md).

Reusa las figuras ya generadas y validadas en figures/ en vez de recalcular
nada, para evitar introducir nuevos bugs en un script de solo reporte.

Uso:
    python docs/results/doctorado/preprocesamiento_diagnostico/scripts/generar_reporte_resumen.py
"""
import os
import sys
import json
import textwrap

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/doctorado/preprocesamiento_diagnostico")
FIG_DIR = os.path.join(OUT_DIR, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PAGE = (11, 8.5)  # horizontal, mejor para figuras anchas


def pagina_imagen(pdf, fig_path, titulo, pie=None):
    img = plt.imread(fig_path)
    fig = plt.figure(figsize=PAGE)
    ax_img = fig.add_axes([0.03, 0.08, 0.94, 0.80])
    ax_img.imshow(img)
    ax_img.axis("off")
    fig.text(0.5, 0.95, titulo, ha="center", fontsize=15, weight="bold")
    if pie:
        fig.text(0.5, 0.03, pie, ha="center", fontsize=9.5, color="#444444")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    out_path = os.path.join(OUT_DIR, "reporte_resumen_preprocesamiento.pdf")

    with open(os.path.join(PROJECT_ROOT, "data/images/crop_fallbacks.json"), encoding="utf-8") as f:
        fallbacks = json.load(f)
    n_fallback = len(fallbacks)
    n_total = 12811

    with PdfPages(out_path) as pdf:
        # ---- Portada ----
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.78, "Auditoria del preprocesamiento", ha="center", fontsize=24, weight="bold")
        fig.text(0.5, 0.71, "02_frame_and_zoom.py  +  03_histogram_equalization.py",
                 ha="center", fontsize=13, color="#444444", family="monospace")
        contexto = (
            "Motivacion: en la defensa de tesis se pidio justificar los parametros de\n"
            "CLAHE y Otsu usados en el pipeline. Esto llevo a una auditoria sistematica\n"
            "de ambos scripts de preprocesamiento, usando metricas propias de la imagen\n"
            "(sin depender del segmentador ni de sus metricas).\n\n"
            "Resultado: 4 defectos encontrados y corregidos en produccion, los\n"
            "parametros quedaron centralizados en config/preprocessing.py para poder\n"
            "ablacionarlos, y el dataset completo (12,811 imagenes) se re-genero con\n"
            "todos los fixes aplicados.\n\n"
            "Documentacion tecnica completa: docs/results/doctorado/preprocesamiento_diagnostico/"
        )
        fig.text(0.5, 0.60, contexto, ha="center", va="top", fontsize=11.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Resumen de hallazgos ----
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.93, "Resumen de hallazgos", ha="center", fontsize=17, weight="bold")
        filas = [
            ("1", "Recorte deja 27% de fondo residual en promedio", "Diagnosticado",
             "Cuantificado (fill ratio); causa raiz en hallazgos 2, 4 y 5"),
            ("2", "Bajo contraste global rompe la separacion de Otsu", "Sin fix automatico",
             "2 alternativas probadas y descartadas (no superan el 5% actual); "
             "se documenta y registra en vez de forzar un recorte"),
            ("3", "clipLimit=2.0 de CLAHE no es el optimo", "Corregido",
             "clipLimit=4.0: mas contraste, menos saturacion (validado en 12,811 img.)"),
            ("4", "Bug: la rotacion pierde contenido (canvas fijo)", "Corregido",
             "Canvas expandido antes de rotar. Descartes 5.6% -> 2.1% (n=1500)"),
            ("5", "Bug: rotacion espuria de 90 en manos ya alineadas", "Corregido",
             "Tolerancia en el angulo de frontera (90 +/- 1). Afectaba 2.5-3.5% de la muestra"),
            ("6", "Bug: convertScaleAbs reflejaba tonos oscuros del fondo", "Corregido",
             "Reemplazado por recorte explicito a [0, 255]"),
        ]
        col_x = [0.05, 0.09, 0.42, 0.58]
        y = 0.84
        fig.text(col_x[0], y, "#", fontsize=10.5, weight="bold")
        fig.text(col_x[1], y, "Hallazgo", fontsize=10.5, weight="bold")
        fig.text(col_x[2], y, "Estado", fontsize=10.5, weight="bold")
        fig.text(col_x[3], y, "Resultado", fontsize=10.5, weight="bold")
        fig.add_artist(plt.Line2D([0.05, 0.96], [y - 0.02, y - 0.02], color="black", lw=0.8, transform=fig.transFigure))
        y -= 0.06
        color_estado = {"Diagnosticado": "#444444", "Sin fix automatico": "#b9770e", "Corregido": "#27ae60"}
        line_frac = 1.2 / 72 / PAGE[1]  # fraccion de figura por punto de fuente, 1 linea
        for num, hallazgo, estado, resultado in filas:
            hallazgo_w = textwrap.fill(hallazgo, width=38)
            resultado_w = textwrap.fill(resultado, width=32)
            n_lineas = max(hallazgo_w.count("\n"), resultado_w.count("\n")) + 1
            fig.text(col_x[0], y, num, fontsize=10, va="top")
            fig.text(col_x[1], y, hallazgo_w, fontsize=10, va="top")
            fig.text(col_x[2], y, estado, fontsize=10, color=color_estado[estado], weight="bold", va="top")
            fig.text(col_x[3], y, resultado_w, fontsize=9, va="top")
            y -= 10 * line_frac * n_lineas + 0.025
        fig.text(0.06, 0.08,
                  "Ademas: se reviso 04_segment_images.py con el mismo nivel de detalle; "
                  "sin defectos equivalentes encontrados.",
                  fontsize=9.5, color="#444444", style="italic")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Caso 1: corrimiento de pixeles ----
        pagina_imagen(
            pdf, os.path.join(FIG_DIR, "bug_corrimiento_1748.png"),
            "Hallazgo 4: la rotacion perdia la mano por completo",
            "Al rotar alrededor del centro de la imagen con un canvas del mismo tamano, el contenido que cae\n"
            "fuera del cuadro se pierde. Caso 1748.png: el recorte resultante era una tira de ruido sin la mano.",
        )

        # ---- Caso 2: rotacion espuria ----
        pagina_imagen(
            pdf, os.path.join(FIG_DIR, "comparacion_angulo.png"),
            "Hallazgo 5: manos ya alineadas se rotaban 90 grados de mas",
            "cv2.minAreaRect devuelve angulo=90 como valor de frontera para rectangulos ya alineados a los ejes;\n"
            "la logica anterior lo interpretaba como 'girar 90'. Afectaba 2.5-3.5% de las imagenes.",
        )

        # ---- Validacion cuantitativa del recorte ----
        fig = plt.figure(figsize=PAGE)
        img = plt.imread(os.path.join(FIG_DIR, "validacion_canvas_expandido.png"))
        ax_img = fig.add_axes([0.08, 0.12, 0.55, 0.72])
        ax_img.imshow(img)
        ax_img.axis("off")
        fig.text(0.5, 0.92, "Validacion a escala: menos descartes, sin regresiones", ha="center", fontsize=15, weight="bold")
        resultado = (
            "Con los fixes de los\n"
            "hallazgos 4 y 5:\n\n"
            f"Descartes (n=1500)\n"
            "  Antes:    5.6%\n"
            "  Despues:  2.1%\n"
            "  (-63%)\n\n"
            "Dataset completo\n"
            f"(12,811 imagenes,\n"
            "ya regenerado):\n"
            f"  Fallback: {n_fallback}/{n_total}\n"
            f"  ({100*n_fallback/n_total:.1f}%)\n\n"
            "Todo el fallback\n"
            "restante es por bajo\n"
            "contraste (hallazgo 2),\n"
            "no por los bugs\n"
            "geometricos."
        )
        fig.text(0.68, 0.80, resultado, va="top", fontsize=11, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- CLAHE ----
        pagina_imagen(
            pdf, os.path.join(FIG_DIR, "clahe_comparativa.png"),
            "Hallazgo 3: clipLimit=4.0 preserva mas contraste",
            "Muestra de 800 imagenes. clipLimit=4.0 (mismo tile 8x8) da mas contraste y menos saturacion que el\n"
            "clipLimit=2.0 anterior. Confirmado en las 12,811 imagenes completas: 85.4% ganan contraste con el cambio.",
        )

        # ---- convertScaleAbs ----
        pagina_imagen(
            pdf, os.path.join(FIG_DIR, "fix_convertscaleabs.png"),
            "Hallazgo 6: el ajuste final de contraste reflejaba el fondo",
            "cv2.convertScaleAbs calcula un valor absoluto, no un recorte: los tonos que debian saturar a 0\n"
            "(fondo oscuro) se reflejaban hacia arriba. Impacto sutil pero real; corregido con np.clip.",
        )

        # ---- Parametrizacion para ablaciones ----
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.93, "Parametros centralizados para ablaciones futuras", ha="center", fontsize=17, weight="bold")
        fig.text(0.5, 0.86, "config/preprocessing.py", ha="center", fontsize=12, family="monospace", color="#444444")
        params = [
            ("GAUSSIAN_BLUR_KERNEL", "(5, 5)", "Suavizado antes de Otsu"),
            ("DILATION_KERNEL_SIZE / ITERATIONS", "(9,9) / 4", "Cierre de huecos antes del contorno"),
            ("ANGLE_BOUNDARY_TOLERANCE", "1.0 grados", "Tolerancia del angulo de frontera (hallazgo 5)"),
            ("MIN_CROP_AREA_RATIO", "0.2", "Umbral de descarte del recorte"),
            ("CLAHE_CLIP_LIMIT", "4.0", "Limite de recorte del histograma CLAHE"),
            ("CLAHE_TILE_GRID_SIZE", "(8, 8)", "Tamano de grilla de CLAHE"),
            ("BLEND_CLAHE_WEIGHT", "0.4", "Proporcion de mezcla CLAHE vs. original"),
            ("FINAL_CONTRAST_ALPHA / BETA", "0.9 / -10", "Ajuste lineal final de contraste/brillo"),
        ]
        col_x = [0.08, 0.50, 0.65]
        y = 0.78
        fig.text(col_x[0], y, "Parametro", fontsize=10.5, weight="bold")
        fig.text(col_x[1], y, "Valor actual", fontsize=10.5, weight="bold")
        fig.text(col_x[2], y, "Que controla", fontsize=10.5, weight="bold")
        fig.add_artist(plt.Line2D([0.08, 0.95], [y - 0.02, y - 0.02], color="black", lw=0.8, transform=fig.transFigure))
        y -= 0.07
        for nombre, valor, descripcion in params:
            fig.text(col_x[0], y, nombre, fontsize=10, family="monospace")
            fig.text(col_x[1], y, valor, fontsize=10, family="monospace", color="#1a5276")
            fig.text(col_x[2], y, descripcion, fontsize=9.5)
            y -= 0.075
        fig.text(0.08, 0.09,
                  "Antes, estos valores estaban fijos dentro de cada funcion (sin justificacion registrada).\n"
                  "Ahora cada parametro tiene su valor y su razon documentados en el propio archivo de config,\n"
                  "y puede barrerse en una ablacion sin tocar el codigo del pipeline.",
                  fontsize=10, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Cierre ----
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.80, "Conclusiones", ha="center", fontsize=18, weight="bold")
        cierre = (
            "- 4 defectos reales corregidos en produccion: 2 geometricos (rotacion/recorte),\n"
            "  1 de parametros (CLAHE) y 1 de aritmetica (convertScaleAbs).\n\n"
            "- 2 alternativas a Otsu investigadas (equalizeHist, sustraccion de fondo) y\n"
            "  descartadas con evidencia: no superan el 5% de casos extremos actual.\n\n"
            "- Parametros de CLAHE y Otsu ahora centralizados y justificados en\n"
            "  config/preprocessing.py, listos para ablaciones sistematicas.\n\n"
            "- Dataset completo (12,811 imagenes) regenerado con todos los fixes; el\n"
            "  recorte final solo falla en el 1.7% de los casos, todos por bajo\n"
            "  contraste global (causa conocida, sin fix automatico adoptado aun).\n\n"
            "Siguiente etapa: usar esta misma metodologia para auditar la segmentacion\n"
            "(01_train_hand_detector.py / 04_segment_images.py)."
        )
        fig.text(0.1, 0.68, cierre, va="top", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Reporte guardado en: {out_path}")


if __name__ == "__main__":
    main()
