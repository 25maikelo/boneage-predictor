"""
Genera reporte PDF comparativo de experimentos 33 y 34.
Uso: python scripts/generate_report_pdf.py [--output PATH]
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paleta de colores (profesional, sobria) ───────────────────
C33    = "#D95F02"   # naranja oscuro — Exp 33 CNN
C34    = "#1B6CA8"   # azul oscuro    — Exp 34 DenseNet121
CGRAY  = "#666666"
CLGRAY = "#AAAAAA"
CLINE  = "#DDDDDD"
CBG    = "#F7F9FC"   # fondo suave para celdas header

# ── Jerarquía tipográfica (puntos) ────────────────────────────
# Portada
FS_COVER_TITLE    = 26
FS_COVER_SUBTITLE = 14
FS_COVER_FOOT     = 10
# Secciones
FS_PAGE_TITLE     = 15   # título de página
FS_SECTION        = 12   # subtítulo dentro de página
FS_BODY           = 9    # texto corrido
FS_CAPTION        = 8    # etiquetas de ejes, pies de figura
FS_LABEL          = 7.5  # texto dentro de diagramas / tablas
FS_TINY           = 6.5  # anotaciones muy pequeñas (solo diagramas)

# ── Espaciado de páginas (fracción de la figura) ──────────────
M_TOP    = 0.93
M_BOTTOM = 0.07
M_LEFT   = 0.08
M_RIGHT  = 0.95


def load_json(path):
    with open(path) as f:
        return json.load(f)


def wrap(text, width=90):
    return "\n".join(textwrap.wrap(text, width))


def page_title(pdf, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.55, title, ha="center", va="center",
            fontsize=FS_COVER_TITLE, fontweight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.43, subtitle, ha="center", va="center",
                fontsize=FS_COVER_SUBTITLE, color=CGRAY, transform=ax.transAxes)
    ax.text(0.5, 0.12, "Bone Age Predictor — RSNA Dataset",
            ha="center", va="center", fontsize=FS_COVER_FOOT, color=CGRAY,
            transform=ax.transAxes)
    ax.plot([0.1, 0.9], [0.36, 0.36], transform=ax.transAxes,
            color=CGRAY, linewidth=0.8, clip_on=False)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def section_header(ax, text, line_y=0.88):
    ax.axis("off")
    ax.text(0, 1, text, fontsize=FS_PAGE_TITLE, fontweight="bold",
            va="top", ha="left", transform=ax.transAxes,
            color="#222222")
    ax.plot([0, 1], [line_y, line_y], transform=ax.transAxes,
            color=CLINE, linewidth=0.8, clip_on=False)


def body_text(ax, text, x=0, y=0.95, fs=FS_BODY, color="#333333"):
    ax.axis("off")
    ax.text(x, y, text, fontsize=fs, va="top", ha="left",
            transform=ax.transAxes, color=color,
            linespacing=1.6)


# ═══════════════════════════════════════════════════════════════
# PÁGINAS
# ═══════════════════════════════════════════════════════════════

def _pipeline_band(ax, steps, colors, y_box, bw, bh):
    """Dibuja una fila de cajas de pipeline con flechas."""
    n = len(steps)
    xs = np.linspace(0.07, 0.93, n)
    for i, ((num, name, out), xc, col) in enumerate(zip(steps, xs, colors)):
        rect = mpatches.FancyBboxPatch(
            (xc - bw/2, y_box - bh/2), bw, bh,
            boxstyle="round,pad=0.01", linewidth=0.8,
            edgecolor="#888", facecolor=col,
            transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(xc, y_box + bh/2 - 0.06, num, ha="center", va="center",
                fontsize=FS_LABEL, fontweight="bold", transform=ax.transAxes)
        ax.text(xc, y_box - 0.01, name, ha="center", va="center",
                fontsize=FS_TINY, transform=ax.transAxes, linespacing=1.3)
        ax.text(xc, y_box - bh/2 - 0.07, out, ha="center", va="top",
                fontsize=FS_TINY, color=CGRAY, transform=ax.transAxes)
        if i < n - 1:
            gap = xs[i+1] - xc
            ax.annotate("", xy=(xc + bw/2 + gap*0.18, y_box),
                        xytext=(xc + bw/2, y_box),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#666", lw=0.9))


def page_pipeline(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 1, figure=fig, hspace=0.15,
                  height_ratios=[0.10, 0.42, 0.38],
                  top=0.93, bottom=0.06, left=0.07, right=0.95)

    ax0 = fig.add_subplot(gs[0])
    section_header(ax0, "1. Pipeline de Procesamiento", line_y=0.0)

    # ── Banda 1: preparación del dataset (ejecución única) ──────────
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # fondo sombreado
    ax1.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.10), 1.0, 0.78, boxstyle="round,pad=0.01",
        linewidth=0.5, edgecolor="#CCCCCC", facecolor="#F5F7FA",
        transform=ax1.transAxes, clip_on=False))

    ax1.text(0.5, 0.96, "PREPARACIÓN DEL DATASET  ·  EJECUCIÓN ÚNICA",
             ha="center", va="top", fontsize=FS_CAPTION, fontweight="bold",
             color="#555555", transform=ax1.transAxes)

    prep_steps = [
        ("00", "Descarga", "13,014 PNG"),
        ("01", "Entrena\nsegmentador", "modelo h5"),
        ("02", "Recorte\nzoom", "12,811"),
        ("03", "CLAHE", "12,811"),
        ("04", "Segmenta\n4 regiones", "51,244"),
        ("05", "Análisis\ndataset", "CSV bal."),
    ]
    prep_colors = ["#AED6F1", "#AED6F1", "#A9DFBF", "#A9DFBF", "#A9DFBF", "#FAD7A0"]
    _pipeline_band(ax1, prep_steps, prep_colors, y_box=0.50, bw=0.12, bh=0.34)

    # ── Banda 2: por experimento ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.08), 1.0, 0.80, boxstyle="round,pad=0.01",
        linewidth=0.5, edgecolor="#CCCCCC", facecolor="#FEF9EF",
        transform=ax2.transAxes, clip_on=False))

    ax2.text(0.5, 0.96, "POR EXPERIMENTO  ·  PREDICTOR",
             ha="center", va="top", fontsize=FS_CAPTION, fontweight="bold",
             color="#8B5E00", transform=ax2.transAxes)

    exp_steps = [
        ("06", "Entrenamiento\nseg + fusión", "modelos .h5"),
        ("07/08", "Validación\nRSNA + MEX", "métricas"),
        ("09", "Análisis\ndesempeño", "reporte"),
    ]
    exp_colors = ["#F1948A", "#F1948A", "#D7BDE2"]
    _pipeline_band(ax2, exp_steps, exp_colors, y_box=0.50, bw=0.18, bh=0.38)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_dataset(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.38,
                  height_ratios=[0.12, 0.38, 0.50],
                  top=0.93, bottom=0.06, left=0.07, right=0.95)

    ax0 = fig.add_subplot(gs[0, :])
    section_header(ax0, "2. Dataset", line_y=0.15)

    ROW_H = 0.115   # altura entre filas de las tablas manuales
    LINE_OFF = 0.07  # separación línea↓ desde top del texto

    # Tabla de procesamiento
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.text(0, 0.97, "Procesamiento de imágenes", fontsize=FS_BODY,
             fontweight="bold", va="top", transform=ax1.transAxes)
    rows = [
        ("RSNA original",           "13,014",  False),
        ("Eliminadas (calidad)",     "13",      False),
        ("Volteadas (izquierda)",    "190",     False),
        ("Dataset raw final",        "12,811",  True),
        ("Cropped",                  "12,811",  False),
        ("Equalized (CLAHE)",        "12,811",  False),
        ("Segmented (4 regiones)",   "51,244",  False),
    ]
    for i, (label, val, bold) in enumerate(rows):
        y = 0.84 - i * ROW_H
        fw = "bold" if bold else "normal"
        fc = "#111" if bold else "#444"
        ax1.text(0.02, y, label, fontsize=FS_CAPTION, va="top",
                 transform=ax1.transAxes, fontweight=fw, color=fc)
        ax1.text(0.82, y, val, fontsize=FS_CAPTION, va="top", ha="right",
                 transform=ax1.transAxes, fontweight=fw, color=fc)
        ax1.plot([0, 0.85], [y - LINE_OFF, y - LINE_OFF],
                 transform=ax1.transAxes, color=CLINE, linewidth=0.5, clip_on=False)

    # Tabla entrenamiento
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0, 0.97, "Dataset de entrenamiento",
             fontsize=FS_BODY, fontweight="bold", va="top",
             transform=ax2.transAxes)
    rows2 = [
        ("CSV training dataset",        "12,611"),
        ("Con 4 segmentos",             "12,611"),
        ("Filtro edad (24–216 m)",       "~12,499"),
        ("Filtro ≥ 50 muestras/edad",    "11,783 · 36 edades"),
        ("Split test (20 %)",            "~2,500"),
        ("Split train+val (80 %)",       "~10,000"),
    ]
    for i, (label, val) in enumerate(rows2):
        y = 0.84 - i * ROW_H
        ax2.text(0.02, y, label, fontsize=FS_CAPTION, va="top",
                 transform=ax2.transAxes, color="#444")
        ax2.text(0.98, y, val, fontsize=FS_CAPTION, va="top", ha="right",
                 transform=ax2.transAxes, color="#444")
        ax2.plot([0, 1], [y - LINE_OFF, y - LINE_OFF],
                 transform=ax2.transAxes, color=CLINE, linewidth=0.5, clip_on=False)

    # Distribución de edades — histograma
    ax3 = fig.add_subplot(gs[2, :])
    ages_balanced = [24,36,42,48,50,54,60,69,72,78,82,84,88,94,96,100,
                     106,108,114,120,126,132,138,144,150,156,159,162,165,
                     168,174,180,186,192,204,216]
    counts_balanced = [77,106,89,71,95,89,278,193,254,55,385,274,55,492,
                       302,60,478,312,63,992,198,1084,529,657,678,1113,69,
                       682,64,892,97,418,138,172,200,72]

    ax3.bar(ages_balanced, counts_balanced, width=4, color="#4A90D9",
            alpha=0.75, edgecolor="white", linewidth=0.3)
    ax3.set_xlabel("Edad ósea (meses)", fontsize=FS_CAPTION)
    ax3.set_ylabel("N imágenes", fontsize=FS_CAPTION)
    ax3.set_title("Distribución del dataset balanceado (11,783 imágenes · 36 edades)",
                  fontsize=FS_CAPTION, pad=4)
    ax3.tick_params(labelsize=7)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_xlim(18, 222)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_architectures(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 2, figure=fig, wspace=0.12,
                  top=0.93, bottom=0.06, left=0.05, right=0.97)

    for col, (exp, label, color) in enumerate([
        ("33", "Exp 33 — CNN Simple", C33),
        ("34", "Exp 34 — DenseNet121", C34),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.5, 0.97, label, ha="center", va="top",
                fontsize=FS_SECTION, fontweight="bold", color=color)

        def box(ax, x, y, w, h, text, fc="#F0F0F0", fs=7.5, bold=False):
            rect = mpatches.FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.008", linewidth=0.8,
                edgecolor="#999", facecolor=fc)
            ax.add_patch(rect)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=fs, fontweight="bold" if bold else "normal",
                    linespacing=1.3)

        def arrow(ax, x, y1, y2):
            ax.annotate("", xy=(x, y2 + 0.01), xytext=(x, y1 - 0.01),
                        arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))

        if col == 0:  # CNN Simple
            segs = ["pinky", "middle", "thumb", "wrist"]
            xs = [0.15, 0.38, 0.62, 0.85]
            for x, seg in zip(xs, segs):
                box(ax, x, 0.85, 0.18, 0.07, f"Input\n{seg}\n(112×112×3)",
                    fc="#FEF3E2", fs=6.5)
                arrow(ax, x, 0.81, 0.74)
                box(ax, x, 0.70, 0.18, 0.10,
                    "Conv×4\nBN+ReLU\nMaxPool",
                    fc="#FDEBD0", fs=6.5)
                arrow(ax, x, 0.65, 0.57)
                box(ax, x, 0.53, 0.18, 0.07,
                    "Flatten\n12,544",
                    fc="#FAD7A0", fs=6.5)
                arrow(ax, x, 0.49, 0.43)

            # flecha hacia concatenate
            for x in xs:
                ax.annotate("", xy=(0.5, 0.36), xytext=(x, 0.39),
                            arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.6))

            box(ax, 0.5, 0.32, 0.75, 0.07,
                "Concatenate  [+ género]  →  50,176 dims",
                fc="#E8F8F5", fs=7, bold=True)
            arrow(ax, 0.5, 0.28, 0.22)
            box(ax, 0.5, 0.18, 0.55, 0.07,
                "Dense(512) → Dropout(0.5)\nDense(256) → Dropout(0.3)",
                fc="#D5F5E3", fs=7)
            arrow(ax, 0.5, 0.14, 0.08)
            box(ax, 0.5, 0.05, 0.3, 0.06,
                "Dense(1)  →  predicción (m)",
                fc="#A9DFBF", fs=7, bold=True)

        else:  # DenseNet121
            segs = ["pinky", "middle", "thumb", "wrist"]
            xs = [0.15, 0.38, 0.62, 0.85]
            for x, seg in zip(xs, segs):
                box(ax, x, 0.85, 0.18, 0.07, f"Input\n{seg}\n(112×112×3)",
                    fc="#EBF5FB", fs=6.5)
                arrow(ax, x, 0.81, 0.74)
                box(ax, x, 0.70, 0.18, 0.10,
                    "DenseNet121\n(top=False\n10 layers)",
                    fc="#D6EAF8", fs=6.5)
                arrow(ax, x, 0.65, 0.57)
                box(ax, x, 0.53, 0.18, 0.07,
                    "GAP\n1,024",
                    fc="#AED6F1", fs=6.5)
                arrow(ax, x, 0.49, 0.43)
                box(ax, x, 0.39, 0.18, 0.07,
                    "Dense(256)\n+ Dropout",
                    fc="#85C1E9", fs=6.5)
                arrow(ax, x, 0.35, 0.29)
                box(ax, x, 0.25, 0.18, 0.07,
                    "Dense(1)\nescalar",
                    fc="#5DADE2", fs=6.5)
                arrow(ax, x, 0.21, 0.17)

            # flechas verticales desde Dense(1) escalar hasta línea de converge
            for x in xs:
                arrow(ax, x, 0.21, 0.15)
            # flechas horizontales convergiendo al concatenate
            for x in xs:
                ax.annotate("", xy=(0.5, 0.11), xytext=(x, 0.15),
                            arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.6))

            box(ax, 0.5, 0.07, 0.75, 0.07,
                "Concatenate  [+ género]  →  5 dims",
                fc="#E8F8F5", fs=7, bold=True)
            arrow(ax, 0.5, 0.03, -0.02)
            box(ax, 0.5, -0.05, 0.45, 0.05,
                "Dense(128) → Dropout → Dense(1)",
                fc="#A9DFBF", fs=7, bold=True)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.12, 1.0)

    fig.suptitle("3. Arquitecturas — Modelos de Segmento y Fusión",
                 fontsize=FS_PAGE_TITLE, fontweight="bold", y=0.99)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _training_curve(ax, mae_train, val_mae, color, title, rsna_line=None):
    epochs = range(1, len(val_mae) + 1)
    ax.plot(epochs, mae_train, "s--", color=color, alpha=0.45,
            linewidth=1.2, markersize=3, label="train MAE")
    ax.plot(epochs, val_mae, "o-", color=color,
            linewidth=1.5, markersize=4, label="val MAE")
    if rsna_line is not None:
        ax.axhline(y=rsna_line, color=color, linestyle=":", linewidth=1, alpha=0.6)
        ax.text(epochs[-1] * 0.98, rsna_line + max(val_mae)*0.03,
                f"RSNA {rsna_line:.1f}", ha="right",
                fontsize=FS_TINY, color=color, alpha=0.8)
    ax.set_title(title, fontsize=FS_CAPTION, pad=4)
    ax.set_xlabel("Época", fontsize=FS_CAPTION)
    ax.set_ylabel("MAE (meses)", fontsize=FS_CAPTION)
    ax.set_ylim(0, max(max(val_mae), max(mae_train)) * 1.15)
    ax.set_xticks(list(epochs))
    ax.legend(fontsize=FS_TINY, framealpha=0.6, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_CAPTION)


def page_training_results(pdf, v33, v34, m33, m34):
    # ── Página 1: MAE segmentos + tabla de tiempos ────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 1, figure=fig, hspace=0.55,
                  height_ratios=[0.58, 0.42],
                  top=0.92, bottom=0.07, left=0.09, right=0.95)

    fig.text(0.5, 0.96, "4. Resultados de Entrenamiento",
             ha="center", fontsize=FS_PAGE_TITLE, fontweight="bold")

    # MAE segmentos
    ax1 = fig.add_subplot(gs[0])
    segs = ["pinky", "middle", "thumb", "wrist"]
    mae33 = [35.43, 27.42, 25.55, 26.73]
    mae34 = [29.17, 28.86, 23.53, 28.70]
    x = np.arange(len(segs))
    w = 0.35
    bars33 = ax1.bar(x - w/2, mae33, w, color=C33, alpha=0.85, label="Exp 33 CNN")
    bars34 = ax1.bar(x + w/2, mae34, w, color=C34, alpha=0.85, label="Exp 34 DenseNet121")
    ax1.set_xticks(x); ax1.set_xticklabels(segs, fontsize=FS_CAPTION)
    ax1.set_ylabel("MAE promedio CV (meses)", fontsize=FS_CAPTION)
    ax1.set_title("MAE por segmento — Cross-Validation (5 folds)", fontsize=FS_CAPTION, pad=5)
    ax1.set_ylim(0, 42)
    ax1.legend(fontsize=FS_CAPTION, framealpha=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=FS_CAPTION)
    for bar, val in zip(bars33, mae33):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{val:.1f}", ha="center", fontsize=FS_TINY)
    for bar, val in zip(bars34, mae34):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{val:.1f}", ha="center", fontsize=FS_TINY)

    # Tabla tiempos por script
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.text(0.5, 1.02, "Tiempos por script y experimento", ha="center",
             fontsize=FS_BODY, fontweight="bold", va="bottom",
             transform=ax2.transAxes)
    col_labels = ["Script · Fase", "Exp 33 — CNN", "Exp 34 — DenseNet121"]
    table_data = [
        ["06 · Segmento pinky  (5 folds × 15 ép.)",  "~4.65 h",  "6.36 h"],
        ["06 · Segmento middle (5 folds × 15 ép.)",  "6.47 h",   "6.29 h"],
        ["06 · Segmento thumb  (5 folds × 15 ép.)",  "7.03 h",   "8.80 h"],
        ["06 · Segmento wrist  (5 folds × 15 ép.)",  "5.92 h",   "7.43 h"],
        ["06 · Fusión (20 ép.)",                      "7.97 h",   "6.91 h"],
        ["06 · Fine-tuning (10 ép.)",                 "2.05 h",   "4.41 h"],
        ["06 · Total entrenamiento",                  "~34 h",    "~40 h"],
        ["07 · Validación RSNA",                      "~13 min",  "~15 min"],
        ["08 · Validación MEX",                       "<1 min",   "~3 min"],
        ["09 · Análisis desempeño",                   "<1 min",   "~3 min"],
    ]
    tbl = ax2.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center",
                    colWidths=[0.52, 0.24, 0.24])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS_TINY)
    tbl.scale(1, 1.35)
    total_row = 6  # índice fila "Total" (base 1 por header)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(CBG)
            cell.set_text_props(fontweight="bold")
        elif r == total_row + 1:
            cell.set_facecolor("#EAF5EA")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor(CLINE)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Página 2: históricos fusión + fine-tuning ─────────────
    fig2 = plt.figure(figsize=(11, 8.5))
    fig2.patch.set_facecolor("white")
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.52, wspace=0.35,
                   top=0.92, bottom=0.08, left=0.09, right=0.95)

    fig2.text(0.5, 0.96, "4b. Histórico de Entrenamiento — Fusión y Fine-tuning",
              ha="center", fontsize=FS_PAGE_TITLE, fontweight="bold")

    # Datos históricos
    fus_train33 = [28.63, 24.53, 23.36, 22.81, 21.94]
    fus_val33   = [39.86, 57.49, 50.12, 65.45, 60.55]
    ft_train33  = [23.28, 22.37, 21.61, 21.16, 20.81]
    ft_val33    = [37.79, 49.77, 57.32, 46.65, 49.68]

    fus_train34 = [44.25, 25.01, 23.70, 23.13, 22.44, 22.08, 20.66, 20.46, 20.62, 20.37, 19.74]
    fus_val34   = [35.49, 41.50, 15.55, 38.44, 23.92, 20.12, 14.34, 27.45, 18.29, 14.37, 14.05]
    ft_train34  = [20.25, 19.74, 19.82, 19.45, 19.45, 19.40, 19.01, 19.16, 19.12, 18.90]
    ft_val34    = [17.80, 14.09, 13.96, 13.45, 13.30, 14.00, 13.48, 13.36, 13.55, 13.57]

    ax_f33 = fig2.add_subplot(gs2[0, 0])
    _training_curve(ax_f33, fus_train33, fus_val33, C33,
                    "Exp 33 — Fusión")

    ax_f34 = fig2.add_subplot(gs2[0, 1])
    _training_curve(ax_f34, fus_train34, fus_val34, C34,
                    "Exp 34 — Fusión")

    ax_t33 = fig2.add_subplot(gs2[1, 0])
    _training_curve(ax_t33, ft_train33, ft_val33, C33,
                    "Exp 33 — Fine-tuning (modelo completo)", rsna_line=39.23)

    ax_t34 = fig2.add_subplot(gs2[1, 1])
    _training_curve(ax_t34, ft_train34, ft_val34, C34,
                    "Exp 34 — Fine-tuning (modelo completo)", rsna_line=14.57)

    pdf.savefig(fig2, bbox_inches="tight")
    plt.close(fig2)


def page_validation(pdf, v33, v34, m33, m34):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.32,
                  top=0.92, bottom=0.06, left=0.09, right=0.95)

    fig.text(0.5, 0.96, "5. Validación",
             ha="center", fontsize=FS_PAGE_TITLE, fontweight="bold")

    # ── Scatter exp 33 ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sc33 = v33["scatter"]
    ax1.scatter(sc33["trues"], sc33["preds"], alpha=0.15, s=4, color=C33)
    lim = [0, 240]
    ax1.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("Edad real (m)", fontsize=FS_CAPTION)
    ax1.set_ylabel("Predicción (m)", fontsize=FS_CAPTION)
    ax1.set_title(f"Exp 33 — RSNA  |  MAE = {v33['summary']['mae']:.1f} m",
                  fontsize=FS_CAPTION, pad=4, color=C33, fontweight="bold")
    ax1.tick_params(labelsize=FS_CAPTION)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Scatter exp 34 ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sc34 = v34["scatter"]
    ax2.scatter(sc34["trues"], sc34["preds"], alpha=0.15, s=4, color=C34)
    ax2.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("Edad real (m)", fontsize=FS_CAPTION)
    ax2.set_ylabel("Predicción (m)", fontsize=FS_CAPTION)
    ax2.set_title(f"Exp 34 — RSNA  |  MAE = {v34['summary']['mae']:.1f} m",
                  fontsize=FS_CAPTION, pad=4, color=C34, fontweight="bold")
    ax2.tick_params(labelsize=FS_CAPTION)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Scatter MEX exp 33 ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sm33 = m33["scatter"]
    ax3.scatter(sm33["trues"], sm33["preds"], alpha=0.5, s=18, color=C33)
    ax3.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
    ax3.set_xlim([0, 240]); ax3.set_ylim([0, 240])
    ax3.set_xlabel("Edad ósea real (m)", fontsize=FS_CAPTION)
    ax3.set_ylabel("Predicción (m)", fontsize=FS_CAPTION)
    ax3.set_title(f"Exp 33 — MEX  |  MAE = {m33['summary']['mae']:.1f} m",
                  fontsize=FS_CAPTION, pad=4, color=C33, fontweight="bold")
    ax3.tick_params(labelsize=FS_CAPTION)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Scatter MEX exp 34 ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sm34 = m34["scatter"]
    ax4.scatter(sm34["trues"], sm34["preds"], alpha=0.5, s=18, color=C34)
    ax4.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
    ax4.set_xlim([0, 240]); ax4.set_ylim([0, 240])
    ax4.set_xlabel("Edad ósea real (m)", fontsize=FS_CAPTION)
    ax4.set_ylabel("Predicción (m)", fontsize=FS_CAPTION)
    ax4.set_title(f"Exp 34 — MEX  |  MAE = {m34['summary']['mae']:.1f} m",
                  fontsize=FS_CAPTION, pad=4, color=C34, fontweight="bold")
    ax4.tick_params(labelsize=FS_CAPTION)
    ax4.spines[["top", "right"]].set_visible(False)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_summary(pdf, v33, v34, m33, m34):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 1, figure=fig, hspace=0.5,
                  top=0.92, bottom=0.06, left=0.08, right=0.95)

    fig.text(0.5, 0.96, "6. Resumen Comparativo",
             ha="center", fontsize=FS_PAGE_TITLE, fontweight="bold")

    # ── Tabla resumen ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    col_labels = ["Métrica", "Exp 33 — CNN", "Exp 34 — DenseNet121", "Δ"]
    table_data = [
        ["Extractor por segmento",   "CNN desde cero",      "DenseNet121",         "—"],
        ["Fusión recibe",            "4 × Flatten ~50K",    "4 escalares",         "—"],
        ["MAE CV — pinky",           "35.43 ± 2.73 m",      "29.17 ± 4.84 m",     "−17.6 %"],
        ["MAE CV — middle",          "27.42 ± 1.07 m",      "28.86 ± 4.56 m",     "+5.2 %"],
        ["MAE CV — thumb",           "25.55 ± 2.01 m",      "23.53 ± 3.31 m",     "−7.9 %"],
        ["MAE CV — wrist",           "26.73 ± 1.52 m",      "28.70 ± 5.86 m",     "+7.4 %"],
        ["val_mae fusión",           "~50 m (inestable)",   "~13.4 m",             "−73 %"],
        ["MAE RSNA (1,393 imgs)",    "39.2 m",              "14.6 m",              "−62.8 %"],
        ["MAE MEX (98 imgs)",        "35.9 m",              "17.6 m",              "−50.9 %"],
        ["Tiempo total",             "~34 h",               "~40 h",               "+17.6 %"],
    ]
    tbl = ax1.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center",
                    colWidths=[0.30, 0.25, 0.28, 0.17])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.55)
    highlight_rows = {7, 8, 9}
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DDEEFF")
            cell.set_text_props(fontweight="bold", fontsize=FS_LABEL)
        elif r - 1 in highlight_rows:
            cell.set_facecolor("#EAF5EA")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#cccccc")

    # ── Gráfica de barras comparativa ─────────────────────────
    ax2 = fig.add_subplot(gs[1])
    labels = ["CV\npinky", "CV\nmiddle", "CV\nthumb", "CV\nwrist",
              "Val\nRSNA", "Val\nMEX"]
    vals33 = [35.43, 27.42, 25.55, 26.73, 39.23, 35.93]
    vals34 = [29.17, 28.86, 23.53, 28.70, 14.57, 17.61]
    x = np.arange(len(labels))
    w = 0.35
    b33 = ax2.bar(x - w/2, vals33, w, color=C33, alpha=0.85, label="Exp 33 CNN")
    b34 = ax2.bar(x + w/2, vals34, w, color=C34, alpha=0.85, label="Exp 34 DenseNet121")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=FS_CAPTION)
    ax2.set_ylabel("MAE (meses)", fontsize=FS_CAPTION)
    ax2.set_title("Comparativa de MAE — Todas las métricas", fontsize=FS_CAPTION, pad=4)
    ax2.legend(fontsize=FS_CAPTION, framealpha=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=FS_CAPTION)
    ax2.axvline(x=3.5, color="#ccc", linewidth=0.8, linestyle="--")
    ax2.text(3.6, ax2.get_ylim()[1] * 0.95, "Validación →",
             fontsize=FS_LABEL, color=CGRAY)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(BASE, "docs", "report_exp33_34.pdf"))
    args = parser.parse_args()

    v33 = load_json(os.path.join(BASE, "experiments/33/validation/plot_data.json"))
    m33 = load_json(os.path.join(BASE, "experiments/33/mex-validation/plot_data.json"))
    v34 = load_json(os.path.join(BASE, "experiments/34/validation/plot_data.json"))
    m34 = load_json(os.path.join(BASE, "experiments/34/mex-validation/plot_data.json"))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with PdfPages(args.output) as pdf:
        page_title(pdf,
                   "Predicción de Edad Ósea",
                   "Comparativa Exp 33 (CNN Simple) vs Exp 34 (DenseNet121)")
        page_pipeline(pdf)
        page_dataset(pdf)
        page_architectures(pdf)
        page_training_results(pdf, v33, v34, m33, m34)
        page_validation(pdf, v33, v34, m33, m34)
        page_summary(pdf, v33, v34, m33, m34)

        info = pdf.infodict()
        info["Title"] = "Bone Age Predictor — Exp 33 vs 34"
        info["Author"] = "Bone Age Predictor Pipeline"

    print(f"PDF generado: {args.output}")


if __name__ == "__main__":
    main()
