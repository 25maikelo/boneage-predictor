"""
Genera docs/dataset_report.pdf con la información completa del dataset.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# ── Colores ─────────────────────────────────────────────────────────────
C_RAW      = "#2196A8"   # training raw
C_BAL      = "#4CAF77"   # training balanceado
C_RSNA     = "#E07B39"   # validación RSNA
C_MEX      = "#9C5EC9"   # validación MEX
C_ELIM     = "#AAAAAA"   # edades eliminadas
CLINE      = "#DDDDDD"
CBG        = "#F7F9FC"
CGRAY      = "#666666"

# ── Tipografía ───────────────────────────────────────────────────────────
FS_COVER_TITLE   = 26
FS_COVER_SUB     = 13
FS_PAGE_TITLE    = 15
FS_SECTION       = 11
FS_BODY          = 9
FS_CAPTION       = 8
FS_LABEL         = 7.5
FS_TINY          = 6.5

# ── Datos ────────────────────────────────────────────────────────────────

# Training raw — 12,611 imágenes, 160 edades
ages_raw = [
    1,4,6,9,10,12,13,14,15,16,17,18,20,21,24,27,28,29,30,32,33,34,36,37,
    38,39,40,42,43,45,46,48,49,50,51,52,54,55,56,57,58,60,62,63,64,65,66,
    67,69,70,72,74,75,76,77,78,80,81,82,84,86,87,88,90,91,93,94,96,100,
    101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,
    118,120,121,123,124,125,126,128,129,130,132,133,134,135,136,137,138,
    139,140,141,142,143,144,146,147,148,149,150,151,152,153,154,156,158,
    159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,176,
    177,179,180,182,183,184,186,188,189,192,194,196,197,198,200,204,206,
    210,212,214,216,222,228
]
counts_raw = [
    1,1,2,5,4,11,2,1,16,3,2,27,1,15,77,9,10,1,38,24,13,3,106,1,1,16,2,
    89,1,10,5,71,1,95,11,1,89,12,2,18,4,278,3,2,7,2,39,3,193,4,254,1,15,
    25,1,55,1,2,385,274,1,3,55,49,2,1,492,302,60,4,46,1,3,1,478,1,312,1,
    1,2,8,48,63,5,1,1,2,992,2,2,4,6,198,3,1,2,1084,1,3,27,6,4,529,3,4,2,
    1,1,657,3,8,1,2,678,1,5,42,3,1113,4,69,3,1,682,2,6,64,2,1,892,2,3,
    1,1,1,97,3,1,1,418,1,5,3,138,1,15,172,1,2,1,31,2,200,1,12,2,1,72,2,19
]

# Training balanceado — 11,783 imágenes, 36 edades
ages_bal = [24,36,42,48,50,54,60,69,72,78,82,84,88,94,96,100,106,108,114,
            120,126,132,138,144,150,156,159,162,165,168,174,180,186,192,204,216]
counts_bal = [77,106,89,71,95,89,278,193,254,55,385,274,55,492,302,60,478,
              312,63,992,198,1084,529,657,678,1113,69,682,64,892,97,418,138,172,200,72]

# RSNA validación — 1,425 imágenes, 82 edades
ages_rsna = [3,6,12,14,15,16,18,21,24,26,28,30,32,34,36,38,39,42,43,48,50,51,
             53,54,55,57,60,63,64,65,66,69,72,76,78,82,84,88,90,94,96,100,102,
             106,107,108,113,114,116,120,126,132,135,138,141,144,147,148,150,
             152,153,156,159,160,162,165,166,167,168,174,176,180,183,184,186,
             188,189,192,204,210,216,228]
counts_rsna = [1,1,1,1,3,1,4,1,11,1,2,1,1,1,13,1,4,7,1,16,6,2,1,10,1,2,31,
               1,1,1,3,21,31,6,6,39,32,11,6,59,27,6,9,54,1,37,4,7,1,114,21,
               125,2,55,1,74,2,1,78,1,8,116,11,3,80,4,1,1,105,5,1,45,2,1,16,
               1,1,23,24,5,6,4]

# MEX validación — 100 imágenes, 26 edades
ages_mex = [19,24,36,43,60,71,72,73,84,85,96,97,101,108,110,120,132,144,151,
            156,163,168,180,192,204,216]
counts_mex = [1,1,4,1,1,2,5,1,7,4,1,3,1,1,1,13,7,10,1,10,10,8,2,2,2,1]

# Edades eliminadas (< 50 muestras)
ages_elim  = [1,4,6,9,10,12,13,14,15,16,17,18,20,21,27,28,29,30,32,33,34,37,38,39,40,43,45,46,49,51,52,55,56,57,58,62,63,64,65,66,67,70,74,75,76,77,80,81,86,87,90,91,93,101,102,103,104,105,107,109,110,111,112,113,115,116,117,118,121,123,124,125,128,129,130,133,134,135,136,137,139,140,141,142,143,146,147,148,149,151,152,153,154,158,160,161,163,164,166,167,169,170,171,172,173,176,177,179,182,183,184,188,189,194,196,197,198,200,206,210,212,214,222,228]
counts_elim = [1,1,2,5,4,11,2,1,16,3,2,27,1,15,9,10,1,38,24,13,3,1,1,16,2,1,10,5,1,11,1,12,2,18,4,3,2,7,2,39,3,4,1,15,25,1,1,2,1,3,49,2,1,4,46,1,3,1,1,1,1,2,8,48,5,1,1,2,2,2,4,6,3,1,2,1,3,27,6,4,3,4,2,1,1,3,8,1,2,1,5,42,3,4,3,1,2,6,2,1,2,3,1,1,1,3,1,1,1,5,3,1,15,1,2,1,31,2,1,12,2,1,2,19]


# ── Helpers ──────────────────────────────────────────────────────────────

def page_cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1A2E44")

    fig.text(0.5, 0.62, "Dataset Report", ha="center",
             fontsize=FS_COVER_TITLE, fontweight="bold", color="white")
    fig.text(0.5, 0.52, "Bone Age Predictor — RSNA · IMSS",
             ha="center", fontsize=FS_COVER_SUB, color="#AAC8E0")
    fig.text(0.5, 0.40,
             "Procesamiento · Distribuciones de edad · Conjuntos de validación",
             ha="center", fontsize=FS_BODY, color="#88A8C0")

    # Leyenda de colores
    patches = [
        mpatches.Patch(color=C_RAW,  label="Training raw  (12,611 imgs · 160 edades)"),
        mpatches.Patch(color=C_BAL,  label="Training balanceado  (11,783 imgs · 36 edades)"),
        mpatches.Patch(color=C_RSNA, label="Validación RSNA  (1,425 imgs · 82 edades)"),
        mpatches.Patch(color=C_MEX,  label="Validación MEX  (100 imgs · 26 edades)"),
    ]
    legend = fig.legend(handles=patches, loc="lower center", ncol=2,
                        bbox_to_anchor=(0.5, 0.12),
                        fontsize=FS_CAPTION, framealpha=0.15,
                        labelcolor="white", facecolor="#2A3E54",
                        edgecolor="#446688")
    for text in legend.get_texts():
        text.set_color("white")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_tables(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.08, 0.46, 0.36],
                  hspace=0.5, wspace=0.35,
                  top=0.93, bottom=0.06, left=0.07, right=0.95)

    # Header
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    ax0.text(0, 1, "1. Procesamiento y Construcción del Dataset",
             fontsize=FS_PAGE_TITLE, fontweight="bold", va="top",
             transform=ax0.transAxes, color="#222222")
    ax0.plot([0, 1], [0.0, 0.0], transform=ax0.transAxes,
             color=CLINE, linewidth=0.8)

    ROW_H = 0.115
    LINE_OFF = 0.07

    def draw_table(ax, title, rows, col_widths=(0.72, 0.28)):
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0, 0.97, title, fontsize=FS_BODY, fontweight="bold",
                va="top", transform=ax.transAxes, color="#222")
        for i, row in enumerate(rows):
            y = 0.84 - i * ROW_H
            label, val = row[0], row[1]
            bold = len(row) > 2 and row[2]
            fw = "bold" if bold else "normal"
            fc = "#111" if bold else "#444"
            ax.text(0.02, y, label, fontsize=FS_CAPTION, va="top",
                    transform=ax.transAxes, fontweight=fw, color=fc)
            ax.text(col_widths[0] + col_widths[1] - 0.02, y, val,
                    fontsize=FS_CAPTION, va="top", ha="right",
                    transform=ax.transAxes, fontweight=fw, color=fc)
            ax.plot([0, sum(col_widths)], [y - LINE_OFF, y - LINE_OFF],
                    transform=ax.transAxes, color=CLINE,
                    linewidth=0.5, clip_on=False)

    # Tabla procesamiento de imágenes
    ax1 = fig.add_subplot(gs[1, 0])
    draw_table(ax1, "Procesamiento de imágenes", [
        ("RSNA descarga original",       "13,014"),
        ("Eliminadas (calidad)",          "−13"),
        ("Volteadas (orientación)",       "+190"),
        ("Dataset raw final",             "12,811", True),
        ("Cropped (recorte + zoom)",       "12,811"),
        ("Equalized (CLAHE)",             "12,811"),
        ("Segmented (4 regiones)",         "51,244"),
    ])

    # Tabla construcción dataset entrenamiento
    ax2 = fig.add_subplot(gs[1, 1])
    draw_table(ax2, "Dataset de entrenamiento", [
        ("CSV training dataset",          "12,611"),
        ("Con 4 segmentos completos",      "12,611"),
        ("Edades sin dato en CSV",         "−200"),
        ("Filtro < 50 imgs/edad",          "−828"),
        ("Dataset balanceado",             "11,783", True),
        ("Split test (20 %)",             "~2,357"),
        ("Split train+val (80 %)",        "~9,426"),
    ])

    # Fila inferior: CV + validación
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.text(0, 0.97, "Cross-Validation", fontsize=FS_BODY,
             fontweight="bold", va="top", transform=ax3.transAxes)
    cv_rows = [
        ("Estrategia",              "K-Fold estratificado"),
        ("Número de folds",         "5"),
        ("Train por fold",          "~7,540 imgs"),
        ("Validación por fold",     "~1,883 imgs"),
        ("Aplicado a",              "4 modelos de segmento"),
    ]
    for i, (lbl, val) in enumerate(cv_rows):
        y = 0.82 - i * ROW_H
        ax3.text(0.02, y, lbl, fontsize=FS_CAPTION, va="top",
                 transform=ax3.transAxes, color="#444")
        ax3.text(0.98, y, val, fontsize=FS_CAPTION, va="top", ha="right",
                 transform=ax3.transAxes, color="#444")
        ax3.plot([0, 1], [y - LINE_OFF, y - LINE_OFF],
                 transform=ax3.transAxes, color=CLINE,
                 linewidth=0.5, clip_on=False)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.text(0, 0.97, "Conjuntos de evaluación", fontsize=FS_BODY,
             fontweight="bold", va="top", transform=ax4.transAxes)
    val_rows = [
        ("Validación RSNA",         "1,425 imgs · 82 edades"),
        ("  — en training bal.",    "36 edades coinciden"),
        ("  — sin ejemplos train.", "46 edades"),
        ("Validación MEX (IMSS)",   "100 imgs · 26 edades"),
        ("  — edades intermedias",  "no en training bal."),
    ]
    for i, (lbl, val) in enumerate(val_rows):
        y = 0.82 - i * ROW_H
        ax4.text(0.02, y, lbl, fontsize=FS_CAPTION, va="top",
                 transform=ax4.transAxes, color="#444")
        ax4.text(0.98, y, val, fontsize=FS_CAPTION, va="top", ha="right",
                 transform=ax4.transAxes, color="#444")
        ax4.plot([0, 1], [y - LINE_OFF, y - LINE_OFF],
                 transform=ax4.transAxes, color=CLINE,
                 linewidth=0.5, clip_on=False)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_distributions(pdf):
    """Tablas de edad/cantidad para los 4 conjuntos."""

    def _page_header(fig, number, title, subtitle=""):
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.955, f"{number}. {title}", fontsize=FS_PAGE_TITLE,
                 fontweight="bold", va="top", color="#222222")
        if subtitle:
            fig.text(0.07, 0.928, subtitle, fontsize=FS_CAPTION,
                     va="top", color=CGRAY)
        fig.add_artist(plt.Line2D([0.07, 0.95], [0.918, 0.918],
                                  transform=fig.transFigure,
                                  color=CLINE, linewidth=0.8))

    def _age_table(ax, ages, counts, color, ncols=8, highlight_set=None):
        """
        Tabla multi-columna con celdas y bordes explícitos.
        ncols: número de grupos (edad | n) en paralelo.
        """
        ax.axis("off")
        n = len(ages)
        nrows = (n + ncols - 1) // ncols

        # Construir matriz de datos: 2 col reales por grupo
        real_cols = ncols * 2
        col_labels = []
        for _ in range(ncols):
            col_labels += ["Edad (m)", "n"]

        cell_data = []
        for ri in range(nrows):
            row = []
            for ci in range(ncols):
                idx = ri * ncols + ci
                if idx < n:
                    row += [str(ages[idx]), f"{counts[idx]:,}"]
                else:
                    row += ["", ""]
            cell_data.append(row)

        # Anchos que suman ~0.92 sin importar ncols
        total_w = 0.92
        age_w = total_w / ncols * 0.58
        n_w   = total_w / ncols * 0.42
        col_widths = [age_w, n_w] * ncols

        tbl = ax.table(
            cellText=cell_data,
            colLabels=col_labels,
            colWidths=col_widths,
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(FS_TINY)
        tbl.scale(1, 1.25)

        set_bal_local = highlight_set or set()

        for (r, c), cell in tbl.get_celld().items():
            cell.set_linewidth(0.4)
            if r == 0:
                # Cabecera: fondo de color
                cell.set_facecolor(color)
                cell.set_text_props(fontweight="bold", color="white")
                cell.set_edgecolor("#BBBBBB")
            else:
                idx = (r - 1) * ncols + (c // 2)
                # Separador de grupo: borde izquierdo más grueso cada 2 cols
                if c % 2 == 0 and c > 0:
                    cell.set_linewidth(0.9)
                # Fila alternada
                cell.set_facecolor("#F5F8FC" if r % 2 == 0 else "white")
                cell.set_edgecolor("#DDDDDD")

                # Resaltar edades en highlight_set
                if c % 2 == 0 and idx < n and ages[idx] in set_bal_local:
                    cell.set_text_props(fontweight="bold", color=color)
                elif c % 2 == 1 and idx < n and ages[idx] in set_bal_local:
                    cell.set_text_props(fontweight="bold", color=color)

    # ── Página 3: Training raw (160 edades) ─────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    _page_header(fig, "2", "Training raw — 12,611 imágenes",
                 "160 edades únicas · rango 1–228 meses · CSV boneage-training-dataset.csv")
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.85])
    _age_table(ax, ages_raw, counts_raw, C_RAW, ncols=8,
               highlight_set=set(ages_bal))
    fig.text(0.5, 0.015, "Edades en negrita = incluidas en el training balanceado (≥ 50 imgs)",
             ha="center", fontsize=FS_TINY, color=CGRAY)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── Página 4: Training balanceado + MEX ─────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    _page_header(fig, "3", "Training balanceado y Validación MEX")

    ax_bal = fig.add_axes([0.05, 0.08, 0.42, 0.80])
    ax_bal.text(0.0, 1.02, "Training balanceado  — 11,783 imgs · 36 edades",
                fontsize=FS_BODY, fontweight="bold", va="bottom",
                transform=ax_bal.transAxes, color="#222")
    _age_table(ax_bal, ages_bal, counts_bal, C_BAL, ncols=4)

    ax_mex = fig.add_axes([0.54, 0.08, 0.42, 0.80])
    ax_mex.text(0.0, 1.02, "Validación MEX (IMSS)  — 100 imgs · 26 edades",
                fontsize=FS_BODY, fontweight="bold", va="bottom",
                transform=ax_mex.transAxes, color="#222")
    _age_table(ax_mex, ages_mex, counts_mex, C_MEX, ncols=4)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── Página 5: Validación RSNA (82 edades) ───────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    _page_header(fig, "4", "Validación RSNA — 1,425 imágenes",
                 "82 edades únicas · rango 3–228 meses  "
                 "· Negrita = edad presente en training balanceado")
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.85])
    _age_table(ax, ages_rsna, counts_rsna, C_RSNA, ncols=8,
               highlight_set=set(ages_bal))
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_overlap(pdf):
    """Tabla de solapamiento de edades entre conjuntos."""
    set_bal  = set(ages_bal)
    set_rsna = set(ages_rsna)
    set_mex  = set(ages_mex)

    rsna_in  = sorted(set_rsna & set_bal)
    rsna_out = sorted(set_rsna - set_bal)
    mex_in   = sorted(set_mex  & set_bal)
    mex_out  = sorted(set_mex  - set_bal)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.955, "5. Solapamiento de Edades con Training Balanceado",
             fontsize=FS_PAGE_TITLE, fontweight="bold", va="top", color="#222222")
    fig.add_artist(plt.Line2D([0.07, 0.95], [0.918, 0.918],
                              transform=fig.transFigure,
                              color=CLINE, linewidth=0.8))

    def _list_block(ax, title, ages_list, color, note=""):
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0, 1.0, title, fontsize=FS_BODY, fontweight="bold",
                va="top", transform=ax.transAxes, color=color)
        if note:
            ax.text(0, 0.91, note, fontsize=FS_TINY, va="top",
                    transform=ax.transAxes, color=CGRAY)
        # muestra edades en filas de 10
        ncols, row_h = 10, 0.075
        y0 = 0.82
        for i, age in enumerate(ages_list):
            ci, ri = i % ncols, i // ncols
            x = ci / ncols + 0.005
            y = y0 - ri * row_h
            ax.text(x, y, str(age), fontsize=FS_TINY, va="top",
                    transform=ax.transAxes, color="#333")

    # Resumen numérico
    fig.text(0.07, 0.90,
             f"RSNA: {len(rsna_in)} edades en training balanceado · "
             f"{len(rsna_out)} sin ejemplos en entrenamiento   |   "
             f"MEX: {len(mex_in)} edades en training balanceado · "
             f"{len(mex_out)} sin ejemplos en entrenamiento",
             fontsize=FS_CAPTION, va="top", color=CGRAY)

    ax1 = fig.add_axes([0.05, 0.50, 0.42, 0.36])
    _list_block(ax1, f"RSNA ∩ training bal.  ({len(rsna_in)} edades)",
                rsna_in, C_RSNA)

    ax2 = fig.add_axes([0.53, 0.50, 0.42, 0.36])
    _list_block(ax2, f"RSNA fuera del training  ({len(rsna_out)} edades)",
                rsna_out, C_RSNA,
                note="El modelo nunca vio ejemplos de estas edades")

    ax3 = fig.add_axes([0.05, 0.08, 0.42, 0.36])
    _list_block(ax3, f"MEX ∩ training bal.  ({len(mex_in)} edades)",
                mex_in, C_MEX)

    ax4 = fig.add_axes([0.53, 0.08, 0.42, 0.36])
    _list_block(ax4, f"MEX fuera del training  ({len(mex_out)} edades)",
                mex_out, C_MEX)

    # Separador horizontal
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.48, 0.48],
                              transform=fig.transFigure,
                              color=CLINE, linewidth=0.6))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_eliminated(pdf):
    """Tabla de edades eliminadas del training (< 50 muestras)."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.955, "6. Edades Eliminadas del Training  (< 50 muestras · 124 edades)",
             fontsize=FS_PAGE_TITLE, fontweight="bold", va="top", color="#222222")
    fig.text(0.07, 0.928, "Total eliminado: 828 imágenes de 12,611  (6.6 %)",
             fontsize=FS_CAPTION, va="top", color=CGRAY)
    fig.add_artist(plt.Line2D([0.07, 0.95], [0.918, 0.918],
                              transform=fig.transFigure,
                              color=CLINE, linewidth=0.8))

    ax = fig.add_axes([0.05, 0.05, 0.90, 0.85])
    ax.axis("off")

    ncols = 8
    n = len(ages_elim)
    nrows = (n + ncols - 1) // ncols
    col_labels = ["Edad (m)", "n"] * ncols
    col_widths  = [0.072, 0.048] * ncols

    cell_data = []
    for ri in range(nrows):
        row = []
        for ci in range(ncols):
            idx = ri * ncols + ci
            row += ([str(ages_elim[idx]), str(counts_elim[idx])]
                    if idx < n else ["", ""])
        cell_data.append(row)

    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS_TINY)
    tbl.scale(1, 1.18)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(C_ELIM)
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_edgecolor("#999999")
        else:
            cell.set_facecolor("#F5F5F5" if r % 2 == 0 else "white")
            cell.set_edgecolor("#DDDDDD")
            if c % 2 == 0 and c > 0:
                cell.set_linewidth(0.9)


def main():
    out = "docs/dataset_report.pdf"
    with PdfPages(out) as pdf:
        page_cover(pdf)
        page_tables(pdf)
        page_distributions(pdf)
        page_overlap(pdf)
        page_eliminated(pdf)
    print(f"PDF generado: {out}")


if __name__ == "__main__":
    main()
