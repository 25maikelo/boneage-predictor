"""
Genera un PDF visual que explica el pipeline completo y los dos flujos de entrenamiento.
Uso: python scripts/generate_pipeline_pdf.py
Salida: docs/pipeline_overview.pdf
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "pipeline_overview.pdf")

# ─── Colores ───────────────────────────────────────────────────────────────────
C_PRE   = "#4A90D9"   # preprocesamiento
C_TRAIN = "#E67E22"   # entrenamiento
C_VAL   = "#27AE60"   # validación
C_DATA  = "#95A5A6"   # datos
C_CNN   = "#8E44AD"   # CNN pura
C_BB    = "#C0392B"   # backbone
C_FUS   = "#2C3E50"   # fusión
C_BG    = "#F8F9FA"
WHITE   = "#FFFFFF"
DARK    = "#2C3E50"

def box(ax, x, y, w, h, text, color, fontsize=9, text_color=WHITE, radius=0.02, alpha=1.0):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0.01,rounding_size={radius}",
                       facecolor=color, edgecolor=WHITE, linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4, wrap=True,
            multialignment="center")

def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=2)

def label(ax, x, y, text, fontsize=7.5, color="#555555", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Pipeline completo (00–09)
# ══════════════════════════════════════════════════════════════════════════════
def page_pipeline(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(5, 13.3, "Pipeline de Predicción de Edad Ósea", ha="center", va="center",
            fontsize=15, fontweight="bold", color=DARK)
    ax.text(5, 12.85, "Flujo completo — scripts 00 a 09", ha="center", va="center",
            fontsize=9, color="#666666")

    steps = [
        # (y,   color,    número, nombre,                  entrada,              salida)
        (12.1, C_PRE,  "00", "Descarga dataset",        "Kaggle API",          "data/images/raw/"),
        (11.0, C_PRE,  "01", "Entrena detector mano",   "Imágenes etiquetadas","models/hand-detector/"),
        ( 9.9, C_PRE,  "02", "Rotación y recorte",      "raw/",                "data/images/cropped/"),
        ( 8.8, C_PRE,  "03", "Ecualización CLAHE",      "cropped/",            "data/images/equalized/"),
        ( 7.7, C_PRE,  "04", "Segmentación en 4 zonas", "equalized/ + modelo", "segmented/{pinky,middle,thumb,wrist}/"),
        ( 6.6, C_TRAIN,"05", "Análisis del dataset",    "CSVs + segmented/",   "balanced_dataset.csv"),
        ( 5.5, C_TRAIN,"06", "Entrenamiento",           "segmented/ + config", "models/ (segmentos + fusión)"),
        ( 4.4, C_VAL,  "07", "Validación RSNA",         "data/validation/",    "scatter, MAE, saliencias"),
        ( 3.3, C_VAL,  "08", "Validación mexicana",     "data/mex-validation/","scatter, MAE, saliencias"),
        ( 2.2, C_VAL,  "09", "Análisis desempeño",      "models/ + segmented/","tabla comparativa, saliencias"),
    ]

    for (y, color, num, name, inp, out) in steps:
        # Número
        circle = plt.Circle((1.3, y), 0.32, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(1.3, y, num, ha="center", va="center", fontsize=9,
                fontweight="bold", color=WHITE, zorder=4)
        # Caja principal
        box(ax, 4.0, y, 4.4, 0.62, name, color, fontsize=10)
        # Entrada / salida
        label(ax, 6.55, y + 0.18, f"← {inp}", fontsize=6.5, ha="left", color="#444")
        label(ax, 6.55, y - 0.18, f"→ {out}", fontsize=6.5, ha="left", color="#444")

    # Flechas entre steps
    for i in range(len(steps) - 1):
        y1, y2 = steps[i][0] - 0.31, steps[i+1][0] + 0.31
        arrow(ax, 4.0, y1, 4.0, y2, color="#AAAAAA", lw=1.2)

    # Leyenda
    for label_txt, color, xpos in [("Preprocesamiento", C_PRE, 1.8),
                                    ("Entrenamiento",    C_TRAIN, 4.2),
                                    ("Validación",       C_VAL,   6.6)]:
        b = FancyBboxPatch((xpos - 0.9, 0.9), 1.8, 0.45,
                           boxstyle="round,pad=0.04", facecolor=color,
                           edgecolor=WHITE, linewidth=1, alpha=0.9, zorder=3)
        ax.add_patch(b)
        ax.text(xpos, 1.12, label_txt, ha="center", va="center",
                fontsize=7.5, color=WHITE, fontweight="bold", zorder=4)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — Script 06: Flujo CNN vs Backbone
# ══════════════════════════════════════════════════════════════════════════════
def page_training_flow(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(5, 13.4, "Script 06 — Entrenamiento", ha="center", fontsize=15,
            fontweight="bold", color=DARK)
    ax.text(5, 12.95, "CNN pura (exp 31)  vs  DenseNet121 (exp 32)", ha="center",
            fontsize=9, color="#666")

    # ── Columna izquierda: CNN ──────────────────────────────────────────────
    cx = 2.5
    ax.text(cx, 12.5, "CNN pura", ha="center", fontsize=11,
            fontweight="bold", color=C_CNN)
    ax.axvline(5, color="#CCCCCC", lw=1.5, ymin=0.05, ymax=0.93, zorder=1)

    # ── Columna derecha: Backbone ───────────────────────────────────────────
    bx = 7.5
    ax.text(bx, 12.5, "DenseNet121", ha="center", fontsize=11,
            fontweight="bold", color=C_BB)

    # ── Pasos compartidos (encima) ──────────────────────────────────────────
    shared_top = [
        (12.0, "CSV balanceado\n+ filtro de edades"),
        (11.1, "K-Fold CV (5 folds)\npartición train / val"),
        (10.2, "Generador de imágenes\n+ género (USE_GENDER=True)"),
    ]
    for y, txt in shared_top:
        box(ax, 5.0, y, 6.5, 0.62, txt, C_DATA, fontsize=8.5,
            text_color=DARK, alpha=0.85)

    for i in range(len(shared_top) - 1):
        y1 = shared_top[i][0] - 0.31
        y2 = shared_top[i+1][0] + 0.31
        arrow(ax, 5.0, y1, 5.0, y2, color="#AAAAAA")

    arrow(ax, 5.0, shared_top[-1][0] - 0.31, cx, 9.04, color="#AAAAAA")
    arrow(ax, 5.0, shared_top[-1][0] - 0.31, bx, 9.04, color="#AAAAAA")

    # ── Sección divergente ──────────────────────────────────────────────────
    diverge = [
        # (y,  txt_cnn,                              txt_bb)
        (8.7,  "Conv2D × 4\n(sin pesos previos)",    "DenseNet121\n(sin pesos previos)"),
        (7.7,  "Flatten → Dense(512)\n→ salida",     "GAP → Dense(256)\n→ salida"),
        (6.7,  "EarlyStopping + ReduceLR\nguarda mejor fold", "EarlyStopping + ReduceLR\nguarda mejor fold"),
        (5.75, "Extrae capa\nflatten_features",       "Usa modelo completo\n(predictions)"),
        (4.8,  "Concat features × 4\n+ género\n→ Dense(512)→Dense(256)",
                                                      "Concat outputs × 4\n+ género\n→ Dense(128)"),
        (3.8,  "Fine-tuning fusión\n(lr × 0.1)",     "Fine-tuning fusión\n(lr × 0.1)"),
    ]

    section_labels = [
        (8.7, "Modelo de segmento"),
        (6.7, "Entrenamiento por fold"),
        (5.75,"Fusión — carga segmentos"),
        (4.8, "Fusión — head"),
        (3.8, "Fine-tuning"),
    ]
    for y, lbl in section_labels:
        ax.text(5.0, y + 0.5, lbl, ha="center", va="center", fontsize=7,
                color="#888", style="italic")

    for i, (y, tc, tb) in enumerate(diverge):
        box(ax, cx, y, 3.8, 0.65, tc, C_CNN, fontsize=8)
        box(ax, bx, y, 3.8, 0.65, tb, C_BB,  fontsize=8)
        if i < len(diverge) - 1:
            y_next = diverge[i+1][0]
            arrow(ax, cx, y - 0.33, cx, y_next + 0.33, color=C_CNN)
            arrow(ax, bx, y - 0.33, bx, y_next + 0.33, color=C_BB)

    # ── Reconvergen en salida ───────────────────────────────────────────────
    y_end = 2.9
    arrow(ax, cx, diverge[-1][0] - 0.33, 5.0, y_end + 0.31, color="#AAAAAA")
    arrow(ax, bx, diverge[-1][0] - 0.33, 5.0, y_end + 0.31, color="#AAAAAA")
    box(ax, 5.0, y_end, 5.5, 0.55,
        "fusion_model  →  predicción edad ósea (meses)", C_FUS, fontsize=9)

    # ── Nota diferencias ───────────────────────────────────────────────────
    ax.text(5.0, 2.1, "Diferencias entre flujos", ha="center", fontsize=8.5,
            fontweight="bold", color=DARK)
    diffs = [
        ("Parámetros",   "~500 K",              "~7 M (DenseNet121)"),
        ("Fusión",       "Extrae features intermedios", "Usa outputs finales"),
        ("Head fusión",  "Dense(512)→Dense(256)","Dense(128)"),
        ("Todo lo demás","─── idéntico ───","─── idéntico ───"),
    ]
    cols = [1.0, 3.5, 6.5, 9.0]
    ax.text(cols[0], 1.75, "Aspecto",   ha="center", fontsize=8, fontweight="bold", color=DARK)
    ax.text(cols[1], 1.75, "CNN pura",  ha="center", fontsize=8, fontweight="bold", color=C_CNN)
    ax.text(cols[2], 1.75, "DenseNet",  ha="center", fontsize=8, fontweight="bold", color=C_BB)
    for j, (asp, vc, vb) in enumerate(diffs):
        yrow = 1.4 - j * 0.28
        bg = "#EEEEEE" if j % 2 == 0 else C_BG
        ax.add_patch(mpatches.Rectangle((0.05, yrow - 0.12), 9.9, 0.27,
                                        facecolor=bg, edgecolor="none", zorder=1))
        ax.text(cols[0], yrow, asp,  ha="center", fontsize=7.5, color=DARK)
        ax.text(cols[1], yrow, vc,   ha="center", fontsize=7.5, color=C_CNN)
        ax.text(cols[2], yrow, vb,   ha="center", fontsize=7.5, color=C_BB)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Scripts 07, 08, 09
# ══════════════════════════════════════════════════════════════════════════════
def page_validation(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(5, 13.4, "Scripts 07, 08 y 09 — Validación y Análisis",
            ha="center", fontsize=14, fontweight="bold", color=DARK)

    # Entrada compartida
    box(ax, 5, 12.5, 7, 0.6,
        "Modelos entrenados  (segmentos + fusión + detector de mano)",
        C_TRAIN, fontsize=9)

    arrow(ax, 3.0, 12.2, 2.0, 11.55, color="#AAAAAA")
    arrow(ax, 5.0, 12.2, 5.0, 11.55, color="#AAAAAA")
    arrow(ax, 7.0, 12.2, 8.0, 11.55, color="#AAAAAA")

    scripts = [
        (2.0, C_VAL, "07\nValidación RSNA",
         ["Dataset RSNA estándar", "Preprocesa imagen", "Predice edad ósea"],
         ["Scatter real vs pred", "MAE / tiempos", "Saliencias (GradCAM)", "Histograma edades"]),
        (5.0, C_VAL, "08\nValidación mexicana",
         ["Dataset pacientes MX", "Preprocesa imagen", "Predice edad ósea"],
         ["Scatter real vs pred", "MAE / tiempos", "Saliencias (GradCAM)", "Histograma edades"]),
        (8.0, C_VAL, "09\nAnálisis desempeño",
         ["Dataset de entrenamiento", "Muestras de cada segmento"],
         ["Tabla comparativa Loss/MAE", "Saliencias por segmento"]),
    ]

    for (x, color, title, inputs, outputs) in scripts:
        box(ax, x, 11.1, 2.8, 0.65, title, color, fontsize=9.5)

        # Entradas
        ax.text(x, 10.5, "Entradas", ha="center", fontsize=7.5,
                fontweight="bold", color="#555")
        for k, inp in enumerate(inputs):
            yi = 10.18 - k * 0.32
            label(ax, x, yi, f"• {inp}", fontsize=7.5)

        arrow(ax, x, 10.18 - len(inputs)*0.32 + 0.05,
              x, 10.18 - len(inputs)*0.32 - 0.25, color="#AAAAAA")

        # Salidas
        y_out_start = 10.18 - len(inputs)*0.32 - 0.5
        ax.text(x, y_out_start, "Salidas", ha="center", fontsize=7.5,
                fontweight="bold", color="#555")
        for k, out in enumerate(outputs):
            yo = y_out_start - 0.3 - k * 0.32
            box(ax, x, yo, 2.6, 0.26, out, color, fontsize=7, alpha=0.7)

    # Flujo de inferencia (shared)
    ax.text(5.0, 5.5, "Flujo de inferencia (scripts 07 y 08)", ha="center",
            fontsize=10, fontweight="bold", color=DARK)

    inference = [
        (4.8, C_DATA,  "Imagen RX de entrada"),
        (4.0, C_PRE,   "Detector mano → recorte + CLAHE"),
        (3.2, C_PRE,   "Segmentación → pinky / middle / thumb / wrist"),
        (2.4, C_TRAIN, "Modelos CNN independientes → 4 predicciones"),
        (1.6, C_TRAIN, "Modelo fusión + género → predicción final"),
        (0.8, C_VAL,   "Edad ósea predicha (meses)"),
    ]
    for i, (y, color, txt) in enumerate(inference):
        box(ax, 5.0, y, 6.5, 0.52, txt, color, fontsize=8.5)
        if i < len(inference) - 1:
            arrow(ax, 5.0, y - 0.26, 5.0, inference[i+1][0] + 0.26, color="#AAAAAA")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
with PdfPages(OUT_PATH) as pdf:
    page_pipeline(pdf)
    page_training_flow(pdf)
    page_validation(pdf)

    d = pdf.infodict()
    d["Title"]   = "Bone Age Predictor — Pipeline Overview"
    d["Author"]  = "boneage-predictor"
    d["Subject"] = "Flujo de scripts y comparación CNN vs Backbone"

print(f"PDF generado: {OUT_PATH}")
