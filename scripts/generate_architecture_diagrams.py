"""
Genera diagramas visuales de las 4 arquitecturas en docs/arquitecturas.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Paleta ───────────────────────────────────────────────────────────
C_IMG    = "#AED6F1"   # entrada imagen
C_CONV   = "#A9DFBF"   # capas convolucionales
C_FEAT   = "#F9E79F"   # extracción de características
C_FUSE   = "#F1948A"   # fusión
C_OUT    = "#D7BDE2"   # salida
C_GENDER = "#FAD7A0"   # género
C_HEADER = "#2C3E50"   # fondo cabecera
CGRAY    = "#555555"
CLINE    = "#CCCCCC"

FS_TITLE  = 13
FS_HEAD   = 10
FS_BOX    = 7.5
FS_LABEL  = 6.5
FS_TINY   = 6.0


def _box(ax, x, y, w, h, text, fc, ec="#999", fs=FS_BOX,
         bold=False, text_color="black", radius=0.015):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.008,rounding_size={radius}",
                          facecolor=fc, edgecolor=ec, linewidth=0.7,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=text_color, transform=ax.transAxes, linespacing=1.3,
            wrap=False)


def _arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=CGRAY,
                                lw=0.8, mutation_scale=8))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.01, my, label, ha="left", va="center",
                fontsize=FS_TINY, color=CGRAY, transform=ax.transAxes)


def _harrow(ax, x0, x1, y, label=""):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=CGRAY,
                                lw=0.8, mutation_scale=8))
    if label:
        ax.text((x0 + x1)/2, y + 0.025, label, ha="center", va="bottom",
                fontsize=FS_TINY, color=CGRAY, transform=ax.transAxes)


def _section(ax, title, color=C_HEADER):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0.7), 1.0, 0.3,
                                boxstyle="round,pad=0.01",
                                facecolor=color, edgecolor="none",
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.85, title, ha="center", va="center",
            fontsize=FS_HEAD, fontweight="bold", color="white",
            transform=ax.transAxes)


# ═══════════════════════════════════════════════════════════════════
# PÁGINA 1 — BACKBONE
# ═══════════════════════════════════════════════════════════════════

def page_backbone(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.96, "Arquitectura 1 — Backbone  (MODEL_TYPE = \"backbone\")",
             ha="center", fontsize=FS_TITLE, fontweight="bold", color=C_HEADER)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.935, 0.935],
                              transform=fig.transFigure, color=CLINE, lw=0.8))

    # ── Segmento (izquierda) ──────────────────────────────────────
    ax1 = fig.add_axes([0.04, 0.08, 0.40, 0.83])
    ax1.axis("off"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.text(0.5, 0.97, "Modelo de Segmento", ha="center", va="top",
             fontsize=FS_HEAD, fontweight="bold", color=C_HEADER)

    bw, bh = 0.70, 0.072
    cx = 0.50
    layers = [
        (0.87, "(H, W, 3)",                          C_IMG),
        (0.75, "Backbone preentrenado\n(DenseNet121 / VGG16 / ResNet50)",  C_CONV),
        (0.63, "GlobalAveragePooling2D",              C_CONV),
        (0.52, "Concatenate(género)  ←  USE_GENDER", C_GENDER),
        (0.41, "Dense(256, relu)\n name=backbone_features",  C_FEAT),
        (0.30, "Dropout(0.5)",                        C_FEAT),
        (0.19, "Dense(1, linear)\npredicción (meses)", C_OUT),
    ]
    for y, txt, fc in layers:
        _box(ax1, cx, y, bw, bh, txt, fc)
    for i in range(len(layers) - 1):
        _arrow(ax1, cx, layers[i][0] - bh/2, cx, layers[i+1][0] + bh/2)

    # ── Fusión (derecha) ──────────────────────────────────────────
    ax2 = fig.add_axes([0.52, 0.08, 0.45, 0.83])
    ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.5, 0.97, "Modelo de Fusión  (4 escalares)", ha="center",
             va="top", fontsize=FS_HEAD, fontweight="bold", color=C_HEADER)

    segs = ["pinky", "middle", "thumb", "wrist"]
    ys_in = [0.82, 0.67, 0.52, 0.37]
    for seg, y in zip(segs, ys_in):
        _box(ax2, 0.18, y, 0.28, 0.08, f"Seg. {seg}\n(escalar)", C_IMG)
        _harrow(ax2, 0.32, 0.48, y)

    # líneas de convergencia
    for y in ys_in:
        ax2.plot([0.48, 0.55], [y, 0.595],
                 transform=ax2.transAxes, color=CGRAY, lw=0.7)

    _box(ax2, 0.62, 0.62, 0.20, 0.08, "Concatenate\n[4 + 1 género]", C_GENDER)
    _arrow(ax2, 0.62, 0.58, 0.62, 0.49)
    _box(ax2, 0.62, 0.45, 0.20, 0.08, "Dense(128, relu)", C_FUSE)
    _arrow(ax2, 0.62, 0.41, 0.62, 0.32)
    _box(ax2, 0.62, 0.28, 0.20, 0.08, "Dropout(0.5)", C_FUSE)
    _arrow(ax2, 0.62, 0.24, 0.62, 0.15)
    _box(ax2, 0.62, 0.11, 0.20, 0.08, "Dense(1, linear)\npredicción final", C_OUT, bold=True)

    # Separador vertical
    fig.add_artist(plt.Line2D([0.50, 0.50], [0.08, 0.93],
                              transform=fig.transFigure,
                              color=CLINE, lw=0.8, linestyle="--"))

    # Leyenda
    patches = [mpatches.Patch(color=C_IMG, label="Entrada imagen"),
               mpatches.Patch(color=C_CONV, label="Backbone / Pooling"),
               mpatches.Patch(color=C_FEAT, label="Extracción features"),
               mpatches.Patch(color=C_GENDER, label="Entrada género"),
               mpatches.Patch(color=C_FUSE, label="Fusión"),
               mpatches.Patch(color=C_OUT, label="Salida")]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.01), fontsize=FS_TINY,
               framealpha=0.5, edgecolor=CLINE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PÁGINA 2 — SIMPLE CNN
# ═══════════════════════════════════════════════════════════════════

def page_simple_cnn(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.96, "Arquitectura 2 — CNN Simple  (MODEL_TYPE = \"simple_cnn\")",
             ha="center", fontsize=FS_TITLE, fontweight="bold", color=C_HEADER)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.935, 0.935],
                              transform=fig.transFigure, color=CLINE, lw=0.8))

    # ── Segmento (izquierda) ──────────────────────────────────────
    ax1 = fig.add_axes([0.04, 0.08, 0.38, 0.83])
    ax1.axis("off"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.text(0.5, 0.97, "Modelo de Segmento", ha="center", va="top",
             fontsize=FS_HEAD, fontweight="bold", color=C_HEADER)

    cx, bw, bh = 0.50, 0.78, 0.068
    layers = [
        (0.89, "(H, W, 3)",                                    C_IMG),
        (0.79, "Conv2D(32)→BN→ReLU→MaxPool  [56×56×32]",      C_CONV),
        (0.70, "Conv2D(64)→BN→ReLU→MaxPool  [28×28×64]",      C_CONV),
        (0.61, "Conv2D(128)→BN→ReLU→MaxPool [14×14×128]",     C_CONV),
        (0.52, "Conv2D(256)→BN→ReLU→MaxPool [7×7×256]",       C_CONV),
        (0.42, "Flatten  name=flatten_features  [12,544]",     C_FEAT),
        (0.33, "Concatenate(género)  ←  USE_GENDER",           C_GENDER),
        (0.24, "Dense(512, relu) → Dropout(0.3)",              C_FEAT),
        (0.14, "Dense(1, linear)  predicción (meses)",         C_OUT),
    ]
    for y, txt, fc in layers:
        _box(ax1, cx, y, bw, bh, txt, fc)
    for i in range(len(layers) - 1):
        _arrow(ax1, cx, layers[i][0] - bh/2, cx, layers[i+1][0] + bh/2)

    # ── Fusión (derecha) ──────────────────────────────────────────
    ax2 = fig.add_axes([0.47, 0.08, 0.50, 0.83])
    ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.5, 0.97, "Modelo de Fusión  (4 vectores flatten ~12K-dim)",
             ha="center", va="top", fontsize=FS_HEAD, fontweight="bold", color=C_HEADER)

    segs = ["pinky", "middle", "thumb", "wrist"]
    ys_in = [0.83, 0.68, 0.53, 0.38]
    for seg, y in zip(segs, ys_in):
        _box(ax2, 0.18, y, 0.28, 0.08,
             f"{seg}\n flatten [12,544]", C_FEAT)
        ax2.plot([0.32, 0.50], [y, 0.595],
                 transform=ax2.transAxes, color=CGRAY, lw=0.7)

    _box(ax2, 0.62, 0.62, 0.26, 0.08,
         "Concatenate  [50,176]\n+ género → [50,177]", C_GENDER)
    _arrow(ax2, 0.62, 0.58, 0.62, 0.49)
    _box(ax2, 0.62, 0.45, 0.22, 0.07, "Dense(512, relu)", C_FUSE)
    _arrow(ax2, 0.62, 0.415, 0.62, 0.345)
    _box(ax2, 0.62, 0.31, 0.22, 0.07, "Dropout(0.5)", C_FUSE)
    _arrow(ax2, 0.62, 0.275, 0.62, 0.205)
    _box(ax2, 0.62, 0.17, 0.22, 0.07, "Dense(256, relu)", C_FUSE)
    _arrow(ax2, 0.62, 0.135, 0.62, 0.065)
    _box(ax2, 0.62, 0.03, 0.22, 0.07, "Dense(1, linear)\npredicción final",
         C_OUT, bold=True)

    fig.add_artist(plt.Line2D([0.45, 0.45], [0.08, 0.93],
                              transform=fig.transFigure,
                              color=CLINE, lw=0.8, linestyle="--"))

    patches = [mpatches.Patch(color=C_IMG, label="Entrada imagen"),
               mpatches.Patch(color=C_CONV, label="Conv→BN→ReLU→Pool"),
               mpatches.Patch(color=C_FEAT, label="Flatten / Dense"),
               mpatches.Patch(color=C_GENDER, label="Entrada género"),
               mpatches.Patch(color=C_FUSE, label="Fusión"),
               mpatches.Patch(color=C_OUT, label="Salida")]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.01), fontsize=FS_TINY,
               framealpha=0.5, edgecolor=CLINE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PÁGINA 3 — BACKBONE VECTORS
# ═══════════════════════════════════════════════════════════════════

def page_backbone_vectors(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.96,
             "Arquitectura 3 — Backbone Vectors  (MODEL_TYPE = \"backbone_vectors\")",
             ha="center", fontsize=FS_TITLE, fontweight="bold", color=C_HEADER)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.935, 0.935],
                              transform=fig.transFigure, color=CLINE, lw=0.8))

    # ── Segmento (izquierda) ──────────────────────────────────────
    ax1 = fig.add_axes([0.04, 0.08, 0.40, 0.83])
    ax1.axis("off"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.text(0.5, 0.97, "Modelo de Segmento", ha="center", va="top",
             fontsize=FS_HEAD, fontweight="bold", color=C_HEADER)

    cx, bw, bh = 0.50, 0.78, 0.072
    layers = [
        (0.87, "(H, W, 3)",                                           C_IMG),
        (0.76, "Backbone preentrenado\n(DenseNet121 / VGG16 / ResNet50)", C_CONV),
        (0.65, "GlobalAveragePooling2D",                              C_CONV),
        (0.54, "Concatenate(género)  ←  USE_GENDER",                  C_GENDER),
        (0.43, "Dense(256, relu)\n★ name=\"backbone_features\"",      C_FEAT),
        (0.32, "Dropout(0.5)",                                        C_FEAT),
        (0.21, "Dense(1, linear)\n(usado solo en fase de segmentos)", C_OUT),
    ]
    for y, txt, fc in layers:
        bold = "backbone_features" in txt
        _box(ax1, cx, y, bw, bh, txt, fc, bold=bold)
    for i in range(len(layers) - 1):
        _arrow(ax1, cx, layers[i][0] - bh/2, cx, layers[i+1][0] + bh/2)

    # Nota sobre extracción
    ax1.text(0.5, 0.09,
             "★ En fusión se extrae hasta aquí\n"
             "   (Dense(1) no se usa)",
             ha="center", va="center", fontsize=FS_TINY,
             color="#AA6600", style="italic", transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E7", ec="#DDBB88", lw=0.6))

    # ── Fusión (derecha) ──────────────────────────────────────────
    ax2 = fig.add_axes([0.50, 0.08, 0.47, 0.83])
    ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.5, 0.97, "Modelo de Fusión  (4 vectores 256-dim)",
             ha="center", va="top", fontsize=FS_HEAD, fontweight="bold", color=C_HEADER)

    segs = ["pinky", "middle", "thumb", "wrist"]
    ys_in = [0.83, 0.68, 0.53, 0.38]
    for seg, y in zip(segs, ys_in):
        _box(ax2, 0.18, y, 0.28, 0.08,
             f"{seg}\n backbone_features [256]", C_FEAT)
        ax2.plot([0.32, 0.50], [y, 0.595],
                 transform=ax2.transAxes, color=CGRAY, lw=0.7)

    _box(ax2, 0.63, 0.62, 0.26, 0.08,
         "Concatenate  [1,024]\n+ género → [1,025]", C_GENDER)
    _arrow(ax2, 0.63, 0.58, 0.63, 0.49)
    _box(ax2, 0.63, 0.45, 0.22, 0.07, "Dense(512, relu)", C_FUSE)
    _arrow(ax2, 0.63, 0.415, 0.63, 0.345)
    _box(ax2, 0.63, 0.31, 0.22, 0.07, "Dropout(0.5)", C_FUSE)
    _arrow(ax2, 0.63, 0.275, 0.63, 0.205)
    _box(ax2, 0.63, 0.17, 0.22, 0.07, "Dense(256, relu)", C_FUSE)
    _arrow(ax2, 0.63, 0.135, 0.63, 0.065)
    _box(ax2, 0.63, 0.03, 0.22, 0.07, "Dense(1, linear)\npredicción final",
         C_OUT, bold=True)

    # Nota FREEZE
    ax2.text(0.50, 0.20,
             "FREEZE_EXTRACTORS=True → pesos backbone fijos\n"
             "FREEZE_EXTRACTORS=False → se entrenan con fusión",
             ha="center", va="center", fontsize=FS_TINY,
             color="#555", style="italic", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", fc="#F0F0F0", ec=CLINE, lw=0.5))

    fig.add_artist(plt.Line2D([0.48, 0.48], [0.08, 0.93],
                              transform=fig.transFigure,
                              color=CLINE, lw=0.8, linestyle="--"))

    patches = [mpatches.Patch(color=C_IMG, label="Entrada imagen"),
               mpatches.Patch(color=C_CONV, label="Backbone / Pooling"),
               mpatches.Patch(color=C_FEAT, label="backbone_features [256]"),
               mpatches.Patch(color=C_GENDER, label="Entrada género"),
               mpatches.Patch(color=C_FUSE, label="Fusión"),
               mpatches.Patch(color=C_OUT, label="Salida")]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.01), fontsize=FS_TINY,
               framealpha=0.5, edgecolor=CLINE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PÁGINA 4 — UNIFIED CNN
# ═══════════════════════════════════════════════════════════════════

def page_unified_cnn(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.96,
             "Arquitectura 4 — Unified CNN  (MODEL_TYPE = \"unified_cnn\")",
             ha="center", fontsize=FS_TITLE, fontweight="bold", color=C_HEADER)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.935, 0.935],
                              transform=fig.transFigure, color=CLINE, lw=0.8))

    ax = fig.add_axes([0.04, 0.06, 0.92, 0.87])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    note = ("Entrenamiento end-to-end en una sola fase  ·  "
            "Sin pre-entrenamiento de segmentos  ·  "
            "Gradiente global actualiza todas las ramas simultáneamente")
    ax.text(0.5, 0.96, note, ha="center", va="top", fontsize=FS_TINY,
            color="#AA6600", style="italic", transform=ax.transAxes)

    # 4 ramas CNN
    segs  = ["pinky", "middle", "thumb", "wrist"]
    xs    = [0.10, 0.28, 0.46, 0.64]
    bw, bh = 0.15, 0.062

    conv_layers = [
        "Conv2D(32)→BN→\nReLU→MaxPool",
        "Conv2D(64)→BN→\nReLU→MaxPool",
        "Conv2D(128)→BN→\nReLU→MaxPool",
        "Conv2D(256)→BN→\nReLU→MaxPool",
    ]
    ys_conv = [0.86, 0.76, 0.66, 0.56]
    y_flat  = 0.46
    y_cat   = 0.33
    y_g     = 0.33
    y_d1    = 0.22
    y_d2    = 0.12

    for xi, seg in zip(xs, segs):
        # header rama
        _box(ax, xi, 0.93, bw, 0.055,
             f"input_{seg}\n(H,W,3)", C_IMG, fs=FS_TINY)
        # capas conv
        for yi, txt in zip(ys_conv, conv_layers):
            _box(ax, xi, yi, bw, bh, txt, C_CONV, fs=FS_TINY)
        # flechas verticales conv
        _arrow(ax, xi, 0.905, xi, 0.893)
        for i in range(len(ys_conv) - 1):
            _arrow(ax, xi, ys_conv[i] - bh/2, xi, ys_conv[i+1] + bh/2)
        # flatten
        _box(ax, xi, y_flat, bw, bh, "Flatten\n[12,544]", C_FEAT, fs=FS_TINY)
        _arrow(ax, xi, ys_conv[-1] - bh/2, xi, y_flat + bh/2)
        # línea convergencia al concatenate
        ax.plot([xi, 0.79], [y_flat - bh/2, y_cat + 0.03],
                transform=ax.transAxes, color=CGRAY, lw=0.7)

    # Concatenate + gender
    _box(ax, 0.79, y_cat, 0.18, 0.065,
         "Concatenate [50,176]\n+ género → [50,177]", C_GENDER, fs=FS_TINY)
    _arrow(ax, 0.79, y_cat - 0.033, 0.79, y_d1 + 0.033)

    # Dense layers
    _box(ax, 0.79, y_d1, 0.16, 0.058,
         "Dense(512, relu)\nDropout(0.3)", C_FUSE, fs=FS_TINY)
    _arrow(ax, 0.79, y_d1 - 0.029, 0.79, y_d2 + 0.029)
    _box(ax, 0.79, y_d2, 0.16, 0.058,
         "Dense(256, relu)\nDropout(0.3)", C_FUSE, fs=FS_TINY)
    _arrow(ax, 0.79, y_d2 - 0.029, 0.79, 0.032)
    _box(ax, 0.79, 0.02, 0.16, 0.058,
         "Dense(1, linear)\npredicción final", C_OUT, bold=True, fs=FS_TINY)

    patches = [mpatches.Patch(color=C_IMG, label="Entrada imagen"),
               mpatches.Patch(color=C_CONV, label="Conv→BN→ReLU→Pool"),
               mpatches.Patch(color=C_FEAT, label="Flatten"),
               mpatches.Patch(color=C_GENDER, label="Concatenate + género"),
               mpatches.Patch(color=C_FUSE, label="Fusión Dense"),
               mpatches.Patch(color=C_OUT, label="Salida")]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.005), fontsize=FS_TINY,
               framealpha=0.5, edgecolor=CLINE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PÁGINA 5 — COMPARATIVA
# ═══════════════════════════════════════════════════════════════════

def page_comparison(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.96, "Comparativa de Arquitecturas",
             ha="center", fontsize=FS_TITLE, fontweight="bold", color=C_HEADER)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.935, 0.935],
                              transform=fig.transFigure, color=CLINE, lw=0.8))

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.82])
    ax.axis("off")

    headers = ["Aspecto", "backbone", "simple_cnn", "backbone_vectors", "unified_cnn"]
    rows = [
        ["Info. a fusión",          "4 escalares",       "4 × 12K flatten",   "4 × 256 vectores",   "— (end-to-end)"],
        ["Fases de entrenamiento",  "3 (seg+fus+ft)",    "3 (seg+fus+ft)",    "3 (seg+fus+ft)",     "1 (todo junto)"],
        ["Pesos iniciales",         "ImageNet opcional", "Desde cero",        "ImageNet opcional",  "Desde cero"],
        ["Paráms. por segmento",    "~7M (DenseNet)",    "~3–5M",             "~7M (DenseNet)",     "~3–5M"],
        ["FREEZE_EXTRACTORS",       "N/A",               "Sí",                "Sí",                 "N/A"],
        ["Riesgo sobreajuste",      "Medio",             "Alto (50K→fusión)", "Bajo (256→fusión)",  "Alto (sin pretrain)"],
        ["Experimentos",            "34, 37, 40",        "33, 36, 39, 42",    "35, 38, 41, 43",     "44, 45, 46"],
    ]

    col_labels = ["Fase / Etapa",
                  "1 · Segmentos", "2 · Fusión", "3 · Fine-tuning"]
    phase_rows = [
        ["backbone",          "Backbone → escalar\n(CV 5-fold)",
                               "4 escalares → Dense(128)",
                               "Descongelar N capas\nbajo LR"],
        ["simple_cnn",        "CNN → flatten\n(CV 5-fold)",
                               "4 flattens → Dense(512)",
                               "Descongelar extractores\nbajo LR"],
        ["backbone_vectors",  "Backbone → vector 256\n(CV 5-fold)",
                               "4 × 256 → Dense(512)",
                               "Descongelar N capas\nbajo LR"],
        ["unified_cnn",       "—\n(no existe)",
                               "—\n(no existe)",
                               "—\n(todo en 1 fase)"],
    ]

    col_widths = [0.28, 0.18, 0.18, 0.18, 0.18]
    col_colors = [C_HEADER, C_CONV, C_FEAT, C_FUSE, C_GENDER]
    text_colors = ["white", "black", "black", "black", "black"]

    n_rows = len(rows)
    row_h = 0.78 / (n_rows + 1)
    y0 = 0.88

    # Header
    x = 0.0
    for ci, (hdr, cw, cc, tc) in enumerate(zip(headers, col_widths, col_colors, text_colors)):
        ax.add_patch(FancyBboxPatch((x + 0.002, y0 - row_h + 0.005), cw - 0.004, row_h - 0.005,
                                   boxstyle="round,pad=0.005",
                                   facecolor=cc, edgecolor="white", lw=0.5,
                                   transform=ax.transAxes))
        ax.text(x + cw/2, y0 - row_h/2, hdr,
                ha="center", va="center", fontsize=FS_BOX,
                fontweight="bold", color=tc, transform=ax.transAxes)
        x += cw

    # Filas
    for ri, row in enumerate(rows):
        y = y0 - (ri + 1) * row_h
        fc_row = "#F7F9FC" if ri % 2 == 0 else "white"
        x = 0.0
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            ax.add_patch(FancyBboxPatch((x + 0.002, y - row_h + 0.003), cw - 0.004, row_h - 0.003,
                                       boxstyle="round,pad=0.003",
                                       facecolor="#E8F0FE" if ci == 0 else fc_row,
                                       edgecolor=CLINE, lw=0.4,
                                       transform=ax.transAxes))
            fw = "bold" if ci == 0 else "normal"
            ax.text(x + cw/2, y - row_h/2, cell,
                    ha="center", va="center", fontsize=FS_LABEL,
                    fontweight=fw, color="#222", transform=ax.transAxes,
                    linespacing=1.3)
            x += cw

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    out = "docs/arquitecturas.pdf"
    with PdfPages(out) as pdf:
        page_backbone(pdf)
        page_simple_cnn(pdf)
        page_backbone_vectors(pdf)
        page_unified_cnn(pdf)
        page_comparison(pdf)
    print(f"PDF generado: {out}")


if __name__ == "__main__":
    main()
