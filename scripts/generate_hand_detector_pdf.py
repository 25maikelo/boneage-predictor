#!/usr/bin/env python3
"""
Genera un PDF con los resultados de todos los experimentos del hand detector.
Por cada experimento incluye:
  - Portada   : tabla resumen de todos los runs
  - Arquitecturas: diagramas visuales de las 4 arquitecturas (encoder-decoder)
  - Por run   : config + curvas de entrenamiento + tabla de métricas + predicción

Uso:
    python generate_hand_detector_pdf.py
    python generate_hand_detector_pdf.py --out models/hand-detector/mi_reporte.pdf
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR       = os.path.join(PROJECT_ROOT, "models", "hand-detector")
DEFAULT_OUTPUT = os.path.join(RUNS_DIR, "hand_detector_experiments.pdf")


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_runs(runs_dir):
    pattern = os.path.join(runs_dir, "hand-detector_[0-9][0-9]")
    dirs = sorted(glob.glob(pattern))
    runs = []
    for d in dirs:
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        metrics_path = os.path.join(d, "evaluation", "metrics.json")
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
        runs.append({"dir": d, "name": os.path.basename(d), "cfg": cfg, "metrics": metrics})
    return runs


def get_top5(runs):
    """Devuelve un set con los índices (posición en la lista) de los 5 mejores por val_dice."""
    scored = [
        (i, r["metrics"]["val_dice"])
        for i, r in enumerate(runs)
        if r["metrics"] is not None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {i for i, _ in scored[:5]}


def fmt_cfg(cfg):
    arch    = cfg.get("ARCHITECTURE", cfg.get("ENCODER", "?"))
    weights = cfg.get("ENCODER_WEIGHTS")
    train   = cfg.get("BASE_MODEL_TRAINABLE")
    ch      = cfg.get("INPUT_CHANNELS", 3)
    aug     = cfg.get("DATA_AUGMENTATION", False)
    epochs  = cfg.get("EPOCHS", "?")
    bs      = cfg.get("BATCH_SIZE", "?")
    ts      = cfg.get("timestamp", "")
    lines = [
        f"Arquitectura    : {arch}",
        f"Encoder weights : {weights if weights else '—'}",
        f"Base trainable  : {train if train is not None else '—'}",
        f"Input channels  : {ch}",
        f"Data aug        : {aug}",
        f"Epochs          : {epochs}   |   Batch size: {bs}",
        f"Timestamp       : {ts}",
    ]
    return "\n".join(lines)


def img_or_blank(path, ax, title=""):
    if path and os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "imagen no disponible",
                ha="center", va="center", fontsize=9, color="gray",
                transform=ax.transAxes)
    ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")


# ─── diagramas de arquitectura ────────────────────────────────────────────────

# Paleta compartida
C = {
    "input":      "#2c3e50",
    "conv":       "#2980b9",
    "mnv2":       "#1a5276",
    "dw":         "#117a65",
    "bottleneck": "#922b21",
    "dec_conv":   "#d35400",
    "dec_dw":     "#1d8348",
    "output":     "#6c3483",
    "skip":       "#27ae60",
    "text":       "white",
}


def _block(ax, cx, cy, w, h, line1, line2, color, fs=6.5):
    """Dibuja un bloque redondeado centrado en (cx, cy)."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=color, edgecolor="white", linewidth=1.2,
        transform=ax.transData, clip_on=False,
    )
    ax.add_patch(rect)
    if line2:
        ax.text(cx, cy + h * 0.15, line1, ha="center", va="center",
                fontsize=fs, color=C["text"], fontweight="bold")
        ax.text(cx, cy - h * 0.2, line2, ha="center", va="center",
                fontsize=fs - 0.5, color=C["text"], alpha=0.9)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=fs, color=C["text"], fontweight="bold")


def _arrow(ax, x1, y1, x2, y2, color="#888888", lw=1.1):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=8),
                annotation_clip=False)


def _skip(ax, x1, y1, x2, y2):
    """Línea punteada de skip connection."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["skip"],
                                lw=0.9, mutation_scale=7,
                                linestyle="dashed"),
                annotation_clip=False)


def _draw_unet(ax, enc_color, dec_color, enc_label, dec_label,
               pretrained=False, title="U-Net", notes=""):
    """
    Dibuja un diagrama encoder-decoder estilo U-Net.
    5 niveles de resolución: 224 → 112 → 56 → 28 → 14 (bottleneck).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)

    EX = 0.23   # encoder x
    DX = 0.77   # decoder x
    BX = 0.50   # bottleneck x
    W  = 0.38   # block width
    H  = 0.10   # block height

    # Niveles: (res, channels_enc, channels_dec)
    levels = [
        ("224×224", "64",   "64"),
        ("112×112", "128",  "128"),
        ("56×56",   "256",  "256"),
        ("28×28",   "512",  "512"),
    ]
    n = len(levels)

    # Y positions: empezar más abajo para dejar espacio al título del subplot
    ys = [0.82 - i * 0.150 for i in range(n)]
    by = ys[-1] - 0.150  # bottleneck

    # Input
    _block(ax, EX, ys[0] + 0.120, W, H * 0.85, "Input", None, C["input"])
    _arrow(ax, EX, ys[0] + 0.120 - H * 0.425, EX, ys[0] + H / 2)

    # Encoder blocks
    for i, (res, ch_e, ch_d) in enumerate(levels):
        label2 = f"{res}×{ch_e}"
        if i == 0 and pretrained:
            _block(ax, EX, ys[i], W, H, enc_label, label2, C["mnv2"])
        else:
            _block(ax, EX, ys[i], W, H, enc_label, label2, enc_color)
        if i < n - 1:
            _arrow(ax, EX, ys[i] - H / 2, EX, ys[i + 1] + H / 2)

    # Bottleneck
    _arrow(ax, EX, ys[-1] - H / 2, BX - W / 2 + 0.01, by + H * 0.3)
    _block(ax, BX, by, W * 1.05, H, "Bottleneck", f"14×14×1024", C["bottleneck"])
    _arrow(ax, BX + W / 2 - 0.01, by + H * 0.3, DX, ys[-1] - H / 2)

    # Decoder blocks
    for i, (res, ch_e, ch_d) in enumerate(reversed(levels)):
        dy = ys[n - 1 - i]
        _block(ax, DX, dy, W, H, dec_label, f"{res}×{ch_d}", dec_color)
        if i < n - 1:
            _arrow(ax, DX, dy + H / 2, DX, ys[n - 2 - i] - H / 2)

    # Output
    _block(ax, DX, ys[0] + 0.120, W, H * 0.85, "Output", "224×224×5", C["output"])
    _arrow(ax, DX, ys[0] + H / 2, DX, ys[0] + 0.120 - H * 0.425)

    # Skip connections
    for i in range(n):
        _skip(ax, EX + W / 2, ys[i], DX - W / 2, ys[n - 1 - i])

    # Nota al pie
    if notes:
        ax.text(0.5, 0.01, notes, ha="center", va="bottom",
                fontsize=6, color="#555555", style="italic")

    # Leyenda de colores
    legend_items = [
        mpatches.Patch(color=C["input"],      label="Input"),
        mpatches.Patch(color=enc_color,       label="Encoder"),
        mpatches.Patch(color=C["bottleneck"], label="Bottleneck"),
        mpatches.Patch(color=dec_color,       label="Decoder"),
        mpatches.Patch(color=C["output"],     label="Output"),
        mpatches.Patch(color=C["skip"],       label="Skip conn."),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              fontsize=5.5, framealpha=0.8, ncol=2)


def page_architecture_diagrams(pdf):
    """Una página con los 4 diagramas de arquitectura en cuadrícula 2×2."""
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("Arquitecturas — Diagramas Encoder-Decoder",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.05,
                        hspace=0.30, wspace=0.08)

    # 1. U-Net clásica
    _draw_unet(
        axes[0, 0],
        enc_color=C["conv"], dec_color=C["dec_conv"],
        enc_label="Conv×2 + Pool", dec_label="UpConv + Concat",
        pretrained=False,
        title="U-Net  (desde cero)",
        notes="Encoder: Conv2D 3×3. Decoder: Conv2DTranspose. Soporta 1 o 3 canales.",
    )

    # 2. U-Net + MobileNetV2 encoder
    _draw_unet(
        axes[0, 1],
        enc_color=C["mnv2"], dec_color=C["dec_conv"],
        enc_label="MNv2 Block", dec_label="UpConv + Concat",
        pretrained=True,
        title="U-Net + MobileNetV2  (encoder preentrenado)",
        notes="Encoder: MobileNetV2 pretrained (imagenet opcional). Decoder: Conv2D estándar.",
    )

    # 3. MobileNetV2 encoder + decoder depthwise
    _draw_unet(
        axes[1, 0],
        enc_color=C["mnv2"], dec_color=C["dec_dw"],
        enc_label="MNv2 Block", dec_label="Depthwise Block",
        pretrained=True,
        title="MobileNetV2  (encoder + decoder depthwise)",
        notes="Encoder: MobileNetV2. Decoder: Inverted-residual blocks. Más ligero que U-Net estándar.",
    )

    # 4. Bloques MobileNetV2 completo (sin preentrenado)
    _draw_unet(
        axes[1, 1],
        enc_color=C["dw"], dec_color=C["dec_dw"],
        enc_label="Depthwise Block", dec_label="Depthwise Block",
        pretrained=False,
        title="MobileNetV2 Blocks  (todo desde cero)",
        notes="Encoder y decoder con inverted-residual blocks. Sin pesos preentrenados.",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── páginas por experimento ──────────────────────────────────────────────────

def page_summary(pdf, run, is_top5=False):
    name = run["name"]
    cfg  = run["cfg"]
    arch = cfg.get("ARCHITECTURE", cfg.get("ENCODER", "?"))

    history_img = os.path.join(run["dir"], "training_history", "training_history.png")
    perf_img    = os.path.join(run["dir"], "evaluation", "performance_table.png")

    fig = plt.figure(figsize=(11.69, 8.27))
    title = f"{name}  —  {arch}"
    if is_top5:
        title = f"★  {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98,
                 color="#856404" if is_top5 else "black")
    if is_top5:
        fig.patch.set_facecolor("#fffdf0")

    gs = fig.add_gridspec(
        2, 2,
        left=0.04, right=0.97, top=0.91, bottom=0.04,
        hspace=0.35, wspace=0.25, height_ratios=[1, 1.4],
    )

    ax_cfg = fig.add_subplot(gs[0, 0])
    ax_cfg.axis("off")
    ax_cfg.set_title("Configuración", fontsize=9, pad=4, loc="left")
    ax_cfg.text(0.02, 0.95, fmt_cfg(cfg),
                transform=ax_cfg.transAxes,
                fontsize=8.5, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#f5f5f5", edgecolor="#cccccc"))

    ax_perf = fig.add_subplot(gs[0, 1])
    img_or_blank(perf_img, ax_perf, "Métricas de validación")

    ax_hist = fig.add_subplot(gs[1, :])
    img_or_blank(history_img, ax_hist, "Curvas de entrenamiento")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_prediction(pdf, run, is_top5=False):
    name     = run["name"]
    arch     = run["cfg"].get("ARCHITECTURE", run["cfg"].get("ENCODER", "?"))
    pred_img = os.path.join(run["dir"], "evaluation", "test_prediction.png")

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    title = f"{name}  —  {arch}  |  Predicción de prueba"
    if is_top5:
        title = f"★  {title}"
        fig.patch.set_facecolor("#fffdf0")
    fig.suptitle(title, fontsize=12, fontweight="bold",
                 color="#856404" if is_top5 else "black")
    img_or_blank(pred_img, ax)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── descripciones de arquitecturas ──────────────────────────────────────────

ARCH_DESCRIPTIONS = {
    "unet": {
        "titulo": "U-Net (desde cero)",
        "descripcion": (
            "Red encoder-decoder clásica entrenada completamente desde cero. "
            "El encoder reduce la resolución con capas Conv2D + MaxPooling; el decoder la recupera "
            "con convoluciones transpuestas. Skip connections entre niveles equivalentes preservan "
            "detalles espaciales finos. No usa backbone ni pesos preentrenados — "
            "ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE se ignoran. Soporta 1 o 3 canales de entrada."
        ),
    },
    "unet_mobilenetv2": {
        "titulo": "U-Net + MobileNetV2 (encoder ImageNet)",
        "descripcion": (
            "Misma estructura decoder que U-Net, pero el encoder es MobileNetV2, una red eficiente "
            "basada en bloques inverted residual (expand → depthwise → project). "
            "El encoder puede inicializarse con pesos ImageNet (Keras built-in) o entrenarse desde cero. "
            "No es compatible con pesos RadImageNet — RadImageNet no ofrece pesos para MobileNetV2. "
            "Soporta 1 canal replicando el canal único x3 antes del backbone."
        ),
    },
    "mobilenetv2_sym": {
        "titulo": "MobileNetV2 simétrico (encoder + decoder espejo)",
        "descripcion": (
            "Usa MobileNetV2 como encoder. A diferencia de unet_mobilenetv2, el decoder es un espejo "
            "del encoder: cada bloque sigue la misma estructura expand → upsample+depthwise → project, "
            "manteniendo la eficiencia de MobileNetV2 en todo el modelo. "
            "Pesos disponibles: ImageNet o desde cero. "
            "No compatible con RadImageNet (no existen pesos RadImageNet para MobileNetV2)."
        ),
    },
    "unet_resnet50": {
        "titulo": "U-Net + ResNet50 (encoder ImageNet / RadImageNet)",
        "descripcion": (
            "Encoder basado en ResNet50, arquitectura con conexiones residuales que facilitan el "
            "entrenamiento profundo (~25M parámetros). El decoder usa Conv2DTranspose estándar con "
            "skip connections desde el encoder, igual que U-Net clásica. "
            "Pesos disponibles: ImageNet (Keras built-in), RadImageNet (archivo .h5 local, "
            "Mei et al. 2022) o desde cero. Es el principal candidato para explotar RadImageNet."
        ),
    },
    "unet_densenet121": {
        "titulo": "U-Net + DenseNet121 (encoder ImageNet / RadImageNet)",
        "descripcion": (
            "Encoder basado en DenseNet121, donde cada capa recibe las salidas de todas las capas "
            "anteriores del mismo bloque (dense connections), favoreciendo la reutilización de "
            "características. Es el encoder más ligero con backbone (~8M parámetros), especialmente "
            "efectivo con datasets pequeños. Pesos disponibles: ImageNet (Keras built-in), "
            "RadImageNet (archivo .h5 local) o desde cero. El decoder usa la misma estructura que U-Net."
        ),
    },
}


ARCH_PSEUDOCODE = {
    "unet": {
        "titulo": "U-Net  —  Red entrenada desde cero",
        "color":  "#2c3e50",
        "pasos": [
            ("ENTRADA",
             "Imagen de mano en escala de grises o color (224×224 píxeles)"),
            ("COMPRESIÓN — Paso 1  →  imagen 112×112",
             "Se aplican dos filtros para detectar bordes y texturas.\n"
             "Se guarda una copia de este resultado (conexión directa al decoder).\n"
             "La imagen se reduce a la mitad con un MaxPooling."),
            ("COMPRESIÓN — Paso 2  →  imagen 56×56",
             "Dos filtros más complejos detectan formas y contornos.\n"
             "Se guarda copia. La imagen se reduce de nuevo."),
            ("COMPRESIÓN — Paso 3  →  imagen 28×28",
             "Se detectan patrones más abstractos (partes de dedos, muñeca).\n"
             "Se guarda copia. Nueva reducción."),
            ("COMPRESIÓN — Paso 4  →  imagen 14×14",
             "Patrones de alto nivel (estructura global de la mano).\n"
             "Se guarda copia. Nueva reducción."),
            ("PUNTO MÍNIMO (Bottleneck)  →  imagen 7×7",
             "La imagen está en su máxima compresión.\n"
             "El modelo tiene aquí una 'visión global' de toda la mano."),
            ("RECONSTRUCCIÓN — Paso 1  →  imagen 14×14",
             "Se amplía la imagen y se combina con la copia guardada en Compresión 4.\n"
             "Esto recupera detalles espaciales que se habían perdido al comprimir."),
            ("RECONSTRUCCIÓN — Pasos 2, 3, 4  →  28×28 → 56×56 → 112×112",
             "Se repite el proceso de ampliar + combinar con la copia correspondiente.\n"
             "Cada paso recupera más detalle fino de la imagen original."),
            ("RECONSTRUCCIÓN — Paso final  →  imagen 224×224",
             "Se amplía a la resolución original sin combinar con ninguna copia."),
            ("SALIDA",
             "Cada píxel recibe una etiqueta: fondo, meñique, medio, pulgar o muñeca."),
        ],
    },
    "unet_mobilenetv2": {
        "titulo": "U-Net + MobileNetV2  —  Encoder con pesos preentrenados (ImageNet)",
        "color":  "#1a5276",
        "pasos": [
            ("ENTRADA",
             "Imagen de mano. Si es en escala de grises, se replica 3 veces\n"
             "para que el encoder preentrenado pueda procesarla."),
            ("ENCODER: MobileNetV2 (preentrenado en ImageNet)",
             "En lugar de aprender desde cero, se usa MobileNetV2, una red que ya\n"
             "aprendió a reconocer formas en millones de imágenes del mundo real.\n"
             "Cada bloque usa una técnica eficiente: primero amplía canales, aplica\n"
             "un filtro ligero por canal, y luego comprime. Es más rápido que U-Net puro.\n"
             "Se guardan 4 copias intermedias (112×112, 56×56, 28×28, 14×14)."),
            ("BOTTLENECK  →  imagen 7×7",
             "Salida final del encoder. Representación compacta de la mano completa."),
            ("DECODER: convoluciones transpuestas estándar",
             "Igual que U-Net clásica: se amplía y se combina con cada copia guardada.\n"
             "14×14 → 28×28 → 56×56 → 112×112 → 224×224\n"
             "El decoder aprende desde cero en cada entrenamiento."),
            ("SALIDA",
             "Mapa de segmentación 224×224 con 5 etiquetas por píxel."),
        ],
    },
    "mobilenetv2_sym": {
        "titulo": "MobileNetV2 Simétrico  —  Encoder y decoder con la misma arquitectura",
        "color":  "#117a65",
        "pasos": [
            ("ENTRADA",
             "Imagen de mano. Si es en escala de grises, se replica 3 veces."),
            ("ENCODER: MobileNetV2 (idéntico a unet_mobilenetv2)",
             "Se usa MobileNetV2 para comprimir la imagen.\n"
             "Se guardan 4 copias en los mismos puntos: 112×112, 56×56, 28×28, 14×14."),
            ("BOTTLENECK  →  imagen 7×7",
             "Representación compacta de la mano."),
            ("DECODER: bloques espejo de MobileNetV2",
             "A diferencia de U-Net, el decoder no usa convoluciones transpuestas simples.\n"
             "Cada bloque sigue la misma lógica que el encoder:\n"
             "  1. Ampliar canales  2. Doblar resolución con filtro ligero  3. Comprimir canales.\n"
             "El skip connection (copia guardada) se inserta entre los pasos 2 y 3.\n"
             "Esto mantiene la eficiencia de MobileNetV2 también en la parte de reconstrucción."),
            ("SALIDA",
             "Mapa de segmentación 224×224 con 5 etiquetas por píxel."),
        ],
    },
    "unet_resnet50": {
        "titulo": "U-Net + ResNet50  —  Encoder con pesos ImageNet o RadImageNet",
        "color":  "#922b21",
        "pasos": [
            ("ENTRADA",
             "Imagen de mano en color (3 canales). ResNet50 no soporta 1 canal directamente."),
            ("ENCODER: ResNet50 (preentrenado)",
             "ResNet50 aprende de forma más profunda gracias a sus conexiones residuales:\n"
             "cada bloque suma su propia salida con su entrada (atajo directo).\n"
             "Esto evita que la red 'olvide' lo aprendido en capas anteriores.\n"
             "Tiene ~25 millones de parámetros y 4 niveles de compresión.\n"
             "Pesos disponibles: ImageNet (objetos cotidianos) o RadImageNet\n"
             "(imágenes médicas — CT, MRI, ultrasonido). RadImageNet está más cerca\n"
             "de nuestro dominio aunque no incluye radiografías de mano directamente."),
            ("BOTTLENECK  →  imagen 7×7",
             "Representación global de la mano con 2048 canales de información."),
            ("DECODER: convoluciones transpuestas estándar (igual que U-Net)",
             "Se amplía paso a paso combinando con las copias guardadas del encoder.\n"
             "7×7 → 14×14 → 28×28 → 56×56 → 112×112 → 224×224"),
            ("SALIDA",
             "Mapa de segmentación 224×224 con 5 etiquetas por píxel."),
        ],
    },
    "unet_densenet121": {
        "titulo": "U-Net + DenseNet121  —  Encoder con conexiones densas",
        "color":  "#6c3483",
        "pasos": [
            ("ENTRADA",
             "Imagen de mano en color (3 canales)."),
            ("ENCODER: DenseNet121 (preentrenado)",
             "DenseNet121 usa una estrategia diferente a ResNet: cada capa recibe\n"
             "las salidas de TODAS las capas anteriores dentro del mismo bloque,\n"
             "no solo la inmediatamente anterior. Esto reutiliza características\n"
             "aprendidas en capas anteriores sin necesidad de volver a aprenderlas.\n"
             "Es el encoder más ligero (~8M parámetros) y especialmente efectivo\n"
             "con datasets pequeños, como el nuestro (~379 imágenes).\n"
             "Pesos disponibles: ImageNet o RadImageNet."),
            ("ENTRE BLOQUES: capas de transición",
             "Entre cada bloque denso hay una capa que comprime los canales a la mitad\n"
             "y reduce la resolución con un pooling. Estas son los puntos de skip connection."),
            ("BOTTLENECK  →  imagen 7×7",
             "Representación compacta con 1024 canales."),
            ("DECODER: convoluciones transpuestas estándar (igual que U-Net)",
             "Amplía la imagen combinando con las copias de las transiciones.\n"
             "7×7 → 14×14 → 28×28 → 56×56 → 112×112 → 224×224"),
            ("SALIDA",
             "Mapa de segmentación 224×224 con 5 etiquetas por píxel."),
        ],
    },
}


def page_architecture_descriptions(pdf):
    """Una página por arquitectura — bloques apilados sin solapamiento."""
    LINE_H   = 0.32   # altura por línea de descripción (unidades de datos)
    TITLE_H  = 0.55   # altura fija para el título de cada bloque
    GAP      = 0.12   # espacio entre bloques
    PAD_Y    = 0.20   # padding interno vertical
    X0, X1  = 0.5, 99.5  # márgenes horizontales (espacio de datos 0–100)

    for arch, info in ARCH_PSEUDOCODE.items():
        pasos = info["pasos"]
        color = info["color"]

        # Calcular altura total necesaria (de abajo hacia arriba)
        heights = []
        for _, desc in pasos:
            n_lines = desc.count("\n") + 1
            heights.append(TITLE_H + n_lines * LINE_H + 2 * PAD_Y)

        total_h = sum(heights) + GAP * len(pasos) + 1.5  # +1.5 para header

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, total_h)
        ax.axis("off")
        ax.invert_yaxis()

        # ── Header ────────────────────────────────────────────────────────────
        header = plt.matplotlib.patches.FancyBboxPatch(
            (0, 0), 100, 1.3,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=color, transform=ax.transData, clip_on=False,
        )
        ax.add_patch(header)
        ax.text(50, 0.65, info["titulo"],
                fontsize=13, fontweight="bold", color="white",
                ha="center", va="center")

        # ── Bloques de pasos ──────────────────────────────────────────────────
        y = 1.5  # cursor y (crece hacia abajo)
        bg_colors = ["#eef3fb", "#ffffff"]

        for i, ((titulo_paso, descripcion), h) in enumerate(zip(pasos, heights)):
            # Fondo del bloque
            bg = plt.matplotlib.patches.FancyBboxPatch(
                (X0, y), X1 - X0, h,
                boxstyle="round,pad=0.05", linewidth=0.6,
                edgecolor="#dddddd", facecolor=bg_colors[i % 2],
                transform=ax.transData,
            )
            ax.add_patch(bg)

            # Pastilla numerada
            badge = plt.matplotlib.patches.FancyBboxPatch(
                (X0 + 0.3, y + PAD_Y), 3.2, TITLE_H,
                boxstyle="round,pad=0.05", linewidth=0,
                facecolor=color, alpha=0.85,
                transform=ax.transData,
            )
            ax.add_patch(badge)
            ax.text(X0 + 1.9, y + PAD_Y + TITLE_H / 2, str(i + 1),
                    fontsize=9, fontweight="bold", color="white",
                    ha="center", va="center")

            # Título del paso
            ax.text(X0 + 4.5, y + PAD_Y + TITLE_H / 2, titulo_paso,
                    fontsize=9, fontweight="bold", color="#1a1a1a",
                    ha="left", va="center")

            # Descripción
            ax.text(X0 + 4.5, y + PAD_Y + TITLE_H + 0.15, descripcion,
                    fontsize=8, color="#444444",
                    ha="left", va="top", linespacing=1.5,
                    fontfamily="monospace")

            y += h + GAP

        fig.tight_layout(pad=0)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ─── portada ──────────────────────────────────────────────────────────────────

def page_cover(pdf, runs):
    # ── Página 1: título + top 5 ─────────────────────────────────────────────
    top5 = get_top5(runs)
    scored = sorted(
        [(i, r) for i, r in enumerate(runs) if r["metrics"] is not None],
        key=lambda x: x[1]["metrics"]["val_dice"], reverse=True
    )[:5]

    fig_title = plt.figure(figsize=(11.69, 8.27))
    ax_t = fig_title.add_axes([0, 0, 1, 1])
    ax_t.axis("off")

    ax_t.text(0.5, 0.88, "Hand Detector",
              ha="center", va="center", fontsize=36, fontweight="bold",
              transform=ax_t.transAxes, color="#2c3e50")
    ax_t.text(0.5, 0.79, "Resultados de experimentos",
              ha="center", va="center", fontsize=22,
              transform=ax_t.transAxes, color="#555555")
    ax_t.text(0.5, 0.72, f"{len(runs)} runs registrados",
              ha="center", va="center", fontsize=13,
              transform=ax_t.transAxes, color="#888888")

    # ── Tabla top 5 ──────────────────────────────────────────────────────────
    ax_t.text(0.5, 0.63, "★  Top 5 — Mejor Dice Score",
              ha="center", va="center", fontsize=12, fontweight="bold",
              transform=ax_t.transAxes, color="#856404")

    ax_rank = fig_title.add_axes([0.03, 0.28, 0.94, 0.33])
    ax_rank.axis("off")

    rank_headers = ["Pos.", "Experimento", "Arquitectura", "Weights", "Aug",
                    "Loss ↓", "Accuracy ↑", "IoU ↑", "Dice ↑"]
    rank_rows = []
    for pos, (idx, r) in enumerate(scored, 1):
        c = r["cfg"]
        m = r["metrics"]
        arch    = c.get("ARCHITECTURE", "?")
        weights = c.get("ENCODER_WEIGHTS") or "—"
        aug     = "Sí" if c.get("DATA_AUGMENTATION", False) else "No"
        rank_rows.append([
            f"#{pos}", r["name"], arch, weights, aug,
            f"{m['val_loss']:.4f}",
            f"{m['val_accuracy']:.4f}",
            f"{m['val_iou']:.4f}",
            f"{m['val_dice']:.4f}",
        ])

    rank_table = ax_rank.table(
        cellText=rank_rows, colLabels=rank_headers,
        loc="center", cellLoc="center",
    )
    rank_table.auto_set_font_size(False)
    rank_table.set_fontsize(8.5)
    rank_table.scale(1, 2.0)

    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#fff3cd", "#fff3cd"]
    for j in range(len(rank_headers)):
        rank_table[0, j].set_facecolor("#2c3e50")
        rank_table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rank_rows) + 1):
        for j in range(len(rank_headers)):
            rank_table[i, j].set_facecolor(medal_colors[i - 1])
        rank_table[i, 0].set_text_props(fontweight="bold")

    # ── Descripción de métricas en subaxes separados ─────────────────────────
    import textwrap
    metric_descs = [
        ("Loss ↓",     "#c0392b",
         "Error promedio del modelo. Menor es mejor.\n"
         "Mide qué tan lejos están las predicciones\n"
         "de las etiquetas reales."),
        ("Accuracy ↑", "#27ae60",
         "Porcentaje de píxeles clasificados\n"
         "correctamente. Puede ser engañoso si\n"
         "el fondo domina (clase mayoritaria)."),
        ("IoU ↑",      "#2980b9",
         "Intersección sobre Unión. Mide el\n"
         "solapamiento entre máscara predicha\n"
         "y real. Más estricto que Accuracy."),
        ("Dice ↑",     "#856404",
         "Coeficiente Dice (F1 espacial). Similar\n"
         "a IoU pero más sensible a regiones\n"
         "pequeñas. Métrica principal de ranking."),
    ]

    n_metrics = len(metric_descs)
    pad_left  = 0.03
    pad_right = 0.03
    gap       = 0.015
    total_w   = 1.0 - pad_left - pad_right - gap * (n_metrics - 1)
    cell_w    = total_w / n_metrics
    cell_h    = 0.20
    cell_y    = 0.03

    for k, (name, color, desc) in enumerate(metric_descs):
        x = pad_left + k * (cell_w + gap)
        ax_m = fig_title.add_axes([x, cell_y, cell_w, cell_h])
        ax_m.set_facecolor("#f8f9fa")
        for spine in ax_m.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
        ax_m.set_xticks([]); ax_m.set_yticks([])

        ax_m.text(0.5, 0.82, name,
                  transform=ax_m.transAxes,
                  fontsize=9, fontweight="bold", color=color,
                  ha="center", va="center")
        ax_m.axhline(y=0.65, xmin=0.05, xmax=0.95,
                     color=color, linewidth=0.8, alpha=0.5)
        ax_m.text(0.5, 0.35, desc,
                  transform=ax_m.transAxes,
                  fontsize=7.5, color="#333333",
                  ha="center", va="center", linespacing=1.5)

    pdf.savefig(fig_title, bbox_inches="tight")
    plt.close(fig_title)

    # ── Página 2: tabla resumen ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    ax  = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.axis("off")

    top5 = get_top5(runs)

    headers = ["#", "Arquitectura", "Weights", "Trainable", "Ch", "Data Aug", "Dice", "Timestamp"]
    rows = []
    for i, r in enumerate(runs):
        c = r["cfg"]
        arch    = c.get("ARCHITECTURE", c.get("ENCODER", "?"))
        weights = c.get("ENCODER_WEIGHTS") or "—"
        train   = c.get("BASE_MODEL_TRAINABLE")
        trainstr = ("—" if train is None else ("Sí" if train else "No"))
        ch      = str(c.get("INPUT_CHANNELS", 3))
        aug     = "Sí" if c.get("DATA_AUGMENTATION", False) else "No"
        dice    = f"{r['metrics']['val_dice']:.4f}" if r["metrics"] else "—"
        ts      = c.get("timestamp", "")[:16]
        rows.append([f"{i:02d}", arch, weights, trainstr, ch, aug, dice, ts])

    table = ax.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        if (i - 1) in top5:
            row_color = "#fff3cd"   # amarillo suave — top 5
        else:
            row_color = "#eaf0fb" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(row_color)
        if (i - 1) in top5:
            table[i, 0].set_text_props(fontweight="bold", color="#856404")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUTPUT,
                        help=f"Ruta del PDF de salida (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--runs-dir", default=RUNS_DIR,
                        help="Directorio con los runs hand-detector_XX")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No se encontraron runs en: {args.runs_dir}")
        sys.exit(1)

    print(f"Runs encontrados: {len(runs)}")
    for r in runs:
        print(f"  {r['name']}  —  {r['cfg'].get('ARCHITECTURE', '?')}")

    top5 = get_top5(runs)

    with PdfPages(args.out) as pdf:
        page_cover(pdf, runs)
        print("  Generando páginas de descripción de arquitecturas...")
        page_architecture_descriptions(pdf)
        for i, run in enumerate(runs):
            is_top5 = i in top5
            label = " ★ TOP 5" if is_top5 else ""
            print(f"  Generando páginas para {run['name']}{label}...")
            page_summary(pdf, run, is_top5=is_top5)
            page_prediction(pdf, run, is_top5=is_top5)

    print(f"\nPDF generado: {args.out}")


if __name__ == "__main__":
    main()
