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
        runs.append({"dir": d, "name": os.path.basename(d), "cfg": cfg})
    return runs


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

    # Y positions (encoder: top → bottom, decoder: bottom → top)
    ys = [0.88 - i * 0.155 for i in range(n)]
    by = ys[-1] - 0.155  # bottleneck

    # Input
    _block(ax, EX, ys[0] + 0.135, W, H * 0.85, "Input", None, C["input"])
    _arrow(ax, EX, ys[0] + 0.135 - H * 0.425, EX, ys[0] + H / 2)

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
    _block(ax, DX, ys[0] + 0.135, W, H * 0.85, "Output", "224×224×5", C["output"])
    _arrow(ax, DX, ys[0] + H / 2, DX, ys[0] + 0.135 - H * 0.425)

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
                 fontsize=13, fontweight="bold", y=1.00)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.03,
                        hspace=0.15, wspace=0.08)

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

def page_summary(pdf, run):
    name = run["name"]
    cfg  = run["cfg"]
    arch = cfg.get("ARCHITECTURE", cfg.get("ENCODER", "?"))

    history_img = os.path.join(run["dir"], "training_history", "training_history.png")
    perf_img    = os.path.join(run["dir"], "evaluation", "performance_table.png")

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(f"{name}  —  {arch}", fontsize=14, fontweight="bold", y=0.98)

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


def page_prediction(pdf, run):
    name     = run["name"]
    arch     = run["cfg"].get("ARCHITECTURE", run["cfg"].get("ENCODER", "?"))
    pred_img = os.path.join(run["dir"], "evaluation", "test_prediction.png")

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.suptitle(f"{name}  —  {arch}  |  Predicción de prueba",
                 fontsize=12, fontweight="bold")
    img_or_blank(pred_img, ax)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── portada ──────────────────────────────────────────────────────────────────

def page_cover(pdf, runs):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax  = fig.add_axes([0.08, 0.08, 0.84, 0.78])
    ax.axis("off")

    fig.suptitle("Hand Detector — Resultados de experimentos",
                 fontsize=18, fontweight="bold", y=0.94)

    headers = ["#", "Arquitectura", "Weights", "Trainable", "Ch", "Timestamp"]
    rows = []
    for i, r in enumerate(runs):
        c = r["cfg"]
        arch    = c.get("ARCHITECTURE", c.get("ENCODER", "?"))
        weights = c.get("ENCODER_WEIGHTS") or "—"
        train   = c.get("BASE_MODEL_TRAINABLE")
        trainstr = ("—" if train is None else ("Sí" if train else "No"))
        ch      = str(c.get("INPUT_CHANNELS", 3))
        ts      = c.get("timestamp", "")[:16]
        rows.append([f"{i:02d}", arch, weights, trainstr, ch, ts])

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
        color = "#eaf0fb" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

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

    with PdfPages(args.out) as pdf:
        page_cover(pdf, runs)
        print("  Generando página de arquitecturas...")
        page_architecture_diagrams(pdf)
        for run in runs:
            print(f"  Generando páginas para {run['name']}...")
            page_summary(pdf, run)
            page_prediction(pdf, run)

    print(f"\nPDF generado: {args.out}")


if __name__ == "__main__":
    main()
