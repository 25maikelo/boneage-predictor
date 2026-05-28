#!/usr/bin/env python3
"""
Análisis de efectividad por rango de edad.

Ejecuta inferencia sobre ambos datasets de validación (estándar + mexicano),
calcula estadísticas por intervalo de edad y genera visualizaciones detalladas.

Uso:
    python src/10_age_range_analysis.py --experiment 34
    python src/10_age_range_analysis.py --experiment 34 --bin-size 6 --dataset val
"""
import os
import sys
import argparse
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model
from scipy import stats

from config.paths import (VALIDATION_CSV, VALIDATION_IMAGES_DIR,
                          MEX_CSV, MEX_IMAGES_DIR)
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.utils.losses import LOSS_MAP, dynamic_attention_loss
from src.utils.timing import setup_logging, timer

START_TIME = time.time()

# Rangos pediátricos estándar (meses)
PEDIATRIC_RANGES = [
    (0,   24,  "Lactante (0–2 a)"),
    (24,  60,  "Preescolar (2–5 a)"),
    (60,  120, "Escolar (5–10 a)"),
    (120, 168, "Adolescente temprano (10–14 a)"),
    (168, 228, "Adolescente tardío (14–19 a)"),
]


# ============================================================
# UTILIDADES (replicadas de 07 para independencia)
# ============================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU: {[g.name for g in gpus]}")
    else:
        print("Sin GPU — usando CPU.")


def get_optimizer(choice, lr):
    if (choice or "").lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def frame_and_zoom(img, kernel_size=(9, 9), dilation_iters=4):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = thresh.shape
    if (np.any(thresh[0, :] == 255) or np.any(thresh[-1, :] == 255) or
            np.any(thresh[:, 0] == 255) or np.any(thresh[:, -1] == 255)):
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones(kernel_size, np.uint8)
    dil = cv2.dilate(thresh, kernel, iterations=dilation_iters)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    angle = rect[-1]
    w_rect, h_rect = rect[1]
    if w_rect < h_rect:
        angle += 90
    if angle > 45:
        angle -= 90
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((len(box), 1))
    box_h = np.hstack([box, ones])
    box_r = np.dot(M, box_h.T).T.astype(int)
    x0, y0 = box_r[:, 0].min(), box_r[:, 1].min()
    x1, y1 = box_r[:, 0].max(), box_r[:, 1].max()
    if x0 >= x1 or y0 >= y1:
        return img
    crop = rot[y0:y1, x0:x1]
    return crop if crop.size >= 0.2 * img.size else img


def clahe_equalize(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    blended = cv2.addWeighted(img, 0.6, eq, 0.4, 0)
    return cv2.convertScaleAbs(blended, alpha=0.9, beta=-10)


def segment_spatial(img, seg_model, segments_order):
    h0, w0 = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(rgb, (224, 224)) / 255.0
    pred = seg_model.predict(inp[np.newaxis, ...], verbose=0)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)
    mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    segments = {}
    for cid, name in enumerate(segments_order, start=1):
        m = (mask == cid).astype(np.uint8)
        m3 = np.repeat(m[:, :, None] * 255, 3, axis=2)
        orig3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        segments[name] = np.where(m3 == 255, orig3, 0).astype(np.uint8)
    return segments


def normalize_image(img):
    return img.astype("float32") / 255.0


def load_all_models(cfg, exp_dir):
    from config.paths import get_segmentation_model_path
    cfg_seg = getattr(cfg, "SEGMENTATION_MODEL", None)
    if cfg_seg:
        seg_path = cfg_seg if os.path.isabs(cfg_seg) else os.path.join(PROJECT_ROOT, cfg_seg)
    else:
        seg_path = get_segmentation_model_path()
    seg_model = load_model(seg_path, compile=False)

    models_dir = os.path.join(exp_dir, "models")
    model_type = getattr(cfg, "MODEL_TYPE", "backbone")
    loss_fn = LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss)
    opt = get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE)

    if model_type == "unified_cnn":
        fusion = load_model(os.path.join(models_dir, "unified_model"), custom_objects=LOSS_MAP)
    else:
        fusion = load_model(os.path.join(models_dir, "fusion_model"), custom_objects=LOSS_MAP)
    fusion.compile(optimizer=opt, loss=loss_fn, metrics=["mae"])
    return seg_model, fusion


# ============================================================
# INFERENCIA
# ============================================================
def _parse_age_to_months(age_str):
    """Convierte edad a meses. Maneja años enteros (≤20), formato 'A,M' y valores ya en meses (>20)."""
    if pd.isnull(age_str):
        return np.nan
    if isinstance(age_str, (int, float)):
        return float(age_str) * 12 if age_str <= 20 else float(age_str)
    s = str(age_str).replace('"', '').strip()
    if "," in s:
        try:
            y, m = s.split(",")
            return int(y) * 12 + int(m)
        except Exception:
            return np.nan
    try:
        num = float(s)
        return num * 12 if num <= 20 else num
    except Exception:
        return np.nan


def _normalize_df(df):
    """Estandariza nombres de columnas para compatibilidad entre datasets."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    needs_age_conversion = False
    for alias, target in [("bone_age", "boneage"), ("real_age", "boneage"),
                           ("gender", "male"), ("id", "id")]:
        if alias in df.columns and target not in df.columns:
            df[target] = df[alias]
            if alias in ("bone_age", "real_age"):
                needs_age_conversion = True
    if needs_age_conversion:
        df["boneage"] = df["boneage"].apply(_parse_age_to_months)
    return df


def run_inference(df, images_dir, seg_model, fusion_model, cfg, dataset_label):
    records = []
    df = _normalize_df(df)

    for _, row in df.iterrows():
        sid = str(row["id"])
        real_age = float(pd.to_numeric(row["boneage"], errors="coerce"))
        if pd.isna(real_age):
            continue
        gender_val = row.get("male", 0)
        if pd.isna(gender_val):
            gender = 0.0
        elif str(gender_val).upper() in ("M", "MALE", "TRUE", "1"):
            gender = 1.0
        elif str(gender_val).upper() in ("F", "FEMALE", "FALSE", "0"):
            gender = 0.0
        else:
            try:
                gender = float(gender_val)
            except (ValueError, TypeError):
                gender = 0.0

        img_path = os.path.join(images_dir, f"{sid}.png")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            records.append({"id": sid, "dataset": dataset_label, "real_age": real_age,
                            "pred_age": None, "error": None, "abs_error": None,
                            "gender": gender, "status": "not_found"})
            continue
        try:
            zoomed = frame_and_zoom(gray)
            eq = clahe_equalize(zoomed)
            segments = segment_spatial(eq, seg_model, cfg.SEGMENTS_ORDER)
            if any(np.sum(s) == 0 for s in segments.values()):
                raise ValueError("empty_segment")

            fusion_inputs = [
                tf.expand_dims(normalize_image(
                    cv2.resize(segments[seg], cfg.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)), 0)
                for seg in cfg.SEGMENTS_ORDER
            ]
            if cfg.USE_GENDER:
                fusion_inputs.append(tf.constant([[gender]], dtype=tf.float32))

            pred_age = float(fusion_model.predict(fusion_inputs, verbose=0).flatten()[0])
            error = pred_age - real_age
            records.append({"id": sid, "dataset": dataset_label, "real_age": real_age,
                            "pred_age": pred_age, "error": error,
                            "abs_error": abs(error), "gender": gender, "status": "ok"})
        except Exception as e:
            records.append({"id": sid, "dataset": dataset_label, "real_age": real_age,
                            "pred_age": None, "error": None, "abs_error": None,
                            "gender": gender, "status": str(e)})

    return records


# ============================================================
# ESTADÍSTICAS POR BIN
# ============================================================
def compute_bin_stats(df_ok, bin_size):
    min_age = int(df_ok["real_age"].min())
    max_age = int(df_ok["real_age"].max()) + 1
    edges = list(range(0, max_age + bin_size, bin_size))

    stats_rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        subset = df_ok[(df_ok["real_age"] >= lo) & (df_ok["real_age"] < hi)]
        if subset.empty:
            continue
        errs = subset["error"].values
        abs_errs = subset["abs_error"].values
        _, p_norm = stats.shapiro(abs_errs) if len(abs_errs) >= 3 else (None, None)
        stats_rows.append({
            "bin_label": f"{lo}–{hi} m ({lo//12}–{hi//12} a)",
            "lo": lo, "hi": hi,
            "n": len(subset),
            "mae": float(np.mean(abs_errs)),
            "rmse": float(np.sqrt(np.mean(errs**2))),
            "bias": float(np.mean(errs)),
            "std": float(np.std(errs)),
            "p10": float(np.percentile(abs_errs, 10)),
            "p90": float(np.percentile(abs_errs, 90)),
            "within_12m": float(np.mean(abs_errs <= 12) * 100),
            "within_24m": float(np.mean(abs_errs <= 24) * 100),
        })
    return pd.DataFrame(stats_rows)


def compute_pediatric_stats(df_ok):
    rows = []
    for lo, hi, label in PEDIATRIC_RANGES:
        subset = df_ok[(df_ok["real_age"] >= lo) & (df_ok["real_age"] < hi)]
        if subset.empty:
            rows.append({"label": label, "n": 0, "mae": None, "bias": None,
                         "within_12m": None, "within_24m": None})
            continue
        errs = subset["error"].values
        abs_errs = subset["abs_error"].values
        rows.append({
            "label": label, "n": len(subset),
            "mae": float(np.mean(abs_errs)),
            "rmse": float(np.sqrt(np.mean(errs**2))),
            "bias": float(np.mean(errs)),
            "std": float(np.std(errs)),
            "within_12m": float(np.mean(abs_errs <= 12) * 100),
            "within_24m": float(np.mean(abs_errs <= 24) * 100),
        })
    return pd.DataFrame(rows)


# ============================================================
# VISUALIZACIONES
# ============================================================
def plot_mae_by_bin(bin_stats, output_dir, bin_size):
    fig, ax = plt.subplots(figsize=(max(10, len(bin_stats) * 0.7), 5))
    colors = plt.cm.RdYlGn_r(
        (bin_stats["mae"] - bin_stats["mae"].min()) /
        (bin_stats["mae"].max() - bin_stats["mae"].min() + 1e-9))
    bars = ax.bar(range(len(bin_stats)), bin_stats["mae"], color=colors, edgecolor="white")
    ax.axhline(bin_stats["mae"].mean(), color="navy", linestyle="--", linewidth=1.5,
               label=f'Global: {bin_stats["mae"].mean():.1f} m')
    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels([f"{r.lo}–{r.hi}" for _, r in bin_stats.iterrows()],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Rango de edad (meses)"); ax.set_ylabel("MAE (meses)")
    ax.set_title(f"MAE por rango de edad (intervalo {bin_size} m)", fontsize=13)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar, (_, r) in zip(bars, bin_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{r["n"]}', ha="center", va="bottom", fontsize=7, color="#333")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_por_rango.png"), dpi=150)
    plt.close()


def plot_bias_by_bin(bin_stats, output_dir, bin_size):
    fig, ax = plt.subplots(figsize=(max(10, len(bin_stats) * 0.7), 5))
    colors = ["#E74C3C" if b > 0 else "#2ECC71" for b in bin_stats["bias"]]
    ax.bar(range(len(bin_stats)), bin_stats["bias"], color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels([f"{r.lo}–{r.hi}" for _, r in bin_stats.iterrows()],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Rango de edad (meses)"); ax.set_ylabel("Sesgo (meses)")
    ax.set_title(f"Sesgo por rango (+ = sobreestima · − = subestima, intervalo {bin_size} m)",
                 fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sesgo_por_rango.png"), dpi=150)
    plt.close()


def plot_scatter_colored(df_ok, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Scatter coloreado por error absoluto
    sc = axes[0].scatter(df_ok["real_age"], df_ok["pred_age"],
                         c=df_ok["abs_error"], cmap="RdYlGn_r",
                         vmin=0, vmax=df_ok["abs_error"].quantile(0.95),
                         alpha=0.6, s=15, edgecolors="none")
    plt.colorbar(sc, ax=axes[0], label="Error absoluto (m)")
    lims = [min(df_ok["real_age"].min(), df_ok["pred_age"].min()),
            max(df_ok["real_age"].max(), df_ok["pred_age"].max())]
    axes[0].plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    axes[0].set_xlabel("Edad real (meses)"); axes[0].set_ylabel("Predicción (meses)")
    axes[0].set_title("Dispersión — coloreado por error"); axes[0].grid(alpha=0.3)

    # Error vs edad real
    axes[1].scatter(df_ok["real_age"], df_ok["error"],
                    c=df_ok["error"], cmap="coolwarm", vmin=-60, vmax=60,
                    alpha=0.5, s=15, edgecolors="none")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Edad real (meses)"); axes[1].set_ylabel("Error (pred − real, meses)")
    axes[1].set_title("Error vs edad real"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_error.png"), dpi=150)
    plt.close()


def plot_violin_by_pediatric(df_ok, output_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    groups, labels, ns = [], [], []
    for lo, hi, label in PEDIATRIC_RANGES:
        subset = df_ok[(df_ok["real_age"] >= lo) & (df_ok["real_age"] < hi)]["error"].dropna()
        if len(subset) >= 5:
            groups.append(subset.values)
            labels.append(f"{label}\n(n={len(subset)})")
            ns.append(len(subset))
    if groups:
        parts = ax.violinplot(groups, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0"); pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_ylabel("Error (pred − real, meses)")
        ax.set_title("Distribución del error por grupo etario")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "violin_grupos_etarios.png"), dpi=150)
    plt.close()


def plot_within_threshold(bin_stats, output_dir):
    fig, ax = plt.subplots(figsize=(max(10, len(bin_stats) * 0.7), 5))
    x = range(len(bin_stats))
    ax.bar(x, bin_stats["within_24m"], label="±24 m", color="#AED6F1", edgecolor="white")
    ax.bar(x, bin_stats["within_12m"], label="±12 m", color="#2980B9", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.lo}–{r.hi}" for _, r in bin_stats.iterrows()],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Rango de edad (meses)")
    ax.set_ylabel("% imágenes dentro del umbral")
    ax.set_title("% predicciones dentro de ±12 m y ±24 m por rango")
    ax.set_ylim(0, 105); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_por_umbral.png"), dpi=150)
    plt.close()


def plot_heatmap(df_ok, output_dir, bin_size=12):
    edges = list(range(0, int(df_ok["real_age"].max()) + bin_size + 1, bin_size))
    real_bins = pd.cut(df_ok["real_age"], bins=edges, right=False)
    pred_bins = pd.cut(df_ok["pred_age"], bins=edges, right=False)
    matrix = pd.crosstab(real_bins, pred_bins)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.values, aspect="auto", cmap="Blues", origin="lower")
    plt.colorbar(im, ax=ax, label="Número de imágenes")
    step = max(1, len(edges) // 10)
    ticks = list(range(0, len(edges) - 1, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    tick_labels = [f"{edges[i]}" for i in ticks]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Edad predicha (meses)"); ax.set_ylabel("Edad real (meses)")
    ax.set_title("Heatmap: edad real vs edad predicha")
    ax.plot([0, len(edges)-2], [0, len(edges)-2], "r--", linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_real_vs_pred.png"), dpi=150)
    plt.close()


def plot_summary_table(bin_stats, pediatric_stats, global_mae, output_dir):
    # Top 3 mejor y peor
    top3_best  = bin_stats.nsmallest(3, "mae")[["bin_label", "n", "mae", "bias", "within_12m"]]
    top3_worst = bin_stats.nlargest(3, "mae")[["bin_label", "n", "mae", "bias", "within_12m"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"Análisis de rangos — MAE global: {global_mae:.2f} m", fontsize=13, fontweight="bold")

    for ax, df_t, title, color in [
        (axes[0], top3_best,  "Rangos más precisos (↓ MAE)", "#1E8449"),
        (axes[1], top3_worst, "Rangos menos precisos (↑ MAE)", "#922B21"),
        (axes[2], pediatric_stats[["label","n","mae","bias","within_12m"]].dropna(),
         "Grupos pediátricos", "#1A5276"),
    ]:
        ax.axis("off")
        df_t = df_t.copy()
        if "mae" in df_t.columns:
            df_t["mae"] = df_t["mae"].apply(lambda x: f"{x:.1f} m" if x else "—")
        if "bias" in df_t.columns:
            df_t["bias"] = df_t["bias"].apply(lambda x: f"{x:+.1f} m" if x else "—")
        if "within_12m" in df_t.columns:
            df_t["within_12m"] = df_t["within_12m"].apply(lambda x: f"{x:.0f}%" if x else "—")
        tbl = ax.table(cellText=df_t.values, colLabels=df_t.columns,
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(color); cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#BDC3C7")
        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resumen_rangos.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_distribution(df_ok, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df_ok["error"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].axvline(df_ok["error"].mean(), color="red", linestyle="--",
                    label=f'Media: {df_ok["error"].mean():.1f} m')
    axes[0].set_xlabel("Error (pred − real, meses)"); axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución del error"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(df_ok["abs_error"], bins=40, color="#E67E22", edgecolor="white", alpha=0.85)
    axes[1].axvline(df_ok["abs_error"].mean(), color="red", linestyle="--",
                    label=f'MAE: {df_ok["abs_error"].mean():.1f} m')
    axes[1].axvline(df_ok["abs_error"].median(), color="navy", linestyle=":",
                    label=f'Mediana: {df_ok["abs_error"].median():.1f} m')
    axes[1].set_xlabel("Error absoluto (meses)"); axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución del error absoluto"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribucion_error.png"), dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Análisis por rango de edad")
    parser.add_argument("--experiment", type=int, required=True)
    parser.add_argument("--bin-size", type=int, default=12,
                        help="Tamaño del intervalo en meses (default: 12)")
    parser.add_argument("--dataset", choices=["val", "mex", "both"], default="both",
                        help="Dataset a evaluar (default: both)")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_gpu()

    cfg = load_experiment_config(args.experiment)
    exp_dir = get_experiment_output_dir(args.experiment)
    output_dir = os.path.join(exp_dir, "age_range_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experimento {args.experiment} | bin={args.bin_size} m | dataset={args.dataset}")

    with timer("Carga de modelos"):
        seg_model, fusion_model = load_all_models(cfg, exp_dir)

    all_records = []

    if args.dataset in ("val", "both"):
        df_val = pd.read_csv(VALIDATION_CSV)
        if "boneage" not in df_val.columns:
            print("ADVERTENCIA: validación sin columna boneage, omitiendo.")
        else:
            print(f"Inferencia sobre validación estándar ({len(df_val)} imágenes)...")
            with timer("Inferencia validación"):
                all_records += run_inference(df_val, VALIDATION_IMAGES_DIR,
                                             seg_model, fusion_model, cfg, "val")

    if args.dataset in ("mex", "both"):
        df_mex = pd.read_csv(MEX_CSV)
        print(f"Inferencia sobre validación mexicana ({len(df_mex)} imágenes)...")
        with timer("Inferencia mex-validation"):
            all_records += run_inference(df_mex, MEX_IMAGES_DIR,
                                         seg_model, fusion_model, cfg, "mex")

    df_all = pd.DataFrame(all_records)
    df_ok  = df_all[df_all["status"] == "ok"].copy()
    n_failed = (df_all["status"] != "ok").sum()

    print(f"\nTotal: {len(df_all)} | OK: {len(df_ok)} | Fallidas: {n_failed}")

    # Guardar resultados por imagen
    df_all.to_csv(os.path.join(output_dir, "per_image_results.csv"), index=False)

    if df_ok.empty:
        print("Sin predicciones válidas. Abortando.")
        return

    global_mae  = df_ok["abs_error"].mean()
    global_bias = df_ok["error"].mean()
    global_rmse = np.sqrt((df_ok["error"]**2).mean())
    within_12   = (df_ok["abs_error"] <= 12).mean() * 100
    within_24   = (df_ok["abs_error"] <= 24).mean() * 100

    print(f"MAE global: {global_mae:.2f} m | Sesgo: {global_bias:+.2f} m | "
          f"RMSE: {global_rmse:.2f} m | ±12m: {within_12:.1f}% | ±24m: {within_24:.1f}%")

    with timer("Estadísticas por rango"):
        bin_stats      = compute_bin_stats(df_ok, args.bin_size)
        pediatric_stats = compute_pediatric_stats(df_ok)

    # Guardar JSONs
    bin_stats.to_json(os.path.join(output_dir, "bin_stats.json"), orient="records", indent=2)
    pediatric_stats.to_json(os.path.join(output_dir, "pediatric_stats.json"), orient="records", indent=2)

    summary = {
        "experiment": args.experiment,
        "bin_size": args.bin_size,
        "dataset": args.dataset,
        "n_total": len(df_all),
        "n_ok": len(df_ok),
        "n_failed": int(n_failed),
        "global_mae": round(global_mae, 2),
        "global_bias": round(global_bias, 2),
        "global_rmse": round(global_rmse, 2),
        "within_12m_pct": round(within_12, 1),
        "within_24m_pct": round(within_24, 1),
        "best_range": bin_stats.loc[bin_stats["mae"].idxmin(), "bin_label"],
        "worst_range": bin_stats.loc[bin_stats["mae"].idxmax(), "bin_label"],
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with timer("Generación de gráficos"):
        plot_mae_by_bin(bin_stats, output_dir, args.bin_size)
        plot_bias_by_bin(bin_stats, output_dir, args.bin_size)
        plot_scatter_colored(df_ok, output_dir)
        plot_violin_by_pediatric(df_ok, output_dir)
        plot_within_threshold(bin_stats, output_dir)
        plot_heatmap(df_ok, output_dir, args.bin_size)
        plot_error_distribution(df_ok, output_dir)
        plot_summary_table(bin_stats, pediatric_stats, global_mae, output_dir)

    print(f"\nResultados en: {output_dir}")
    print(f"Mejor rango:  {summary['best_range']}")
    print(f"Peor rango:   {summary['worst_range']}")


if __name__ == "__main__":
    _args = parse_args()
    _exp_dir = get_experiment_output_dir(_args.experiment)
    setup_logging("10_age_range_analysis.py", log_dir=_exp_dir)
    main()
