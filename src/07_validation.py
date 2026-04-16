#!/usr/bin/env python3
"""
Validación del modelo sobre el dataset de validación estándar.
Genera mapas de saliencia, gráficos y tabla resumen.

Uso:
    python src/validation.py --experiment 26
"""
import os
import sys
import argparse
import gc
import logging
import random
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from collections import defaultdict
from matplotlib import cm
from tensorflow.keras.models import load_model


from config.paths import VALIDATION_CSV, VALIDATION_IMAGES_DIR, EXPERIMENTS_DIR
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.utils.losses import LOSS_MAP, dynamic_attention_loss
from src.utils.timing import report_timing, setup_logging, timer

START_TIME = time.time()
NUM_SAMPLE_RESULTS = 3


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {[g.name for g in gpus]}")
    else:
        print("ADVERTENCIA: No se detectó GPU. Ejecutando en CPU.")


def get_optimizer(choice, lr):
    if choice.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def frame_and_zoom(img, kernel_size=(9, 9), dilation_iters=4):
    """Rotación y recorte — lógica de src/preprocessing/02_frame_and_zoom.py."""
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
    """Ecualización CLAHE — lógica de src/preprocessing/03_histogram_equalization.py."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    blended = cv2.addWeighted(img, 0.6, eq, 0.4, 0)
    return cv2.convertScaleAbs(blended, alpha=0.9, beta=-10)


def segment_spatial(img, model, segments_order):
    h0, w0 = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(rgb, (224, 224)) / 255.0
    pred = model.predict(inp[np.newaxis, ...], verbose=0)[0]
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


def compute_saliency_map(model, inputs):
    img_t = inputs[0]
    with tf.GradientTape() as tape:
        tape.watch(img_t)
        preds = model(inputs, training=False)
        score = tf.reshape(preds, [-1])[0]
    grads = tape.gradient(score, img_t)[0]
    sal = tf.reduce_max(tf.abs(grads), axis=-1)
    return (sal / (tf.reduce_max(sal) + 1e-8)).numpy()


def overlay_masked(base_img, heatmap, alpha=0.4):
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.resize(colored, (base_img.shape[1], base_img.shape[0])).astype(np.float32)
    result = base_img.astype(np.float32)
    th = np.percentile(heatmap, 97)
    mask = heatmap >= th
    for c in range(3):
        result[:, :, c][mask] = result[:, :, c][mask] * (1 - alpha) + colored[:, :, c][mask] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def create_sample_table(text, img, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].axis("off"); ax[0].text(0.5, 0.5, text, ha="center", va="center", fontsize=16, wrap=True)
    ax[1].axis("off"); ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(fig)


def parse_age_to_months(age_str):
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


def _load_seg_model_compat(path):
    return load_model(path, compile=False)


def load_all_models(cfg, exp_dir):
    models_dir = os.path.join(exp_dir, "models")
    from config.paths import get_segmentation_model_path
    cfg_seg = getattr(cfg, "SEGMENTATION_MODEL", None)
    if cfg_seg:
        # Si es un nombre de run (sin separadores de ruta), resuelve via helper
        if os.sep not in cfg_seg and "/" not in cfg_seg and not cfg_seg.endswith(".h5"):
            seg_path = get_segmentation_model_path(cfg_seg)
        else:
            seg_path = cfg_seg if os.path.isabs(cfg_seg) else os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg_seg
            )
    else:
        seg_path = get_segmentation_model_path()
    print(f"Modelo de segmentación: {seg_path}")
    seg_model = _load_seg_model_compat(seg_path)
    segment_models = {}
    for seg in cfg.SEGMENTS_ORDER:
        path = os.path.join(models_dir, f"{seg}_model")
        if os.path.exists(path):
            m = load_model(path, custom_objects=LOSS_MAP)
            m.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
                      loss=LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss),
                      metrics=["mae"])
            segment_models[seg] = m
    fusion = load_model(os.path.join(models_dir, "fusion_model"),
                        custom_objects=LOSS_MAP)
    fusion.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
                   loss=LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss),
                   metrics=["mae"])
    return seg_model, segment_models, fusion


def parse_args():
    parser = argparse.ArgumentParser(description="Validación bone age predictor")
    parser.add_argument("--experiment", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_gpu()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_experiment_config(args.experiment)
    exp_dir = get_experiment_output_dir(args.experiment)
    OUTPUT_FOLDER = os.path.join(exp_dir, "validation")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Cargando modelos del experimento {args.experiment}...")
    with timer("Carga de modelos"):
        seg_model, segment_models, fusion_model = load_all_models(cfg, exp_dir)

    df = pd.read_csv(VALIDATION_CSV)
    df.reset_index(drop=True, inplace=True)
    if "boneage" in df.columns:
        df["boneage"] = df["boneage"].apply(parse_age_to_months)
    max_samples = getattr(cfg, "MAX_SAMPLES", None)
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
        print(f"[Quick test] Limitando validación a {len(df)} muestras.")

    # Guardar edades para regeneración multiidioma
    _plot_data = {"ages": df["boneage"].dropna().tolist()}

    # Histograma de edades
    plt.figure(figsize=(7, 4))
    df["boneage"].hist(bins=20, color="#5b8ff9", edgecolor="k", alpha=0.8)
    plt.xlabel("Edad (meses)", fontsize=14); plt.ylabel("Frecuencia", fontsize=14)
    plt.title("Distribución de Edad (Dataset de Validación)", fontsize=16)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "histograma_edad_validacion.png")); plt.close()

    # Pastel de distribución de sexo
    if "male" in df.columns:
        gender_map = {"TRUE": "Masculino", "FALSE": "Femenino", True: "Masculino", False: "Femenino"}
        gender_labels = df["male"].map(gender_map)
        gender_counts = gender_labels.value_counts()
        _plot_data["gender"] = gender_counts.to_dict()
        colors = [cm.tab10(0), cm.tab10(1)]
        plt.figure(figsize=(5, 5))
        patches, texts, autotexts = plt.pie(
            gender_counts.values, labels=gender_counts.index,
            autopct="%1.1f%%", startangle=90, colors=colors, textprops={"fontsize": 14}
        )
        for t in texts: t.set_fontsize(16)
        for at in autotexts: at.set_fontsize(15); at.set_color("white"); at.set_weight("bold")
        plt.title("Distribución de Sexo (Validación)", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "sexo_pastel_validacion.png")); plt.close()

    rows, preds, trues, failed = [], [], [], []
    times_log = defaultdict(list)
    # Reservoir sampling: guarda solo NUM_SAMPLE_RESULTS candidatos en memoria
    candidates = []
    n_seen = 0

    with timer("Inferencia sobre dataset de validación"):
        for i, (_, row) in enumerate(df.iterrows()):
            sid = str(row["id"])
            real_age = float(row["boneage"])
            t0 = time.time()
            img_path = os.path.join(VALIDATION_IMAGES_DIR, f"{sid}.png")
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                failed.append((sid, "not_found")); continue
            t1 = time.time()
            try:
                zoomed = frame_and_zoom(gray)
                eq = clahe_equalize(zoomed)
            except Exception as e:
                failed.append((sid, f"preprocess: {e}")); continue
            t2 = time.time()
            try:
                segments = segment_spatial(eq, seg_model, cfg.SEGMENTS_ORDER)
                if any(np.sum(s) == 0 for s in segments.values()):
                    failed.append((sid, "empty_segment")); continue
            except Exception as e:
                failed.append((sid, f"segment: {e}")); continue
            t3 = time.time()
            try:
                fusion_inputs = [
                    tf.expand_dims(normalize_image(cv2.resize(segments[seg], cfg.IMAGE_SIZE,
                                                               interpolation=cv2.INTER_CUBIC)), 0)
                    for seg in cfg.SEGMENTS_ORDER
                ]
                if cfg.USE_GENDER:
                    fusion_inputs.append(tf.constant([[float(row.get("male", 0))]], tf.float32))
                pred_age = float(fusion_model.predict(fusion_inputs, verbose=0).flatten()[0])
            except Exception as e:
                failed.append((sid, f"predict: {e}")); continue
            t4 = time.time()

            preds.append(pred_age); trues.append(real_age)
            times_log["preprocess"].append(t2 - t1)
            times_log["segment"].append(t3 - t2)
            times_log["predict"].append(t4 - t3)
            rows.append({"id": sid, "real_age": real_age, "pred_age": pred_age,
                         "diff_months": abs(pred_age - real_age), "status": "ok"})

            # Reservoir sampling — mantiene solo NUM_SAMPLE_RESULTS en memoria
            n_seen += 1
            candidate = {"sid": sid, "eq": eq, "segments": segments,
                         "row": row, "real_age": real_age, "pred_age": pred_age}
            if len(candidates) < NUM_SAMPLE_RESULTS:
                candidates.append(candidate)
            else:
                j = random.randint(0, n_seen - 1)
                if j < NUM_SAMPLE_RESULTS:
                    candidates[j] = candidate

            if i % 100 == 0:
                gc.collect()

    chosen = candidates
    for item in chosen:
        sid, eq, segments, row = item["sid"], item["eq"], item["segments"], item["row"]
        base = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        result = base.copy()
        for seg, model in segment_models.items():
            seg_r = cv2.resize(segments[seg], cfg.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
            tensor = tf.expand_dims(normalize_image(seg_r), 0)
            inp = [tensor]
            if cfg.USE_GENDER and len(model.inputs) > 1:
                inp.append(tf.constant([[float(row.get("male", 0))]], dtype=tf.float32))
            heat = compute_saliency_map(model, inp)
            heat = cv2.resize(heat, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
            result = overlay_masked(result, heat)
        text = (f"ID: {sid}\nReal: {item['real_age']:.1f} m\n"
                f"Predicción: {item['pred_age']:.1f} m\n"
                f"Diferencia: {abs(item['pred_age'] - item['real_age']):.1f} m")
        create_sample_table(text, result, os.path.join(OUTPUT_FOLDER, f"sample_result_{sid}.png"))

    mae = np.mean(np.abs(np.array(preds) - np.array(trues))) if preds else np.nan

    # Guardar datos de scatter y resumen para regeneración multiidioma
    _plot_data["scatter"] = {"trues": [float(v) for v in trues], "preds": [float(v) for v in preds]}
    _plot_data["summary"] = {
        "processed": len(rows), "failed": len(failed), "mae": float(mae) if preds else None,
        "time_preprocess": float(np.mean(times_log["preprocess"])) if times_log["preprocess"] else None,
        "time_segment":    float(np.mean(times_log["segment"]))    if times_log["segment"]    else None,
        "time_predict":    float(np.mean(times_log["predict"]))    if times_log["predict"]    else None,
    }
    import json as _json
    with open(os.path.join(OUTPUT_FOLDER, "plot_data.json"), "w") as _f:
        _json.dump(_plot_data, _f, indent=2)

    # Tabla resumen
    summary_data = [
        ["Métrica", "Valor"],
        ["Imágenes procesadas", str(len(rows))],
        ["Imágenes fallidas", str(len(failed))],
        ["MAE (meses)", f"{mae:.2f}"],
        ["Tiempo medio preprocess (s)",
         f"{np.mean(times_log['preprocess']):.2f}" if times_log["preprocess"] else "N/A"],
        ["Tiempo medio segmentación (s)",
         f"{np.mean(times_log['segment']):.2f}" if times_log["segment"] else "N/A"],
        ["Tiempo medio predicción (s)",
         f"{np.mean(times_log['predict']):.2f}" if times_log["predict"] else "N/A"],
    ]
    fig, ax = plt.subplots(figsize=(9, 3)); ax.axis("off")
    tbl = ax.table(cellText=summary_data, cellLoc="center", loc="center",
                   colWidths=[0.36, 0.24])
    tbl.auto_set_font_size(False); tbl.set_fontsize(16)
    for i in range(len(summary_data)):
        for j in range(2):
            cell = tbl[(i, j)]
            cell.set_facecolor("#4062bb" if i == 0 else ("#dbe2ef" if i % 2 else "#f9f7f7"))
            if i == 0:
                cell.set_text_props(weight="bold", color="white")
            cell.set_edgecolor("#112d4e")
    tbl.scale(1.3, 1.7)
    plt.title("Resumen de Validación", fontsize=19, weight="bold", color="#112d4e", pad=18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "validation_summary.png"), bbox_inches="tight")
    plt.close(fig)

    if preds:
        plt.figure(figsize=(7, 5))
        plt.scatter(trues, preds, c="#325288", alpha=0.7, edgecolors="k", s=80)
        plt.xlabel("Edad real (meses)"); plt.ylabel("Predicción (meses)")
        plt.title("Dispersión: Edad Real vs Predicción")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "scatter_pred_vs_real.png")); plt.close()

    print(f"MAE: {mae:.2f} meses | Procesadas: {len(rows)} | Fallidas: {len(failed)}")


if __name__ == "__main__":
    _args = parse_args()
    _exp_dir = get_experiment_output_dir(_args.experiment)
    setup_logging("07_validation.py", log_dir=_exp_dir)
    main()
    report_timing(START_TIME, "07_validation.py")
