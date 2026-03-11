#!/usr/bin/env python3
"""
Validación del modelo sobre el dataset de validación mexicano.

Uso:
    python src/mex_validation.py --experiment 26
"""
import os
import sys
import argparse
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

from matplotlib import cm
from tensorflow.keras.models import load_model

from config.paths import MEX_CSV, MEX_IMAGES_DIR, EXPERIMENTS_DIR
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.models.losses import LOSS_MAP, dynamic_attention_loss
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


def frame_and_zoom(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = thresh.shape
    if (np.any(thresh[0] == 255) or np.any(thresh[-1] == 255) or
            np.any(thresh[:, 0] == 255) or np.any(thresh[:, -1] == 255)):
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(thresh, kernel, iterations=3)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    box = np.intp(cv2.boxPoints(rect))
    angle = rect[-1]
    if rect[1][0] < rect[1][1]:
        angle += 90
    if angle > 45:
        angle -= 90
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    pts = np.hstack([box, np.ones((4, 1))])
    pts2 = np.dot(M, pts.T).T.astype(int)
    x0, y0 = pts2[:, 0].min(), pts2[:, 1].min()
    x1, y1 = pts2[:, 0].max(), pts2[:, 1].max()
    if x1 <= x0 or y1 <= y0:
        return img
    crop = rot[y0:y1, x0:x1]
    return crop if crop.size >= 0.2 * img.size else img


def clahe_equalize(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    return cv2.convertScaleAbs(cv2.addWeighted(img, 0.6, eq, 0.4, 0), alpha=0.9, beta=-10)


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


def load_all_models(cfg, exp_dir):
    models_dir = os.path.join(exp_dir, "models")
    seg_path = os.path.join(models_dir, "modelo_segmentacion.h5")
    from config.paths import SEGMENTATION_MODEL_PATH
    seg_model = load_model(
        seg_path if os.path.exists(seg_path) else SEGMENTATION_MODEL_PATH,
        compile=False
    )
    segment_models = {}
    for seg in cfg.SEGMENTS_ORDER:
        path = os.path.join(models_dir, f"{seg}_model.keras")
        if os.path.exists(path):
            m = load_model(path, custom_objects=LOSS_MAP, safe_mode=False)
            m.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
                      loss=LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss),
                      metrics=["mae"])
            segment_models[seg] = m
    fusion = load_model(os.path.join(models_dir, "fusion_model.keras"),
                        custom_objects=LOSS_MAP, safe_mode=False)
    fusion.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
                   loss=LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss),
                   metrics=["mae"])
    return seg_model, segment_models, fusion


def parse_args():
    parser = argparse.ArgumentParser(description="Validación dataset mexicano")
    parser.add_argument("--experiment", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_gpu()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_experiment_config(args.experiment)
    exp_dir = get_experiment_output_dir(args.experiment)
    OUTPUT_FOLDER = os.path.join(exp_dir, "mex-validation")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Cargando modelos del experimento {args.experiment}...")
    with timer("Carga de modelos"):
        seg_model, segment_models, fusion_model = load_all_models(cfg, exp_dir)

    df = pd.read_csv(MEX_CSV)
    df.reset_index(drop=True, inplace=True)
    df["real_age"] = df["real_age"].apply(parse_age_to_months)
    df["bone_age"] = df["bone_age"].apply(parse_age_to_months)

    rows, preds, trues, failed = [], [], [], []

    with timer("Inferencia sobre dataset mexicano"):
        for _, row in df.iterrows():
            sid = str(row["ID"])
            real_age = float(row["real_age"])
            img_path = os.path.join(MEX_IMAGES_DIR, f"{sid}.png")
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                failed.append((sid, "not_found")); continue
            try:
                zoomed = frame_and_zoom(gray)
                eq = clahe_equalize(zoomed)
                segments = segment_spatial(eq, seg_model, cfg.SEGMENTS_ORDER)
                if any(np.sum(s) == 0 for s in segments.values()):
                    failed.append((sid, "empty_segment")); continue
            except Exception as e:
                failed.append((sid, f"preprocess: {e}")); continue
            try:
                fusion_inputs = [
                    tf.expand_dims(normalize_image(
                        cv2.resize(segments[seg], cfg.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)), 0)
                    for seg in cfg.SEGMENTS_ORDER
                ]
                if cfg.USE_GENDER and "gender" in row:
                    fusion_inputs.append(
                        tf.constant([[1.0 if row["gender"] == "M" else 0.0]], tf.float32))
                pred_age = float(fusion_model.predict(fusion_inputs, verbose=0).flatten()[0])
            except Exception as e:
                failed.append((sid, f"predict: {e}")); continue

            if not np.isnan(pred_age) and not np.isnan(real_age):
                preds.append(pred_age); trues.append(real_age)
                rows.append({"id": sid, "real_age": real_age, "pred_age": pred_age,
                             "diff_months": abs(pred_age - real_age),
                             "segments": segments, "eq": eq, "row": row})
            else:
                failed.append((sid, "nan"))

    mae = np.mean(np.abs(np.array(preds) - np.array(trues))) if preds else np.nan

    # Tabla resumen
    summary_data = [["Métrica", "Valor"],
                    ["Imágenes procesadas", str(len(rows))],
                    ["Imágenes fallidas", str(len(failed))],
                    ["MAE (meses)", f"{mae:.2f}"]]
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
    plt.title("Resumen de Validación (MEX)", fontsize=19, weight="bold", color="#112d4e", pad=18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "validation_summary.png"), bbox_inches="tight")
    plt.close(fig)

    # Muestras con saliencia
    chosen = random.sample(rows, min(NUM_SAMPLE_RESULTS, len(rows)))
    for item in chosen:
        sid, eq, segments, df_row = item["id"], item["eq"], item["segments"], item["row"]
        base = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        result = base.copy()
        for seg, model in segment_models.items():
            seg_r = cv2.resize(segments[seg], cfg.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
            tensor = tf.expand_dims(normalize_image(seg_r), 0)
            inp = [tensor]
            if cfg.USE_GENDER and "gender" in df_row:
                inp.append(tf.constant([[1.0 if df_row["gender"] == "M" else 0.0]], tf.float32))
            heat = compute_saliency_map(model, inp)
            heat = cv2.resize(heat, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
            result = overlay_masked(result, heat)
        text = (f"ID: {sid}\nReal: {item['real_age']:.1f} m\n"
                f"Predicción: {item['pred_age']:.1f} m\n"
                f"Diferencia: {item['diff_months']:.1f} m")
        create_sample_table(text, result, os.path.join(OUTPUT_FOLDER, f"sample_result_{sid}.png"))

    if preds:
        plt.figure(figsize=(7, 5))
        plt.scatter(trues, preds, c="#325288", alpha=0.7, edgecolors="k", s=80)
        plt.xlabel("Edad real (meses)"); plt.ylabel("Predicción (meses)")
        plt.title("Dispersión: Edad Real vs Predicción (MEX)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "scatter_pred_vs_real.png")); plt.close()

    print(f"MAE: {mae:.2f} meses | Procesadas: {len(rows)} | Fallidas: {len(failed)}")


if __name__ == "__main__":
    setup_logging("08_mex_validation.py")
    main()
    report_timing(START_TIME, "08_mex_validation.py")
