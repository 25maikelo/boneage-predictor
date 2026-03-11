#!/usr/bin/env python3
"""
Análisis de desempeño: tabla comparativa de modelos y mapas de saliencia.

Uso:
    python src/performance_analysis.py --experiment 26
"""
import os
import sys
import argparse
import logging
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

from tensorflow.keras.models import load_model

from config.paths import BALANCED_DATASET_CSV, SEGMENTED_IMAGES_DIR, EQUALIZED_IMAGES_DIR
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.models.losses import LOSS_MAP, dynamic_attention_loss
from src.utils.timing import report_timing, setup_logging, timer

START_TIME = time.time()

SEGMENTS_TRANSLATION = {
    "thumb": "Pulgar", "middle": "Medio",
    "pinky": "Meñique", "wrist": "Muñeca"
}


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


def normalize_image(img):
    return img.astype("float32") / 255.0


def safe_load_image(path, image_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    else:
        img = cv2.resize(img, image_size)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


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
    thresh = np.percentile(heatmap, 97)
    mask = heatmap >= thresh
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    result[mask3] = result[mask3] * (1 - alpha) + colored[mask3] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def create_sample_table(text, image, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].axis("off")
    axes[0].text(0.5, 0.5, text, fontsize=12, ha="center", va="center", wrap=True)
    axes[1].axis("off"); axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)


def generate_comparative_table(fusion_model, segment_models, cfg, evaluation_path):
    df = pd.read_csv(BALANCED_DATASET_CSV)
    tr_df, va_df = df.head(32), df.tail(32)

    def load_seg_batch(sub_df):
        inputs = []
        for seg in cfg.SEGMENTS_ORDER:
            inputs.append(np.stack([
                normalize_image(safe_load_image(
                    os.path.join(SEGMENTED_IMAGES_DIR, seg, f"{r}.png"), cfg.IMAGE_SIZE))
                for r in sub_df["id"]
            ]))
        if cfg.USE_GENDER:
            inputs.append(sub_df["male"].fillna(0).values.reshape(-1, 1))
        return inputs

    tr_in, va_in = load_seg_batch(tr_df), load_seg_batch(va_df)
    tr_t = tr_df["boneage"].astype("float32").values
    va_t = va_df["boneage"].astype("float32").values

    rows = []
    tr_loss, tr_mae = fusion_model.evaluate(tr_in, tr_t, verbose=0)
    va_loss, va_mae = fusion_model.evaluate(va_in, va_t, verbose=0)
    rows.append(["Fusionado", f"{fusion_model.count_params():,}",
                 f"{tr_loss:.4f}", f"{tr_mae:.4f}", f"{va_loss:.4f}", f"{va_mae:.4f}"])

    for seg, model in segment_models.items():
        idx = cfg.SEGMENTS_ORDER.index(seg)
        inp_tr = [tr_in[idx]] + ([tr_in[-1]] if cfg.USE_GENDER else [])
        inp_va = [va_in[idx]] + ([va_in[-1]] if cfg.USE_GENDER else [])
        tr_loss, tr_mae = model.evaluate(inp_tr, tr_t, verbose=0)
        va_loss, va_mae = model.evaluate(inp_va, va_t, verbose=0)
        rows.append([SEGMENTS_TRANSLATION.get(seg, seg), f"{model.count_params():,}",
                     f"{tr_loss:.4f}", f"{tr_mae:.4f}", f"{va_loss:.4f}", f"{va_mae:.4f}"])

    cols = ["Modelo", "Parámetros", "Loss Train", "MAE Train", "Loss Val", "MAE Val"]
    df_res = pd.DataFrame(rows, columns=cols)

    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.4 * len(df_res)))
    fig.patch.set_visible(False); ax.axis("off")
    tbl = ax.table(cellText=df_res.values, colLabels=cols, cellLoc="center", loc="center")
    tbl.scale(1, 2)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f1f1f2" if i % 2 == 0 else "#e0e0e0")
        cell.set_edgecolor("black")
    plt.savefig(os.path.join(evaluation_path, "tabla_comparativa.png"),
                bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    logging.info("Tabla comparativa guardada.")


def parse_args():
    parser = argparse.ArgumentParser(description="Análisis de desempeño")
    parser.add_argument("--experiment", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_gpu()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_experiment_config(args.experiment)
    exp_dir = get_experiment_output_dir(args.experiment)
    evaluation_path = os.path.join(exp_dir, "evaluation")
    os.makedirs(evaluation_path, exist_ok=True)

    models_dir = os.path.join(exp_dir, "models")

    with timer("Carga de modelos"):
        fusion_model = load_model(
            os.path.join(models_dir, "fusion_model.keras"),
            custom_objects=LOSS_MAP, safe_mode=False
        )
        fusion_model.compile(
            optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
            loss=LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss),
            metrics=["mae"]
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

    # Muestras con saliencia
    df = pd.read_csv(BALANCED_DATASET_CSV)
    sample_df = df.sample(n=3, random_state=42)
    with timer("Generación de saliencias"):
        for _, row in sample_df.iterrows():
            sid = str(row["id"])
            real_age = row["boneage"]
            fusion_inputs = [
                tf.expand_dims(normalize_image(safe_load_image(
                    os.path.join(SEGMENTED_IMAGES_DIR, seg, f"{sid}.png"), cfg.IMAGE_SIZE)), 0)
                for seg in cfg.SEGMENTS_ORDER
            ]
            if cfg.USE_GENDER:
                fusion_inputs.append(tf.constant([[float(row.get("male", 0))]], tf.float32))
            pred_age = float(fusion_model.predict(fusion_inputs, verbose=0).flatten()[0])
            diff = abs(pred_age - real_age)

            base_img = safe_load_image(os.path.join(EQUALIZED_IMAGES_DIR, f"{sid}.png"),
                                        cfg.IMAGE_SIZE)
            result = base_img.copy()
            for seg, model in segment_models.items():
                heat = compute_saliency_map(
                    model,
                    [tf.expand_dims(normalize_image(safe_load_image(
                        os.path.join(SEGMENTED_IMAGES_DIR, seg, f"{sid}.png"), cfg.IMAGE_SIZE)), 0)]
                    + ([tf.constant([[float(row.get("male", 0))]], tf.float32)] if cfg.USE_GENDER else [])
                )
                result = overlay_masked(result, heat)

            text = f"ID: {sid}\nReal: {real_age:.1f} m\nPredicción: {pred_age:.1f} m\nDiferencia: {diff:.1f} m"
            create_sample_table(text, result,
                                os.path.join(evaluation_path, f"sample_result_{sid}.png"))

    with timer("Tabla comparativa"):
        generate_comparative_table(fusion_model, segment_models, cfg, evaluation_path)

    logging.info("===== ANÁLISIS COMPLETADO =====")


if __name__ == "__main__":
    setup_logging("09_performance_analysis.py")
    main()
    report_timing(START_TIME, "09_performance_analysis.py")
