#!/usr/bin/env python3
"""
Paso 1: Entrenamiento del detector de mano (segmentación semántica).
Lee anotaciones en formato LabelMe (un JSON por imagen, mismo nombre que la PNG)
y entrena un modelo de segmentación de 5 clases:
  0=fondo, 1=pinky, 2=middle, 3=thumb, 4=wrist

La arquitectura y todos los hiperparámetros se configuran en config/segmentation.py.

Uso:
    python src/preprocessing/01_train_hand_detector.py
"""
import os
import sys
import time
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pandas as pd
import dataframe_image as dfi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
tf.get_logger().setLevel("ERROR")

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from config.paths import (
    HAND_DETECTOR_IMAGES_DIR,
    HAND_DETECTOR_ANNOTATIONS_DIR,
    HAND_DETECTOR_OUTPUT_DIR,
    SEGMENTATION_MODEL_PATH,
)
from config.segmentation import (
    IMAGE_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS, INPUT_CHANNELS,
    ARCHITECTURE, ENCODER_WEIGHTS, BASE_MODEL_TRAINABLE,
    DATA_AUGMENTATION,
    AUG_ROTATION_RANGE, AUG_WIDTH_SHIFT_RANGE, AUG_HEIGHT_SHIFT_RANGE,
    AUG_ZOOM_RANGE, AUG_HORIZONTAL_FLIP, AUG_VERTICAL_FLIP,
    AUG_SHEAR_RANGE, AUG_BRIGHTNESS_RANGE,
)
from src.utils.timing import report_timing, setup_logging, timer

START_TIME = time.time()


# ============================================================
# CARPETA DE RUN SECUENCIAL
# ============================================================
def get_run_dir():
    existing = glob.glob(os.path.join(HAND_DETECTOR_OUTPUT_DIR, "hand-detector_[0-9][0-9]"))
    n = max((int(os.path.basename(d).split("_")[-1]) for d in existing), default=-1) + 1
    run_dir = os.path.join(HAND_DETECTOR_OUTPUT_DIR, f"hand-detector_{n:02d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir):
    uses_backbone = ARCHITECTURE in ("unet_mobilenetv2", "mobilenetv2")
    config = {
        "IMAGE_SIZE": IMAGE_SIZE,
        "INPUT_CHANNELS": INPUT_CHANNELS,
        "NUM_CLASSES": NUM_CLASSES,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "ARCHITECTURE": ARCHITECTURE,
        "ENCODER_WEIGHTS": ENCODER_WEIGHTS if uses_backbone else None,
        "BASE_MODEL_TRAINABLE": BASE_MODEL_TRAINABLE if uses_backbone else None,
        "DATA_AUGMENTATION": DATA_AUGMENTATION,
        "AUG_ROTATION_RANGE": AUG_ROTATION_RANGE if DATA_AUGMENTATION else None,
        "AUG_WIDTH_SHIFT_RANGE": AUG_WIDTH_SHIFT_RANGE if DATA_AUGMENTATION else None,
        "AUG_HEIGHT_SHIFT_RANGE": AUG_HEIGHT_SHIFT_RANGE if DATA_AUGMENTATION else None,
        "AUG_ZOOM_RANGE": AUG_ZOOM_RANGE if DATA_AUGMENTATION else None,
        "AUG_HORIZONTAL_FLIP": AUG_HORIZONTAL_FLIP if DATA_AUGMENTATION else None,
        "AUG_VERTICAL_FLIP": AUG_VERTICAL_FLIP if DATA_AUGMENTATION else None,
        "AUG_SHEAR_RANGE": AUG_SHEAR_RANGE if DATA_AUGMENTATION else None,
        "AUG_BRIGHTNESS_RANGE": list(AUG_BRIGHTNESS_RANGE) if DATA_AUGMENTATION and AUG_BRIGHTNESS_RANGE else None,
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "early_stopping_patience": 5,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 1,
        "reduce_lr_min_lr": 1e-6,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Configuración guardada en: {run_dir}/config.json")


# ============================================================
# GPU
# ============================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {[g.name for g in gpus]}")
    else:
        print("ADVERTENCIA: No se detectó GPU. Ejecutando en CPU.")


# ============================================================
# CARGA DE DATOS (formato LabelMe, imágenes y JSONs en dirs separados)
# ============================================================
def load_data(image_size=IMAGE_SIZE):
    images, masks = [], []
    image_paths = glob.glob(os.path.join(HAND_DETECTOR_IMAGES_DIR, "*.png"))
    skipped = 0
    for image_path in sorted(image_paths):
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(HAND_DETECTOR_ANNOTATIONS_DIR, f"{img_id}.json")
        if not os.path.exists(json_path):
            skipped += 1
            continue

        original_img = cv2.imread(image_path)
        if original_img is None:
            skipped += 1
            continue
        original_size = original_img.shape[:2]  # (h, w)

        if INPUT_CHANNELS == 1:
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, image_size)
            img = np.expand_dims(img, axis=-1)   # (H, W, 1)
        else:
            img = cv2.resize(original_img, image_size)  # (H, W, 3)
        images.append(img)

        mask = np.zeros(image_size, dtype=np.uint8)
        with open(json_path, encoding="utf-8") as f:
            annotations = json.load(f)
        for shape in annotations.get("shapes", []):
            label = shape["label"].lower().strip()
            points = np.array(shape["points"], dtype=np.float32)
            scale_x = image_size[1] / original_size[1]
            scale_y = image_size[0] / original_size[0]
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            points = points.astype(np.int32)
            class_id = {"pinky": 1, "middle": 2, "thumb": 3, "wrist": 4}.get(label, 0)
            cv2.fillPoly(mask, [points], class_id)
        masks.append(mask)

    print(f"  Muestras cargadas: {len(images)}  |  Omitidas: {skipped}")
    images = np.array(images) / 255.0
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)
    return images, masks


# ============================================================
# ARQUITECTURAS
# ============================================================

# ── U-Net puro (sin backbone preentrenado) ────────────────────────────────────
def build_unet(input_size=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    """U-Net estándar entrenada desde cero. ENCODER_WEIGHTS no aplica."""
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(p4)
    bn = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(bn)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(bn)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c7)

    u8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(c9)
    return models.Model(inputs=inputs, outputs=outputs)


# ── U-Net + MobileNetV2 encoder (skip connections) ───────────────────────────
def build_unet_mobilenetv2(input_size=(*IMAGE_SIZE, INPUT_CHANNELS), num_classes=NUM_CLASSES):
    """U-Net con encoder MobileNetV2 y skip connections. Soporta imagenet.
    Si INPUT_CHANNELS=1, el canal se replica 3 veces antes del backbone.
    """
    h, w, channels = input_size
    weights = None if ENCODER_WEIGHTS in (None, "None") else ENCODER_WEIGHTS

    model_input = layers.Input((h, w, channels))
    if channels == 1:
        # Replicar el canal para que MobileNetV2 e imagenet sean compatibles
        backbone_in = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(model_input)
    else:
        backbone_in = model_input

    base_model = MobileNetV2(input_shape=(h, w, 3), include_top=False, weights=weights)
    base_model.trainable = BASE_MODEL_TRAINABLE

    # Sub-modelo multi-salida para extraer los skip connections
    skip_model = models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("block_1_expand_relu").output,   # 112×112
            base_model.get_layer("block_3_expand_relu").output,   # 56×56
            base_model.get_layer("block_6_expand_relu").output,   # 28×28
            base_model.get_layer("block_13_expand_relu").output,  # 14×14
            base_model.get_layer("block_16_project").output,      # 7×7
        ],
    )
    s1, s2, s3, s4, bottleneck = skip_model(backbone_in)

    x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(bottleneck)
    x = layers.concatenate([x, s4])
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.concatenate([x, s3])
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.concatenate([x, s2])
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.concatenate([x, s1])
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x)
    return models.Model(inputs=model_input, outputs=outputs)



# ── MobileNetV2 encoder + decoder simétrico inverted-residual ─────────────────
def _mnv2_decoder_block(x, skip, out_filters, expansion=6):
    """Bloque decoder simétrico al inverted-residual de MobileNetV2.

    Encoder (inverted residual):  expand 1×1 → depthwise 3×3 stride↓2 → project 1×1
    Decoder (simétrico):          expand 1×1 → upsample 2× + depthwise 3×3 → concat skip → project 1×1

    El UpSampling2D + DepthwiseConv2D reemplaza el stride depthwise del encoder:
    ambos operan sobre el mapa expandido en canales antes de proyectar al tamaño final.
    El skip connection se inserta después del upsample (misma resolución que el encoder)
    y antes del project, igual que en U-Net pero integrado dentro del bloque.
    """
    in_ch  = x.shape[-1]
    exp_ch = in_ch * expansion

    # Expand (misma operación que el encoder)
    y = layers.Conv2D(exp_ch, (1, 1), padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU(max_value=6.0)(y)

    # Upsample + Depthwise (invierte el stride=2 depthwise del encoder)
    y = layers.UpSampling2D(size=(2, 2))(y)
    y = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU(max_value=6.0)(y)

    # Skip connection integrado (concatenar antes del project)
    if skip is not None:
        y = layers.concatenate([y, skip])

    # Project (misma operación que el encoder)
    y = layers.Conv2D(out_filters, (1, 1), padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    return y


def build_mobilenetv2_sym(input_size=(*IMAGE_SIZE, INPUT_CHANNELS), num_classes=NUM_CLASSES):
    """MobileNetV2 encoder + decoder simétrico inverted-residual.

    El decoder espeja la estructura interna del encoder bloque a bloque:
    cada etapa del decoder usa _mnv2_decoder_block (expand → upsample+depthwise → project)
    en lugar de Conv2DTranspose + Conv2D convencionales.

    Schedule de canales (espejo del encoder):
        Encoder: 3 → 32(stem) → 16 → 24 → 32 → 64 → 96 → 160 → 320 (bottleneck)
        Decoder:                 16 ← 32 ← 64 ← 96 ← 160 ←         320

    Skip connections: mismos puntos que build_unet_mobilenetv2
        s1: block_1_expand_relu   112×112 × 96
        s2: block_3_expand_relu    56×56  × 144
        s3: block_6_expand_relu    28×28  × 192
        s4: block_13_expand_relu   14×14  × 576

    Si INPUT_CHANNELS=1, el canal se replica ×3 antes del backbone.
    """
    h, w, channels = input_size
    weights = None if ENCODER_WEIGHTS in (None, "None") else ENCODER_WEIGHTS

    model_input = layers.Input((h, w, channels))
    if channels == 1:
        backbone_in = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(model_input)
    else:
        backbone_in = model_input

    # ── Encoder: MobileNetV2 preentrenado ─────────────────────────────────────
    base_model = MobileNetV2(input_shape=(h, w, 3), include_top=False, weights=weights)
    base_model.trainable = BASE_MODEL_TRAINABLE

    skip_model = models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("block_1_expand_relu").output,   # 112×112 × 96
            base_model.get_layer("block_3_expand_relu").output,   # 56×56  × 144
            base_model.get_layer("block_6_expand_relu").output,   # 28×28  × 192
            base_model.get_layer("block_13_expand_relu").output,  # 14×14  × 576
            base_model.get_layer("block_16_project").output,      # 7×7    × 320
        ],
    )
    s1, s2, s3, s4, bottleneck = skip_model(backbone_in)

    # ── Decoder: simétrico al encoder, schedule de canales invertido ───────────
    x = _mnv2_decoder_block(bottleneck, s4,   160, expansion=6)  # 7×7   → 14×14  × 160
    x = _mnv2_decoder_block(x,          s3,    96, expansion=6)  # 14×14 → 28×28  × 96
    x = _mnv2_decoder_block(x,          s2,    64, expansion=6)  # 28×28 → 56×56  × 64
    x = _mnv2_decoder_block(x,          s1,    32, expansion=6)  # 56×56 → 112×112 × 32
    x = _mnv2_decoder_block(x,          None,  16, expansion=6)  # 112×112 → 224×224 × 16

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x)
    return models.Model(inputs=model_input, outputs=outputs)


# ── Factory ───────────────────────────────────────────────────────────────────
_BUILDERS = {
    "unet":             build_unet,
    "unet_mobilenetv2": build_unet_mobilenetv2,
    "mobilenetv2_sym":  build_mobilenetv2_sym,
}


def build_model(input_size=(*IMAGE_SIZE, INPUT_CHANNELS), num_classes=NUM_CLASSES):
    if ARCHITECTURE not in _BUILDERS:
        raise ValueError(
            f"Arquitectura desconocida: '{ARCHITECTURE}'. "
            f"Opciones válidas: {list(_BUILDERS)}"
        )
    uses_backbone = ARCHITECTURE in ("unet_mobilenetv2", "mobilenetv2")
    if not uses_backbone and ENCODER_WEIGHTS not in (None, "None"):
        print(
            f"AVISO: ARCHITECTURE='{ARCHITECTURE}' no usa backbone preentrenado. "
            "ENCODER_WEIGHTS se ignora."
        )
    print(f"Arquitectura seleccionada: {ARCHITECTURE}")
    return _BUILDERS[ARCHITECTURE](input_size, num_classes)


# ============================================================
# MÉTRICAS PERSONALIZADAS
# ============================================================
def iou_metric(y_true, y_pred):
    num_classes = y_pred.shape[-1]
    if y_true.shape.rank == 4:
        y_true = tf.argmax(y_true, axis=-1)
    if y_pred.shape.rank == 4:
        y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    iou_scores = []
    for c in range(num_classes):
        y_true_c = tf.where(y_true == c, 1.0, 0.0)
        y_pred_c = tf.where(y_pred == c, 1.0, 0.0)
        intersection = K.sum(y_true_c * y_pred_c, axis=[1, 2])
        union = K.sum(y_true_c, axis=[1, 2]) + K.sum(y_pred_c, axis=[1, 2]) - intersection
        iou_scores.append(intersection / (union + K.epsilon()))
    return K.mean(tf.stack(iou_scores))


def dice_metric(y_true, y_pred):
    num_classes = y_pred.shape[-1]
    if y_true.shape.rank == 4:
        y_true = tf.argmax(y_true, axis=-1)
    if y_pred.shape.rank == 4:
        y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dice_scores = []
    for c in range(num_classes):
        y_true_c = tf.where(y_true == c, 1.0, 0.0)
        y_pred_c = tf.where(y_pred == c, 1.0, 0.0)
        intersection = K.sum(y_true_c * y_pred_c, axis=[1, 2])
        dice_c = (2.0 * intersection) / (
            K.sum(y_true_c, axis=[1, 2]) + K.sum(y_pred_c, axis=[1, 2]) + K.epsilon()
        )
        dice_scores.append(dice_c)
    return K.mean(tf.stack(dice_scores))


# ============================================================
# VISUALIZACIÓN
# ============================================================
CLASS_COLORS = {
    0: [0, 0, 0],       # fondo  - negro
    1: [255, 0, 0],     # pinky  - rojo
    2: [0, 255, 0],     # middle - verde
    3: [0, 0, 255],     # thumb  - azul
    4: [255, 255, 0],   # wrist  - amarillo
}


def apply_color_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    return colored


def visualize_samples(images, masks, num_samples=5, save_folder=None):
    os.makedirs(save_folder, exist_ok=True)
    for i in range(min(num_samples, len(images))):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        img_show = images[i].squeeze()
        ax[0].imshow(img_show, cmap="gray" if INPUT_CHANNELS == 1 else None)
        ax[0].set_title("Imagen original"); ax[0].axis("off")
        ax[1].imshow(masks[i].squeeze(), cmap="gray")
        ax[1].set_title("Máscara"); ax[1].axis("off")
        if save_folder:
            plt.savefig(os.path.join(save_folder, f"sample_{i}.png"))
        plt.close()


def plot_training_history(history, save_path):
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for key in ("loss", "val_loss"):
        if key in h:
            ax1.plot(h[key], marker="o", label=key)
    ax1.set_title("Loss")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss")
    ax1.set_ylim(bottom=0)
    ax1.legend(); ax1.grid(True)

    for key in ("accuracy", "val_accuracy", "iou_metric", "val_iou_metric",
                "dice_metric", "val_dice_metric"):
        if key in h:
            ax2.plot(h[key], marker="o", label=key)
    ax2.set_title("Métricas")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Valor")
    ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Historial guardado: {save_path}")


def predict_and_save(model, image_path, save_path):
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original_rgb, IMAGE_SIZE)

    if INPUT_CHANNELS == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        inp = np.expand_dims(gray / 255.0, axis=(0, -1))   # (1, H, W, 1)
    else:
        inp = np.expand_dims(resized / 255.0, axis=0)      # (1, H, W, 3)

    pred = model.predict(inp, verbose=0)
    pred_mask = np.argmax(pred[0], axis=-1)
    colored = apply_color_mask(pred_mask)
    overlay = cv2.addWeighted(resized, 0.6, colored, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [resized, colored, overlay],
                               ["Original", "Máscara predicha", "Superposición"]):
        ax.imshow(img); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Predicción de prueba guardada: {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    run_dir      = get_run_dir()
    models_dir   = os.path.join(run_dir, "models")
    history_dir  = os.path.join(run_dir, "training_history")
    eval_dir     = os.path.join(run_dir, "evaluation")
    samples_dir  = os.path.join(eval_dir, "samples")
    for d in [models_dir, history_dir, eval_dir, samples_dir]:
        os.makedirs(d, exist_ok=True)
    setup_logging("01_train_hand_detector.py", log_dir=run_dir)
    save_run_config(run_dir)

    with timer("Carga de datos"):
        print(f"Imágenes: {HAND_DETECTOR_IMAGES_DIR}")
        print(f"Anotaciones: {HAND_DETECTOR_ANNOTATIONS_DIR}")
        images, masks = load_data()

    print(f"Dataset: {images.shape}  |  Máscaras: {masks.shape}")
    unique, counts = np.unique(masks, return_counts=True)
    print(f"Distribución de clases (antes del split): {dict(zip(unique.tolist(), counts.tolist()))}")

    visualize_samples(images, masks, num_samples=5, save_folder=samples_dir)

    num_classes = NUM_CLASSES
    x_train, x_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    print(f"Train: {len(x_train)}  |  Val: {len(x_val)}")

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val,   counts_val   = np.unique(y_val,   return_counts=True)
    print(f"Clases train: {dict(zip(unique_train.tolist(), counts_train.tolist()))}")
    print(f"Clases val:   {dict(zip(unique_val.tolist(),   counts_val.tolist()))}")

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes=num_classes)

    with timer("Construcción del modelo"):
        model = build_model((*IMAGE_SIZE, INPUT_CHANNELS), num_classes)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", iou_metric, dice_metric],
        )
        model.summary()

    # Pesos de clase
    unique_cls, cls_counts = np.unique(y_train.argmax(axis=-1), return_counts=True)
    total_pixels = np.sum(cls_counts)
    class_weights = {int(c): total_pixels / cnt for c, cnt in zip(unique_cls, cls_counts)}
    print(f"Pesos de clase: {class_weights}")

    seed = 42
    batch_size = BATCH_SIZE
    steps_per_epoch  = max(1, len(x_train) // batch_size)
    validation_steps = max(1, len(x_val)   // batch_size)

    # ── Data Augmentation ────────────────────────────────────────────────────
    # Las transformaciones geométricas se aplican igual a imagen y máscara
    # (usando el mismo seed). El brillo solo se aplica a la imagen.
    if DATA_AUGMENTATION:
        print("Data augmentation activado.")
        geo_params = dict(
            rotation_range=AUG_ROTATION_RANGE,
            width_shift_range=AUG_WIDTH_SHIFT_RANGE,
            height_shift_range=AUG_HEIGHT_SHIFT_RANGE,
            zoom_range=AUG_ZOOM_RANGE,
            horizontal_flip=AUG_HORIZONTAL_FLIP,
            vertical_flip=AUG_VERTICAL_FLIP,
            shear_range=AUG_SHEAR_RANGE,
            fill_mode="nearest",
        )
        img_gen_params  = {**geo_params}
        if AUG_BRIGHTNESS_RANGE is not None:
            img_gen_params["brightness_range"] = AUG_BRIGHTNESS_RANGE
        mask_gen_params = geo_params  # sin brillo para las máscaras
    else:
        img_gen_params  = {}
        mask_gen_params = {}

    train_img_gen  = ImageDataGenerator(**img_gen_params).flow(
        x_train, batch_size=batch_size, seed=seed, shuffle=True
    )
    train_mask_gen = ImageDataGenerator(**mask_gen_params).flow(
        y_train_cat, batch_size=batch_size, seed=seed, shuffle=True
    )
    val_img_gen  = ImageDataGenerator().flow(x_val,     batch_size=batch_size, seed=seed, shuffle=False)
    val_mask_gen = ImageDataGenerator().flow(y_val_cat, batch_size=batch_size, seed=seed, shuffle=False)

    def _gen(ig, mg):
        while True:
            yield next(ig), next(mg)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
    ]

    with timer("Entrenamiento"):
        history = model.fit(
            _gen(train_img_gen, train_mask_gen),
            steps_per_epoch=steps_per_epoch,
            validation_data=_gen(val_img_gen, val_mask_gen),
            validation_steps=validation_steps,
            epochs=EPOCHS,
            callbacks=callbacks,
        )

    plot_training_history(history, os.path.join(history_dir, "training_history.png"))

    with timer("Evaluación"):
        results = model.evaluate(x_val, y_val_cat, verbose=0)
        print(f"Val loss:       {results[0]:.4f}")
        print(f"Val accuracy:   {results[1]:.4f}")
        print(f"Val IoU:        {results[2]:.4f}")
        print(f"Val Dice Score: {results[3]:.4f}")

        perf_df = pd.DataFrame({
            "Métrica": ["Loss", "Accuracy", "IoU", "Dice Score"],
            "Valor":   [round(r, 4) for r in results],
        })
        perf_table_path = os.path.join(eval_dir, "performance_table.png")
        dfi.export(perf_df.style, perf_table_path)
        print(f"Tabla de performance guardada: {perf_table_path}")

    # Predicción de prueba con la primera imagen de validación
    sample_imgs = glob.glob(os.path.join(HAND_DETECTOR_IMAGES_DIR, "*.png"))
    if sample_imgs:
        predict_and_save(model, sample_imgs[0],
                         os.path.join(eval_dir, "test_prediction.png"))

    # Guardar modelo
    model_path = os.path.join(models_dir, "segmentation_model.h5")
    model.save(model_path)
    model.save(SEGMENTATION_MODEL_PATH)
    print(f"Modelo guardado en: {model_path}")
    print(f"Modelo guardado en: {SEGMENTATION_MODEL_PATH}")


if __name__ == "__main__":
    setup_gpu()
    main()
    report_timing(START_TIME, "01_train_hand_detector.py")
