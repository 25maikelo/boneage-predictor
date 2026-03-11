#!/usr/bin/env python3
"""
Paso 1: Entrenamiento del detector de mano (U-Net con MobileNetV2).
Lee anotaciones en formato LabelMe (un JSON por imagen) y entrena
un modelo de segmentación de 5 clases:
  0=fondo, 1=pinky, 2=middle, 3=thumb, 4=wrist

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

import numpy as np
import cv2
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, Conv2DTranspose, concatenate, Conv2D, BatchNormalization, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config.paths import (
    HAND_DETECTOR_IMAGES_DIR,
    HAND_DETECTOR_ANNOTATIONS_DIR,
    SEGMENTATION_MODEL_PATH,
    PRETRAINED_MODELS_DIR,
)
from config.segmentation import (
    IMAGE_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, ENCODER_WEIGHTS
)
from src.utils.timing import report_timing, setup_logging, timer

START_TIME = time.time()

# Mapa de etiqueta → índice de clase
LABEL_MAP = {"pinky": 1, "middle": 2, "thumb": 3, "wrist": 4}


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
# ARQUITECTURA U-Net + MobileNetV2
# ============================================================
def build_unet_mobilenetv2(input_shape=(224, 224, 3), num_classes=5):
    """U-Net con MobileNetV2 como encoder."""
    inputs = Input(shape=input_shape)
    encoder = MobileNetV2(input_tensor=inputs, include_top=False, weights=ENCODER_WEIGHTS)

    s1 = encoder.get_layer("input_1").output               # 224x224
    s2 = encoder.get_layer("block_1_expand_relu").output   # 112x112
    s3 = encoder.get_layer("block_3_expand_relu").output   # 56x56
    s4 = encoder.get_layer("block_6_expand_relu").output   # 28x28
    bridge = encoder.get_layer("block_13_expand_relu").output  # 14x14

    def up_block(x, skip, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
        x = concatenate([x, skip])
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    x = up_block(bridge, s4, 256)
    x = up_block(x, s3, 128)
    x = up_block(x, s2, 64)
    x = up_block(x, s1, 32)
    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)


# ============================================================
# CARGA DE DATOS (formato LabelMe)
# ============================================================
def polygon_to_mask(points, height, width):
    """Convierte lista de puntos [x,y] a máscara binaria."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def load_sample(img_id, img_size):
    """Carga imagen y genera máscara multiclase desde JSON LabelMe."""
    h, w = img_size

    img_path = os.path.join(HAND_DETECTOR_IMAGES_DIR, f"{img_id}.png")
    ann_path = os.path.join(HAND_DETECTOR_ANNOTATIONS_DIR, f"{img_id}.json")

    # Imagen
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagen no encontrada: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0

    # Máscara multiclase (fondo=0 por defecto)
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)
    for shape in ann.get("shapes", []):
        label = shape["label"]
        class_idx = LABEL_MAP.get(label, 0)
        if class_idx == 0:
            continue
        points = shape["points"]
        region = polygon_to_mask(points, orig_h, orig_w)
        mask[region == 1] = class_idx

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, mask


def load_all_samples(img_size):
    """Carga todas las muestras disponibles en HAND_DETECTOR_ANNOTATIONS_DIR."""
    ann_files = glob.glob(os.path.join(HAND_DETECTOR_ANNOTATIONS_DIR, "*.json"))
    img_ids = [os.path.splitext(os.path.basename(f))[0] for f in ann_files]

    images, masks = [], []
    skipped = 0
    for img_id in sorted(img_ids):
        img_path = os.path.join(HAND_DETECTOR_IMAGES_DIR, f"{img_id}.png")
        if not os.path.exists(img_path):
            skipped += 1
            continue
        try:
            img, mask = load_sample(img_id, img_size)
            images.append(img)
            masks.append(mask)
        except Exception as e:
            print(f"  [WARN] {img_id}: {e}")
            skipped += 1

    print(f"  Muestras cargadas: {len(images)}  |  Omitidas: {skipped}")
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.int32)


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.isdir(HAND_DETECTOR_ANNOTATIONS_DIR):
        print(f"ERROR: No se encontró directorio de anotaciones: {HAND_DETECTOR_ANNOTATIONS_DIR}")
        sys.exit(1)
    if not os.path.isdir(HAND_DETECTOR_IMAGES_DIR):
        print(f"ERROR: No se encontró directorio de imágenes: {HAND_DETECTOR_IMAGES_DIR}")
        sys.exit(1)

    with timer("Carga de datos"):
        print(f"Cargando imágenes desde: {HAND_DETECTOR_IMAGES_DIR}")
        print(f"Cargando anotaciones desde: {HAND_DETECTOR_ANNOTATIONS_DIR}")
        X, y = load_all_samples(IMAGE_SIZE)

    print(f"Dataset: {X.shape}  |  Máscaras: {y.shape}")
    print(f"Clases únicas en máscaras: {np.unique(y)}")

    # Split 80/20
    n = len(X)
    split = int(n * 0.8)
    idx = np.random.permutation(n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_val,   y_val   = X[idx[split:]], y[idx[split:]]
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    with timer("Construcción del modelo"):
        print("Construyendo U-Net con MobileNetV2...")
        model = build_unet_mobilenetv2((*IMAGE_SIZE, 3), NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

    os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            SEGMENTATION_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    with timer("Entrenamiento"):
        print(f"Entrenando por hasta {EPOCHS} épocas (batch={BATCH_SIZE})...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

    print(f"\nModelo guardado en: {SEGMENTATION_MODEL_PATH}")
    final_acc = history.history["val_accuracy"][-1]
    print(f"Val accuracy final: {final_acc:.4f}")


if __name__ == "__main__":
    setup_logging("01_train_hand_detector.py")
    setup_gpu()
    main()
    report_timing(START_TIME, "01_train_hand_detector.py")
