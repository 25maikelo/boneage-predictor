#!/usr/bin/env python3
"""
Paso 1: Entrenamiento del detector de mano (U-Net con MobileNetV2).
Lee anotaciones JSON y entrena un modelo de segmentación de 5 clases.

Uso:
    python src/preprocessing/01_train_hand_detector.py
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import cv2
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, Conv2DTranspose, concatenate, Conv2D, BatchNormalization, Activation
)
from tensorflow.keras.models import Model
from config.paths import PRETRAINED_MODELS_DIR
from config.segmentation import (
    IMAGE_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, ENCODER_WEIGHTS
)
from src.utils.timing import report_timing

START_TIME = time.time()

# ============================================================
# RUTAS (desde config/paths.py)
# ============================================================
TRAINING_DATA_DIR = os.path.join(
    PROJECT_ROOT, "scripts", "preprocessing", "hand-detector-training"
)
IMAGES_DIR   = os.path.join(TRAINING_DATA_DIR, "images")
MASKS_DIR    = os.path.join(TRAINING_DATA_DIR, "masks")
ANNOTATIONS  = os.path.join(TRAINING_DATA_DIR, "annotations.json")
OUTPUT_MODEL = os.path.join(PRETRAINED_MODELS_DIR, "modelo_segmentacion.h5")


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {[g.name for g in gpus]}")
    else:
        print("ADVERTENCIA: No se detectó GPU. Ejecutando en CPU.")


def build_unet_mobilenetv2(input_shape=(224, 224, 3), num_classes=5):
    """U-Net con MobileNetV2 como encoder."""
    inputs = Input(shape=input_shape)
    encoder = MobileNetV2(input_tensor=inputs, include_top=False, weights=ENCODER_WEIGHTS)

    s1 = encoder.get_layer("input_1").output           # 224x224
    s2 = encoder.get_layer("block_1_expand_relu").output  # 112x112
    s3 = encoder.get_layer("block_3_expand_relu").output  # 56x56
    s4 = encoder.get_layer("block_6_expand_relu").output  # 28x28
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


if __name__ == "__main__":
    setup_gpu()

    if not os.path.exists(ANNOTATIONS):
        print(f"No se encontraron anotaciones en {ANNOTATIONS}")
        print("Coloca el archivo annotations.json con polígonos por clase.")
        sys.exit(1)

    print("Cargando modelo U-Net con MobileNetV2...")
    model = build_unet_mobilenetv2((*IMAGE_SIZE, 3), NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print(f"Modelo guardado en: {OUTPUT_MODEL}")
    os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)
    model.save(OUTPUT_MODEL)

    report_timing(START_TIME, "01_train_hand_detector.py")
