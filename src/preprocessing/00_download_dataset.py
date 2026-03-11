#!/usr/bin/env python3
"""
Paso 0: Descarga el dataset RSNA Bone Age desde Kaggle usando kagglehub.
Las imágenes se copian al directorio images/boneage-training-dataset/ del proyecto.

Uso:
    python src/preprocessing/00_download_dataset.py
"""
import os
import sys
import shutil
import time

# Asegurar que el root del proyecto esté en el path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import RAW_IMAGES_DIR
from src.utils.timing import report_timing

START_TIME = time.time()


def setup_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {[g.name for g in gpus]}")
    else:
        print("ADVERTENCIA: No se detectó GPU. Ejecutando en CPU.")


def download_dataset():
    import kagglehub

    print("Descargando dataset 'kmader/rsna-bone-age' desde Kaggle...")
    path = kagglehub.dataset_download("kmader/rsna-bone-age")
    print(f"Dataset descargado en: {path}")
    return path


def copy_images(kaggle_path: str, dest_dir: str):
    """Copia las imágenes del dataset descargado al directorio del proyecto."""
    os.makedirs(dest_dir, exist_ok=True)

    copied = 0
    skipped = 0
    for root, _, files in os.walk(kaggle_path):
        for fname in files:
            if fname.lower().endswith(".png"):
                src = os.path.join(root, fname)
                dst = os.path.join(dest_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    skipped += 1

    print(f"Imágenes copiadas: {copied} | ya existentes (omitidas): {skipped}")
    print(f"Directorio destino: {dest_dir}")


if __name__ == "__main__":
    kaggle_path = download_dataset()
    copy_images(kaggle_path, RAW_IMAGES_DIR)
    report_timing(START_TIME, "00_download_dataset.py")
