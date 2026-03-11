#!/usr/bin/env python3
"""
Análisis y balanceo del dataset de entrenamiento.
Filtra imágenes que tienen los 4 segmentos y aplica umbral mínimo por edad.

Uso:
    python src/dataset_analysis.py [--experiment N]
"""
import os
import sys
import json
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt

from config.paths import TRAINING_CSV, SEGMENTED_IMAGES_DIR, DATASET_ANALYSIS_DIR
from config.experiment import load_experiment_config
from src.utils.timing import report_timing, setup_logging

START_TIME = time.time()

UMBRAL_MIN_IMAGENES = 50
SEGMENT_CLASSES = ["pinky", "middle", "thumb", "wrist"]


def parse_args():
    parser = argparse.ArgumentParser(description="Análisis y balanceo del dataset")
    parser.add_argument("--experiment", type=int, default=26,
                        help="Número de experimento (para leer SEGMENTS_ORDER del config)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.experiment)
    segments_order = getattr(cfg, "SEGMENTS_ORDER", SEGMENT_CLASSES)

    os.makedirs(DATASET_ANALYSIS_DIR, exist_ok=True)

    df = pd.read_csv(TRAINING_CSV)
    print(f"Total imágenes en CSV: {len(df)}")

    # Verificar que cada imagen tenga los 4 segmentos
    def tiene_todos(img_id):
        return all(
            os.path.exists(os.path.join(SEGMENTED_IMAGES_DIR, seg, f"{img_id}.png"))
            for seg in segments_order
        )

    df["has_all_segments"] = df["id"].apply(tiene_todos)
    completas = df[df["has_all_segments"]]
    faltantes = df[~df["has_all_segments"]]

    print(f"Con los 4 segmentos: {len(completas)} | Con faltantes: {len(faltantes)}")

    # Balanceo por edad
    counts = completas.groupby("boneage").size()
    edades_descartadas = counts[counts < UMBRAL_MIN_IMAGENES].index.tolist()
    print(f"Edades descartadas (< {UMBRAL_MIN_IMAGENES} imgs): {edades_descartadas}")

    edades_ok = counts[counts >= UMBRAL_MIN_IMAGENES].index
    balanceado = completas[completas["boneage"].isin(edades_ok)]
    print(f"Dataset balanceado: {len(balanceado)} imágenes | "
          f"Rango: {edades_ok.min()}-{edades_ok.max()} meses")

    # Guardar dataset balanceado
    balanced_path = os.path.join(DATASET_ANALYSIS_DIR, "balanced_dataset.csv")
    balanceado.to_csv(balanced_path, index=False)
    print(f"Dataset balanceado guardado en: {balanced_path}")

    # Estadísticas
    stats = {
        "total": int(len(df)),
        "con_4_segmentos": int(len(completas)),
        "balanceado": int(len(balanceado)),
        "edad_min": int(edades_ok.min()),
        "edad_max": int(edades_ok.max()),
        "edades_descartadas": edades_descartadas,
        "edades_consideradas": sorted(edades_ok.tolist()),
    }
    with open(os.path.join(DATASET_ANALYSIS_DIR, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Gráficos
    for data, title, fname, color in [
        (df["boneage"], "Dataset Original", "age_distribution.png", "#66b3ff"),
        (completas["boneage"], "Dataset Filtrado", "filtered_age_distribution.png", "#99cc99"),
        (balanceado["boneage"], "Dataset Balanceado", "balanced_age_distribution.png", "#ffcc66"),
    ]:
        plt.figure(figsize=(8, 6))
        data.hist(bins=20, color=color, edgecolor="black")
        plt.xlabel("Edad (meses)"); plt.ylabel("Frecuencia")
        plt.title(f"Distribución de Edad — {title}")
        plt.savefig(os.path.join(DATASET_ANALYSIS_DIR, fname)); plt.close()

    plt.figure(figsize=(6, 6))
    plt.pie(
        [len(completas), len(faltantes)],
        labels=["Completas", "Con faltantes"],
        autopct="%1.1f%%", startangle=90, colors=["#66b3ff", "#ff9999"]
    )
    plt.title("Proporción de Imágenes Originales vs Usables")
    plt.savefig(os.path.join(DATASET_ANALYSIS_DIR, "dataset_proportion.png")); plt.close()

    print("Análisis completado.")


if __name__ == "__main__":
    setup_logging("05_dataset_analysis.py")
    main()
    report_timing(START_TIME, "05_dataset_analysis.py")
