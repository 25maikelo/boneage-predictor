"""
Fuente única de verdad para todas las rutas del proyecto.
Todos los scripts importan rutas desde aquí.
"""
import os

# Raíz del proyecto (carpeta que contiene config/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# IMÁGENES (no versionadas en git)
# ============================================================
RAW_IMAGES_DIR       = os.path.join(PROJECT_ROOT, "images", "boneage-training-dataset")
CROPPED_IMAGES_DIR   = os.path.join(PROJECT_ROOT, "images", "imagenes_recortadas")
EQUALIZED_IMAGES_DIR = os.path.join(PROJECT_ROOT, "images", "imagenes_ecualizadas")
MASKS_DIR            = os.path.join(PROJECT_ROOT, "images", "mascaras_predichas")
SEGMENTS_DIR         = os.path.join(PROJECT_ROOT, "images", "segmentos")
SEGMENTED_IMAGES_DIR = os.path.join(PROJECT_ROOT, "images", "segmentos_spatial")

# ============================================================
# DATOS CSV (no versionados en git, excepto mex_dataset.csv)
# ============================================================
TRAINING_CSV         = os.path.join(PROJECT_ROOT, "training-data", "boneage-training-dataset.csv")
BALANCED_DATASET_CSV = os.path.join(PROJECT_ROOT, "training-data", "dataset_analysis", "balanced_dataset.csv")
DATASET_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "training-data", "dataset_analysis")

VALIDATION_CSV       = os.path.join(PROJECT_ROOT, "validation-data", "validation_dataset.csv")
VALIDATION_IMAGES_DIR= os.path.join(PROJECT_ROOT, "validation-data", "images")

MEX_CSV              = os.path.join(PROJECT_ROOT, "mex-validation-data", "mex_dataset.csv")
MEX_IMAGES_DIR       = os.path.join(PROJECT_ROOT, "mex-validation-data", "images")

# ============================================================
# MODELOS PRE-ENTRENADOS
# ============================================================
PRETRAINED_MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
SEGMENTATION_MODEL_PATH  = os.path.join(PRETRAINED_MODELS_DIR, "modelo_segmentacion.h5")
BASE_MODEL_PATH          = os.path.join(PRETRAINED_MODELS_DIR, "base_model.keras")

# ============================================================
# EXPERIMENTOS
# ============================================================
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# ============================================================
# LOGS
# ============================================================
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")