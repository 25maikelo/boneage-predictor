"""
Fuente única de verdad para todas las rutas del proyecto.
Todos los scripts importan rutas desde aquí.
"""
import os

# Raíz del proyecto (carpeta que contiene config/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# DATOS
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Imágenes (pipeline de preprocesamiento)
IMAGES_DIR           = os.path.join(DATA_DIR, "images")
RAW_IMAGES_DIR       = os.path.join(IMAGES_DIR, "raw")
LABELED_IMAGES_DIR   = os.path.join(IMAGES_DIR, "labeled")
CROPPED_IMAGES_DIR   = os.path.join(IMAGES_DIR, "cropped")
EQUALIZED_IMAGES_DIR = os.path.join(IMAGES_DIR, "equalized")
MASKS_DIR            = os.path.join(IMAGES_DIR, "masks")
SEGMENTS_DIR         = os.path.join(IMAGES_DIR, "segments")
SEGMENTED_IMAGES_DIR = os.path.join(IMAGES_DIR, "segmented")

# Datos de entrenamiento del detector de mano (LabelMe format)
HAND_DETECTOR_DIR             = os.path.join(DATA_DIR, "hand-detector")
HAND_DETECTOR_IMAGES_DIR      = os.path.join(HAND_DETECTOR_DIR, "images")
HAND_DETECTOR_ANNOTATIONS_DIR = os.path.join(HAND_DETECTOR_DIR, "annotations")

# Entrenamiento
TRAINING_DIR         = os.path.join(DATA_DIR, "training")
TRAINING_CSV         = os.path.join(TRAINING_DIR, "boneage-training-dataset.csv")
DATASET_ANALYSIS_DIR = os.path.join(TRAINING_DIR, "dataset_analysis")
BALANCED_DATASET_CSV = os.path.join(DATASET_ANALYSIS_DIR, "balanced_dataset.csv")

# Validación estándar
VALIDATION_DIR        = os.path.join(DATA_DIR, "validation")
VALIDATION_CSV        = os.path.join(VALIDATION_DIR, "validation_dataset.csv")
VALIDATION_IMAGES_DIR = os.path.join(VALIDATION_DIR, "images")

# Validación mexicana
MEX_DIR        = os.path.join(DATA_DIR, "mex-validation")
MEX_CSV        = os.path.join(MEX_DIR, "mex_dataset.csv")
MEX_IMAGES_DIR = os.path.join(MEX_DIR, "images")

# ============================================================
# MODELOS PRE-ENTRENADOS
# ============================================================
PRETRAINED_MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
SEGMENTATION_MODEL_PATH = os.path.join(PRETRAINED_MODELS_DIR, "modelo_segmentacion.h5")
BASE_MODEL_PATH         = os.path.join(PRETRAINED_MODELS_DIR, "base_model.keras")

# ============================================================
# EXPERIMENTOS
# ============================================================
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# ============================================================
# LOGS
# ============================================================
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
