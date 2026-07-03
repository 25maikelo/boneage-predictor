# ============================================================
# Experimento 58 — backbone_vectors libre 224×224, segmentos recortados (cropped)
# Pareja exacta de Exp 57 (misma configuración, solo cambia MODEL_TYPE) para
# comparar backbone vs backbone_vectors bajo idénticas condiciones.
# Base conceptual: Exp 43 (backbone_vectors libre, completo, con género)
# Variables cambiadas vs Exp 43: IMAGE_SIZE=224×224, SEGMENT_MODE="cropped"
# Pregunta: ¿recortar cada segmento a su bounding box mejora backbone_vectors
#           a 224×224, bajo la misma config que Exp 57 (backbone)?
# Comparar contra Exp 43 (Val MAE=23.0m, Mex MAE=18.5m) y Exp 57 (backbone)
# NOTA: requiere data/images/segmented_cropped/ (generado con
#       `python src/preprocessing/04_segment_images.py --mode cropped`)
# NOTA 2: BATCH_SIZE=8 (mismo fix que Exp 55/56, evita OOM en fusión a 224×224)
# ============================================================

MODEL_TYPE = "backbone_vectors"
IMAGE_SIZE = (224, 224)
BASE_MODEL_CHOICE = "densenet121"
WEIGHTS = None
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
NUM_LAYERS_UNFREEZE = 10

BATCH_SIZE = 8
EPOCHS_SEGMENT = 15
FUSION_EPOCHS = 20
FINE_TUNING_EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER_CHOICE = "adam"
TEST_SPLIT = 0.2

AGE_RANGE = (1, 228)
USE_GENDER = True
USE_AUGMENTATION = True

LOSS_FUNCTION_NAME = "attention_loss"
INITIAL_K = 3.0

AUG_RESCALE = 1.0 / 255
AUG_HORIZONTAL_FLIP = False
AUG_ROTATION_RANGE = 20
AUG_BRIGHTNESS_RANGE = [0.8, 1.2]
AUG_ZOOM_RANGE = 0.2

USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_INITIAL_LR = 1e-5

SEGMENTS_ORDER = ["pinky", "middle", "thumb", "wrist"]

USE_CROSS_VALIDATION = False
N_FOLDS = 5

FREEZE_EXTRACTORS = False

DATASET_PATH = "data/training/boneage-training-dataset.csv"

SEGMENT_MODE = "cropped"
SEGMENT_CROP_PADDING = 0.15
SEGMENTS_FOLDER = "/lustre/home/mlozano/boneage-predictor/data/images/segmented_cropped"

SEGMENTATION_MODEL = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
