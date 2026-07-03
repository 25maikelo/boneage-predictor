# ============================================================
# Experimento 57 — backbone 224×224, segmentos recortados (cropped)
# Base: Exp 55 (backbone, 224×224, attention_loss, RSNA MAE=13.4m — mejor global)
# Variable cambiada: SEGMENT_MODE = "cropped" (en vez de "spatial")
# Pregunta: ¿recortar cada segmento a su bounding box (en vez de mantener
#           posición espacial completa) mejora el uso de la resolución
#           y por lo tanto el MAE, especialmente a 224×224?
# Comparar contra Exp 55 (Val MAE=13.4m, Mex MAE=17.4m)
# NOTA: requiere data/images/segmented_cropped/ (generado con
#       `python src/preprocessing/04_segment_images.py --mode cropped`)
# NOTA 2: BATCH_SIZE=8 (mismo fix que Exp 55, evita OOM en fusión a 224×224)
# ============================================================

MODEL_TYPE = "backbone"
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

USE_WARMUP = False
WARMUP_EPOCHS = 5
WARMUP_INITIAL_LR = 1e-5

SEGMENTS_ORDER = ["pinky", "middle", "thumb", "wrist"]

USE_CROSS_VALIDATION = False
N_FOLDS = 5

FREEZE_EXTRACTORS = True

DATASET_PATH = "data/training/boneage-training-dataset.csv"

SEGMENT_MODE = "cropped"
SEGMENT_CROP_PADDING = 0.15
SEGMENTS_FOLDER = "/lustre/home/mlozano/boneage-predictor/data/images/segmented_cropped"

SEGMENTATION_MODEL = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
