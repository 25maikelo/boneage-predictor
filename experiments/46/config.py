# ============================================================
# Experimento 46 — unified_cnn, dataset balanceado (≥50 imgs/edad)
# 4 ramas CNN separadas entrenadas end-to-end (sin pipeline de fusión).
# Comparar contra exp 39 (simple_cnn mismos params, pipeline 2 etapas).
# ============================================================

MODEL_TYPE = "unified_cnn"
IMAGE_SIZE = (112, 112)
BASE_MODEL_CHOICE = None
WEIGHTS = None
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
NUM_LAYERS_UNFREEZE = 0
CNN_FILTERS = [32, 64, 128, 256]
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.3

BATCH_SIZE = 32
EPOCHS_SEGMENT = 30
FUSION_EPOCHS = 0
FINE_TUNING_EPOCHS = 0
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

USE_CROSS_VALIDATION = True
N_FOLDS = 5

FREEZE_EXTRACTORS = True

DATASET_PATH = "data/training/dataset_analysis/balanced_dataset.csv"

SEGMENTATION_MODEL = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
