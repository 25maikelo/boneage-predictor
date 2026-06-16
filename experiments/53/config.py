# ============================================================
# Experimento 53 — backbone_vectors libre, sin género, LR bajo
# Base: Exp 43 (backbone_vectors libre, completo, LR=1e-3, FUSION_EPOCHS=20)
# Variables cambiadas: USE_GENDER=False + LEARNING_RATE=1e-4 + FUSION_EPOCHS=10
# Pregunta: ¿la combinación de quitar género (exp 48, -4.5m) y LR bajo (exp 51, -4.0m)
#           da una mejora aditiva respecto al baseline?
# Comparar contra Exp 43 (base), Exp 48 (solo género) y Exp 51 (solo LR/épocas)
# ============================================================

MODEL_TYPE = "backbone_vectors"
IMAGE_SIZE = (112, 112)
BASE_MODEL_CHOICE = "densenet121"
WEIGHTS = None
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
NUM_LAYERS_UNFREEZE = 10

BATCH_SIZE = 32
EPOCHS_SEGMENT = 15
FUSION_EPOCHS = 10
FINE_TUNING_EPOCHS = 10
LEARNING_RATE = 0.0001
OPTIMIZER_CHOICE = "adam"
TEST_SPLIT = 0.2

AGE_RANGE = (1, 228)
USE_GENDER = False
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

SEGMENTATION_MODEL = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
