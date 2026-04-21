# ============================================================
# Experimento 43 — DenseNet121 backbone_vectors con extractores descongelados, dataset completo
# Réplica de exp 38 (backbone_vectors, AGE_RANGE 1–228) pero con
# FREEZE_EXTRACTORS = False: los 4 extractores DenseNet (256-dim)
# se entrenan junto con las capas de fusión desde el inicio de esa fase.
# Warmup activado — DenseNet tiene ~7M parámetros, riesgo de
# gradientes desestabilizadores si la fusión arranca sin precalentamiento.
# Comparar contra exp 38 para evaluar impacto del descongelamiento.
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

USE_CROSS_VALIDATION = True
N_FOLDS = 5

FREEZE_EXTRACTORS = False

DATASET_PATH = "data/training/boneage-training-dataset.csv"

SEGMENTATION_MODEL = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
