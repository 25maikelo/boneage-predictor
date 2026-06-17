# ============================================================
# Experimento 55 — backbone, imagen 224×224
# Base: Exp 37 (backbone, completo, LR=1e-3, FUSION_EPOCHS=20, IMAGE_SIZE=112×112)
# Variable cambiada: IMAGE_SIZE = (224, 224)
# Pregunta: ¿imágenes más grandes mejoran la predicción en backbone?
# Comparar contra Exp 37 (Val MAE=15.4m, Mex MAE=16.7m)
# NOTA: tiempo estimado ~30-45h — monitorear límite de 4 días
# NOTA 2: BATCH_SIZE reducido a 8 (32 causó OOM en fusión — 4 DenseNet121
#         224x224 simultáneos no caben en GPU con batch=32; segmentos sí)
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

SEGMENTATION_MODEL = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
