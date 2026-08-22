# Experimento 60 — Baseline whole-hand (sin segmentación), Comentario 15
# Idéntico al experimento 27 (F-DenseNet121, modelo de referencia del manuscrito)
# salvo MODEL_TYPE: una sola rama sobre la mano completa (recorte+CLAHE, sin
# segmentar en 4 regiones), mismo split/backbone/hiperparámetros para
# comparación directa y controlada del efecto de la segmentación anatómica.

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

AGE_RANGE = (24, 216)
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

MODEL_TYPE = "whole_hand"
USE_CROSS_VALIDATION = False
N_FOLDS = 5
FREEZE_EXTRACTORS = True
DATASET_PATH = "data/training/dataset_analysis/balanced_dataset.csv"
SEGMENT_MODE = "spatial"
SEGMENTATION_MODEL = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
