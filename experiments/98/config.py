# ============================================================
# Experimento 98 — QUICK TEST backbone_vectors
# Prueba mínima de create_fusion_model_backbone_vectors.
# Datos reducidos, 1 fold, pocas épocas.
# ============================================================

MODEL_TYPE          = "backbone_vectors"
IMAGE_SIZE          = (112, 112)
BASE_MODEL_CHOICE   = "densenet121"
WEIGHTS             = None
DENSE_UNITS         = 256
DROPOUT_RATE        = 0.5
NUM_LAYERS_UNFREEZE = 10

BATCH_SIZE          = 32
EPOCHS_SEGMENT      = 2
FUSION_EPOCHS       = 2
FINE_TUNING_EPOCHS  = 1
LEARNING_RATE       = 0.001
OPTIMIZER_CHOICE    = "adam"
TEST_SPLIT          = 0.2

AGE_RANGE           = (24, 216)
USE_GENDER          = True
USE_AUGMENTATION    = False
MAX_SAMPLES         = 200

LOSS_FUNCTION_NAME  = "attention_loss"
INITIAL_K           = 3.0

AUG_RESCALE         = 1.0 / 255
AUG_HORIZONTAL_FLIP = False
AUG_ROTATION_RANGE  = 20
AUG_BRIGHTNESS_RANGE= [0.8, 1.2]
AUG_ZOOM_RANGE      = 0.2

USE_WARMUP          = False
WARMUP_EPOCHS       = 5
WARMUP_INITIAL_LR   = 1e-5

SEGMENTS_ORDER      = ["pinky", "middle", "thumb", "wrist"]
USE_CROSS_VALIDATION= True
N_FOLDS             = 2

SEGMENTATION_MODEL  = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
FREEZE_EXTRACTORS = True
