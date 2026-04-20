# ============================================================
# Experimento 99 — QUICK TEST (pipeline completo, datos mínimos)
# CNN pura sin backbone para máxima velocidad.
# Cubre todos los caminos de código: CV, fusión, validación, saliencias.
# ============================================================

# ── Modelo ──────────────────────────────────────────────────
MODEL_TYPE          = "simple_cnn"   # CNN sin backbone (rápida)
IMAGE_SIZE          = (64, 64)       # Imágenes pequeñas
BASE_MODEL_CHOICE   = None           # No usado con simple_cnn
WEIGHTS             = None
DENSE_UNITS         = 128
DROPOUT_RATE        = 0.3
NUM_LAYERS_UNFREEZE = 0
CNN_FILTERS         = [16, 32, 64]   # CNN compacta
CNN_KERNEL_SIZE     = 3
CNN_DROPOUT         = 0.3

# ── Entrenamiento ────────────────────────────────────────────
BATCH_SIZE          = 16
EPOCHS_SEGMENT      = 2
FUSION_EPOCHS       = 2
FINE_TUNING_EPOCHS  = 1
LEARNING_RATE       = 0.001
OPTIMIZER_CHOICE    = "adam"
TEST_SPLIT          = 0.2

# ── Dataset ──────────────────────────────────────────────────
AGE_RANGE           = (24, 216)
USE_GENDER          = False          # simple_cnn no usa género
USE_AUGMENTATION    = True
MAX_SAMPLES         = 80             # 80 muestras → ~32 train/8 val por fold

# ── Pérdida ──────────────────────────────────────────────────
LOSS_FUNCTION_NAME  = "attention_loss"
INITIAL_K           = 3.0

# ── Augmentación ─────────────────────────────────────────────
AUG_RESCALE         = 1.0 / 255
AUG_HORIZONTAL_FLIP = False
AUG_ROTATION_RANGE  = 10
AUG_BRIGHTNESS_RANGE= [0.9, 1.1]
AUG_ZOOM_RANGE      = 0.1

# ── Warmup ───────────────────────────────────────────────────
USE_WARMUP          = False
WARMUP_EPOCHS       = 0
WARMUP_INITIAL_LR   = 1e-5

# ── Segmentos y CV ───────────────────────────────────────────
SEGMENTS_ORDER      = ["pinky", "middle", "thumb", "wrist"]
USE_CROSS_VALIDATION= True
N_FOLDS             = 2

# ── Rutas ────────────────────────────────────────────────────
SEGMENTATION_MODEL  = "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
