# ============================================================
# Configuración del Experimento 26
# Solo hiperparámetros — las rutas vienen de config/paths.py
# ============================================================

# Arquitectura
IMAGE_SIZE = (112, 112)
BASE_MODEL_CHOICE = "vgg16"   # vgg16 | densenet121 | inceptionv3 | resnet50
WEIGHTS = None                 # None | "imagenet"
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
NUM_LAYERS_UNFREEZE = 10

# Entrenamiento
BATCH_SIZE = 32
EPOCHS_SEGMENT = 15
FUSION_EPOCHS = 20
FINE_TUNING_EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER_CHOICE = "adam"      # adam | sgd | adamw
TEST_SPLIT = 0.2

# Datos
AGE_RANGE = (24, 216)          # meses
USE_GENDER = True
USE_AUGMENTATION = True

# Función de pérdida
LOSS_FUNCTION_NAME = "attention_loss"
# Opciones: attention_loss | dynamic_attention_loss | custom_mse_loss | custom_huber_loss
INITIAL_K = 3.0                # Solo aplica a dynamic_attention_loss

# Augmentation
AUG_RESCALE = 1.0 / 255
AUG_HORIZONTAL_FLIP = False
AUG_ROTATION_RANGE = 20
AUG_BRIGHTNESS_RANGE = [0.8, 1.2]
AUG_ZOOM_RANGE = 0.2

# Warmup
USE_WARMUP = False
WARMUP_EPOCHS = 5
WARMUP_INITIAL_LR = 1e-5

# Segmentos (orden importa para la fusión)
SEGMENTS_ORDER = ["pinky", "middle", "thumb", "wrist"]
