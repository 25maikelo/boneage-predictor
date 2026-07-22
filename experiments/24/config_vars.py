# ================================
# VARIABLES DE CONFIGURACIÓN
# ================================

# Tamaño de imágenes y lotes
IMAGE_SIZE = tuple([224, 224])
BATCH_SIZE = 32

# Épocas para entrenamiento de segmentos y fusión
EPOCHS_SEGMENT = 15
FUSION_EPOCHS = 20
FINE_TUNING_EPOCHS = 10

# Tasa de aprendizaje y partición de datos
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
AGE_RANGE = tuple([24, 216])

# Uso de género y aumento de datos
USE_GENDER = True
USE_AUGMENTATION = True

# Función de pérdida a utilizar
LOSS_FUNCTION_NAME = "attention_loss"

# Selección de arquitectura base y pesos preentrenados (si se usan)
BASE_MODEL_CHOICE = "resnet50"
WEIGHTS = None

# Rutas de archivos y directorios
DATASET_PATH = "../training-data/dataset_analysis/balanced_dataset.csv"
ANALYSIS_DATASET_PATH = "../training-data/boneage-training-dataset.csv"
IMAGES_FOLDER = "../images/imagenes_ecualizadas"
SEGMENTS_FOLDER = "../images/segmentos_spatial"
MODEL_SAVE_PATH = "../models"
TRAINING_LOGS_PATH = "trainings"

# Orden de los segmentos
SEGMENTS_ORDER = ["pinky", "middle", "thumb", "wrist"]

# ================================
# VARIABLES PARA AUMENTO DE DATOS
# ================================
AUG_RESCALE = 1.0 / 255
AUG_HORIZONTAL_FLIP = False
AUG_ROTATION_RANGE = 20
AUG_BRIGHTNESS_RANGE = [0.8, 1.2]
AUG_ZOOM_RANGE = 0.2

# ================================
# VARIABLES PARA ARQUITECTURA DEL MODELO
# ================================
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
NUM_LAYERS_UNFREEZE = 10

# ================================
# Parámetro para la función de pérdida dinámica (dynamic_attention_loss)
# ================================
INITIAL_K = 3.0

# ================================
# VARIABLES PARA WARMUP Y OPTIMIZADOR
# ================================
USE_WARMUP = False            # True: usa warmup; False: no usar
WARMUP_EPOCHS = 5            # Número de épocas de warmup
WARMUP_INITIAL_LR = 1e-5     # Learning rate inicial durante warmup
OPTIMIZER_CHOICE = "adam"    # Opciones: "adam", "sgd", "adamw"
