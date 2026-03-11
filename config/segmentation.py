"""
Configuración del modelo de segmentación de mano (U-Net + MobileNetV2).
Modifica aquí los parámetros antes de ejecutar:

    python src/preprocessing/01_train_hand_detector.py

Las rutas de entrada/salida vienen de config/paths.py.
"""

# Tamaño de entrada al modelo
IMAGE_SIZE = (224, 224)

# Número de clases: 0=fondo, 1=pinky, 2=middle, 3=thumb, 4=wrist
NUM_CLASSES = 5

# Entrenamiento
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4

# Backbone del encoder (actualmente solo MobileNetV2 soportado)
ENCODER = "mobilenetv2"

# Pesos del encoder
ENCODER_WEIGHTS = "imagenet"   # None | "imagenet"
