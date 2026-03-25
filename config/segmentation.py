"""
Configuración del modelo de segmentación de mano.
Modifica aquí los parámetros antes de ejecutar:

    python src/preprocessing/01_train_hand_detector.py

Las rutas de entrada/salida vienen de config/paths.py.
"""

# ─── Imagen ───────────────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)

# Número de canales de entrada: 1 → escala de grises  |  3 → RGB
# Para arquitecturas con backbone MobileNetV2 ("unet_mobilenetv2", "mobilenetv2"),
# si se elige INPUT_CHANNELS = 1 el canal único se replica 3 veces (tf.repeat)
# antes del backbone, preservando la compatibilidad con pesos imagenet.
# Para "unet" y "mobilenetv2_blocks" los canales se pasan directamente.
INPUT_CHANNELS = 3

NUM_CLASSES = 5           # 0=fondo, 1=pinky, 2=middle, 3=thumb, 4=wrist

# ─── Entrenamiento ────────────────────────────────────────────────────────────
BATCH_SIZE = 8
EPOCHS     = 30

# ─── Arquitectura ─────────────────────────────────────────────────────────────
# Opciones disponibles:
#
#   "unet"               → U-Net puro, sin backbone preentrenado.
#                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE se IGNORAN.
#
#   "unet_mobilenetv2"   → U-Net con encoder MobileNetV2 y skip connections.
#                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE aplican.
#
#   "mobilenetv2_sym"    → MobileNetV2 encoder + decoder simétrico inverted-residual.
#                          El decoder espeja la estructura interna del encoder:
#                          cada etapa usa expand→upsample+depthwise→project,
#                          con el mismo schedule de canales invertido (320→160→96→64→32→16).
#                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE aplican.
#
#   "unet_resnet50"      → U-Net con encoder ResNet50 y skip connections.
#                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE aplican.
#                          Soporta ENCODER_WEIGHTS = "imagenet" | "radimagenet" | None.
#                          Para "radimagenet": requiere models/pretrained/radimagenet_resnet50.h5
#
#   "unet_densenet121"   → U-Net con encoder DenseNet121 y skip connections.
#                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE aplican.
#                          Soporta ENCODER_WEIGHTS = "imagenet" | "radimagenet" | None.
#                          Para "radimagenet": requiere models/pretrained/radimagenet_densenet121.h5
#
ARCHITECTURE = "unet_mobilenetv2"

# Pesos del encoder.
# Aplica a todas las arquitecturas con backbone (unet_mobilenetv2, mobilenetv2_sym,
# unet_resnet50, unet_densenet121). Para "unet" siempre se entrena desde cero.
#
#   "imagenet"      → Pesos preentrenados en ImageNet (descarga automática de Keras).
#   "radimagenet"   → Pesos preentrenados en RadImageNet (Mei et al. 2022).
#                     Requiere el archivo .h5 en models/pretrained/.
#                     Descarga: https://github.com/BMEII-AI/RadImageNet
#   None            → Entrenamiento desde cero.
#
ENCODER_WEIGHTS = "imagenet"   # None | "imagenet" | "radimagenet"

# Permite ajuste fino del backbone durante el entrenamiento.
# Solo aplica a arquitecturas con backbone. Para "unet" no tiene efecto.
BASE_MODEL_TRAINABLE = True

# ─── Data Augmentation ────────────────────────────────────────────────────────
# Activa o desactiva el data augmentation en el conjunto de entrenamiento.
DATA_AUGMENTATION = False

# Transformaciones geométricas — se aplican igual a imagen Y máscara.
AUG_ROTATION_RANGE     = 15      # grados máximos de rotación
AUG_WIDTH_SHIFT_RANGE  = 0.1     # fracción del ancho para desplazamiento horizontal
AUG_HEIGHT_SHIFT_RANGE = 0.1     # fracción de la altura para desplazamiento vertical
AUG_ZOOM_RANGE         = 0.1     # factor de zoom (0.1 → ±10 %)
AUG_HORIZONTAL_FLIP    = False
AUG_VERTICAL_FLIP      = False
AUG_SHEAR_RANGE        = 0.05    # ángulo de corte en radianes

# Transformaciones fotométricas — solo se aplican a la imagen, NO a la máscara.
# Coloca None para desactivar el ajuste de brillo.
AUG_BRIGHTNESS_RANGE = (0.8, 1.2)   # rango multiplicativo de brillo
