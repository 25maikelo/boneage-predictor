# Arquitecturas de Entrenamiento

El pipeline soporta dos modos de entrenamiento controlados por `MODEL_TYPE` en el `config.py` del experimento.

---

## Modo 1: Backbone (`MODEL_TYPE = "backbone"`)

### Modelo de segmento

```
Entrada: (H, W, 3)
    ↓
Backbone preentrenado (VGG16 | DenseNet121 | InceptionV3 | ResNet50)
    ├─ include_top=False
    └─ últimas NUM_LAYERS_UNFREEZE capas entrenables
    ↓
GlobalAveragePooling2D
    ↓
[Concatenate(género)]       ← solo si USE_GENDER=True
    ↓
Dense(DENSE_UNITS, relu)
    ↓
Dropout(DROPOUT_RATE)
    ↓
Dense(1, linear)  ──────── predicción individual (meses)
```

### Modelo de fusión (`create_fusion_model`)

```
output_pinky  (escalar) ──┐
output_middle (escalar) ──┤
output_thumb  (escalar) ──┼── Concatenate [+ género] ── Dense(128, relu) ── Dropout(0.5) ── Dense(1, linear)
output_wrist  (escalar) ──┘
```

---

## Modo 2: CNN Simple (`MODEL_TYPE = "simple_cnn"`)

### Manejo de canales

Las imágenes son escala de grises cargadas con `cv2.imread` y convertidas a RGB. Los 3 canales resultantes son idénticos (R = G = B = gray), lo que permite usar `input_shape=(H, W, 3)` sin modificaciones al pipeline.

### Modelo de segmento

```
Entrada: (H, W, 3)  [e.g. (112, 112, 3)]
    ↓
Conv2D(32, 3×3, padding='same') → BatchNormalization → ReLU → MaxPool(2×2)    [56×56×32]
    ↓
Conv2D(64, 3×3, padding='same') → BatchNormalization → ReLU → MaxPool(2×2)    [28×28×64]
    ↓
Conv2D(128, 3×3, padding='same') → BatchNormalization → ReLU → MaxPool(2×2)   [14×14×128]
    ↓
Conv2D(256, 3×3, padding='same') → BatchNormalization → ReLU → MaxPool(2×2)   [7×7×256]
    ↓
Flatten(name="flatten_features")    [12,544 valores con IMAGE_SIZE=(112,112)]
    ↓
[Concatenate(género)]               ← solo si USE_GENDER=True
    ↓
Dense(512, relu)
    ↓
Dropout(CNN_DROPOUT)
    ↓
Dense(1, linear)  ──────────────── predicción individual (meses)
```

### Modelo de fusión (`create_fusion_model_cnn`)

```
input_pinky  (H,W,3) → feature_extractor_pinky  → flatten_features → [12,544]  ──┐
input_middle (H,W,3) → feature_extractor_middle → flatten_features → [12,544]  ──┤
input_thumb  (H,W,3) → feature_extractor_thumb  → flatten_features → [12,544]  ──┼── Concatenate [50,176]
input_wrist  (H,W,3) → feature_extractor_wrist  → flatten_features → [12,544]  ──┘
                                                                                    ↓
                                                              [Concatenate(género)] ← si USE_GENDER=True
                                                                                    ↓
                                                                       Dense(512, relu)
                                                                                    ↓
                                                                          Dropout(0.5)
                                                                                    ↓
                                                                       Dense(256, relu)
                                                                                    ↓
                                                                          Dropout(0.3)
                                                                                    ↓
                                                                    Dense(1, linear)
```

> Los `feature_extractor_*` usan `seg_model.inputs[0]` (solo imagen) porque cuando `USE_GENDER=True` el modelo de segmento tiene 2 entradas. El layer `flatten_features` solo depende de la imagen.

---

## Comparativa

| Aspecto | Backbone | CNN Simple |
|---|---|---|
| Info. que llega a fusión | 4 escalares | 4 vectores (~12K-dim c/u) |
| Pérdida de info. espacial | Alta (GAP colapsa todo) | Ninguna (Flatten preserva posición) |
| Pesos iniciales | ImageNet (opcional) | Desde cero |
| Parámetros por segmento | ~7M (DenseNet121) | ~3–5M |
| Velocidad por época | ~400s | ~350s |
| Soporte USE_GENDER | Sí | Sí |

---

## Parámetros de Configuración

```python
MODEL_TYPE          = "simple_cnn"   # "simple_cnn" | "backbone"

# Solo para simple_cnn
CNN_FILTERS         = [32, 64, 128, 256]
CNN_KERNEL_SIZE     = 3
CNN_DROPOUT         = 0.3

# Solo para backbone
BASE_MODEL_CHOICE   = "densenet121"  # "vgg16" | "densenet121" | "inceptionv3" | "resnet50"
WEIGHTS             = None           # None | "imagenet"
NUM_LAYERS_UNFREEZE = 10

# Compartidos
USE_GENDER          = True
IMAGE_SIZE          = (112, 112)
DENSE_UNITS         = 256
DROPOUT_RATE        = 0.5
```
