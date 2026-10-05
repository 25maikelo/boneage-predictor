# Arquitecturas de Entrenamiento

El pipeline soporta cinco modos de entrenamiento controlados por `MODEL_TYPE` en el `config.py` del experimento.

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

## Modo 3: Backbone Vectors (`MODEL_TYPE = "backbone_vectors"`)

Variante del Modo 1 donde la fusión recibe el vector intermedio de 256 dimensiones (`backbone_features`) en lugar del escalar de salida. Los extractores pueden estar congelados (`FREEZE_EXTRACTORS=True`) o entrenarse junto con la fusión (`FREEZE_EXTRACTORS=False`).

### Modelo de segmento

Idéntico al Modo 1, con la diferencia de que la capa Dense intermedia tiene nombre explícito:

```
...
Dense(DENSE_UNITS, relu, name="backbone_features")   ← extracción de vector
    ↓
Dropout(DROPOUT_RATE)
    ↓
Dense(1, linear)   ← predicción individual (no se usa en fusión)
```

### Modelo de fusión (`create_fusion_model_backbone_vectors`)

```
input_pinky  (H,W,3) → feature_extractor_pinky  → backbone_features → [256] ──┐
input_middle (H,W,3) → feature_extractor_middle → backbone_features → [256] ──┤
input_thumb  (H,W,3) → feature_extractor_thumb  → backbone_features → [256] ──┼── Concatenate [1,024]
input_wrist  (H,W,3) → feature_extractor_wrist  → backbone_features → [256] ──┘
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

---

## Modo 4: Unified CNN (`MODEL_TYPE = "unified_cnn"`)

Las 4 ramas CNN se entrenan **end-to-end en una sola fase**, sin pipeline de segmentos + fusión. No hay entrenamiento previo de modelos individuales ni fine-tuning posterior. Toda la red se optimiza simultáneamente desde el inicio.

```
input_pinky  (H,W,3) ──► [Conv→BN→ReLU→Pool] × 4 ──► Flatten [12,544] ──┐
input_middle (H,W,3) ──► [Conv→BN→ReLU→Pool] × 4 ──► Flatten [12,544] ──┤
input_thumb  (H,W,3) ──► [Conv→BN→ReLU→Pool] × 4 ──► Flatten [12,544] ──┼── Concatenate [50,176]
input_wrist  (H,W,3) ──► [Conv→BN→ReLU→Pool] × 4 ──► Flatten [12,544] ──┘
                                                                            ↓
                                                        [Concatenate(género)] ← si USE_GENDER=True
                                                                            ↓
                                                               Dense(512, relu)
                                                                            ↓
                                                                  Dropout(0.3)
                                                                            ↓
                                                               Dense(256, relu)
                                                                            ↓
                                                                  Dropout(0.3)
                                                                            ↓
                                                            Dense(1, linear)  →  predicción (meses)
```

> **Diferencia clave vs simple_cnn:** En `simple_cnn` las ramas CNN se entrenan por separado (una por región) y luego se congela su salida para entrenar la fusión. En `unified_cnn` todas las ramas se entrenan simultáneamente con el mismo gradiente de la pérdida global.

---

## Modo 5: Whole-Hand (`MODEL_TYPE = "whole_hand"`)

Baseline sin segmentación anatómica, agregado para el Comentario 15 de la segunda ronda de
revisión (aislar el efecto de la segmentación en 4 regiones). Una sola rama backbone recibe la
mano completa (recorte + CLAHE, sin segmentar) en vez de los 4 segmentos. No hay fusión: es
funcionalmente un `create_segment_model` (igual arquitectura que el Modo 1) aplicado a una única
entrada de "segmento" que es la mano entera.

```
Entrada: (H, W, 3)  [imagen de mano completa, data/images/equalized/]
    ↓
Backbone preentrenado (mismo que Modo 1, WEIGHTS=None en los experimentos usados)
    ├─ include_top=False
    └─ últimas NUM_LAYERS_UNFREEZE capas entrenables (fase 1)
    ↓
GlobalAveragePooling2D
    ↓
[Concatenate(género)]       ← solo si USE_GENDER=True
    ↓
Dense(DENSE_UNITS, relu)
    ↓
Dropout(DROPOUT_RATE)
    ↓
Dense(1, linear)  ──────── predicción de edad ósea (meses)
```

Entrenamiento en dos fases (`train_whole_hand` en `src/06_training.py`), espejo del protocolo de
fusión para que la comparación sea controlada: fase 1 igual a la fase de segmento del Modo 1
(`NUM_LAYERS_UNFREEZE` capas descongeladas, `EPOCHS_SEGMENT` épocas); fase 2 de fine-tuning con
todo el backbone descongelado a `LEARNING_RATE/10` (`FINE_TUNING_EPOCHS` épocas). Mismo split
(`random_state=42`), mismo dataset balanceado, misma función de pérdida que el experimento de
fusión de referencia. Ver experimento 60 y
[`docs/results/experimentos_adicionales/analisis.md`](../results/experimentos_adicionales/analisis.md#15-ablación-insuficiente-falta-baseline-whole-hand)
para el resultado (la segmentación aporta poco en validación interna pero significativamente bajo
distribution shift externo).

---

## Comparativa

| Aspecto | backbone | simple_cnn | backbone_vectors | unified_cnn | whole_hand |
|---|---|---|---|---|---|
| Info. a fusión | 4 escalares | 4 × 12K flatten | 4 × 256 vectores | — (end-to-end) | — (sin fusión, 1 rama) |
| Fases de entrenamiento | 3 (seg + fusión + ft) | 3 (seg + fusión + ft) | 3 (seg + fusión + ft) | 1 (todo junto) | 2 (entrenamiento + ft) |
| Pesos iniciales | ImageNet (opcional) | Desde cero | ImageNet (opcional) | Desde cero | ImageNet (opcional) |
| Parámetros por segmento | ~7M (DenseNet121) | ~3–5M | ~7M (DenseNet121) | ~3–5M | ~7M (DenseNet121) |
| FREEZE_EXTRACTORS | N/A | Sí | Sí | N/A | N/A |
| Soporte USE_GENDER | Sí | Sí | Sí | Sí | Sí |
| Entrada | 4 segmentos | 4 segmentos | 4 segmentos | 4 segmentos | mano completa |

---

## Parámetros de Configuración

```python
MODEL_TYPE          = "simple_cnn"   # "simple_cnn" | "backbone" | "backbone_vectors" | "unified_cnn" | "whole_hand"

# Solo para simple_cnn, backbone_vectors y unified_cnn
CNN_FILTERS         = [32, 64, 128, 256]
CNN_KERNEL_SIZE     = 3
CNN_DROPOUT         = 0.3

# Solo para backbone y backbone_vectors
BASE_MODEL_CHOICE   = "densenet121"  # "vgg16" | "densenet121" | "inceptionv3" | "resnet50"
WEIGHTS             = None           # None | "imagenet"
NUM_LAYERS_UNFREEZE = 10

# Para simple_cnn y backbone_vectors
FREEZE_EXTRACTORS   = True           # False = extractores se entrenan con la fusión

# Compartidos
USE_GENDER          = True
IMAGE_SIZE          = (112, 112)
DENSE_UNITS         = 256
DROPOUT_RATE        = 0.5
```
