# Propuesta: CNN Simple con Flatten para Fusión de Segmentos

## 1. Contexto y Problema Actual

### Arquitectura actual

El pipeline entrena **4 modelos independientes** (uno por segmento anatómico: pinky, middle, thumb, wrist) usando backbones preentrenados configurables (VGG16, DenseNet121, InceptionV3, ResNet50):

```
Imagen segmento
    ↓
Backbone (VGG16 / DenseNet121 / InceptionV3 / ResNet50)
    ↓
GlobalAveragePooling2D          ← colapsa mapa espacial a 1 valor por canal
    ↓
Dense(256, relu)
    ↓
Dropout(0.5)
    ↓
Dense(1, linear)  ──────────────── predicción escalar (meses)
```

```
4 predicciones escalares [pinky, middle, thumb, wrist]
    ↓
Concatenar
    ↓
Dense(128, relu)
    ↓
Dropout(0.5)
    ↓
Dense(1, linear)  ──────────────── predicción final
```

### Problema identificado

El modelo de fusión **solo recibe 4 escalares** (uno por segmento). Toda la información espacial y featurística que el backbone extrajo se descarta en `GlobalAveragePooling2D`. La fusión opera con una representación extremadamente comprimida, lo que limita su capacidad para combinar la información entre segmentos.

Adicionalmente, los backbones grandes (especialmente VGG16 e InceptionV3) introducen:
- Alta carga de parámetros no necesarios para regiones pequeñas y específicas
- Dependencia de pesos ImageNet que pueden no transferir bien a radiografías en escala de grises
- Mayor tiempo de entrenamiento y riesgo de overfitting en datasets pequeños

---

## 2. Propuesta: CNN Simple + Flatten

### Idea central

Reemplazar los backbones configurables por **CNNs simples entrenadas desde cero**, y sustituir `GlobalAveragePooling2D` por `Flatten`. Esto permite que el modelo de fusión reciba **vectores de características completos** en lugar de predicciones escalares, preservando toda la información extraída por cada CNN de segmento.

### Arquitectura propuesta para modelos de segmento

```
Entrada: (H, W, C)  [e.g. (112, 112, 3)]
    ↓
Conv2D(32, 3×3, padding='same') → BatchNorm → ReLU → MaxPool(2×2)
    ↓  [56×56×32]
Conv2D(64, 3×3, padding='same') → BatchNorm → ReLU → MaxPool(2×2)
    ↓  [28×28×64]
Conv2D(128, 3×3, padding='same') → BatchNorm → ReLU → MaxPool(2×2)
    ↓  [14×14×128]
Conv2D(256, 3×3, padding='same') → BatchNorm → ReLU → MaxPool(2×2)
    ↓  [7×7×256]
Flatten
    ↓  [7×7×256 = 12544 valores]
[Dense(512, relu) → Dropout]       ← cabeza para entrenamiento individual
    ↓
Dense(1, linear)  ──────────────── predicción individual (meses)
```

Los hiperparámetros (número de bloques, filtros por bloque, tamaño del kernel) son configurables desde `experiments/NN/config.py`.

### Arquitectura propuesta para el modelo de fusión

```
Segmento pinky  → CNN_pinky  → Flatten → feature_vector_pinky   [12544-dim]
Segmento middle → CNN_middle → Flatten → feature_vector_middle  [12544-dim]
Segmento thumb  → CNN_thumb  → Flatten → feature_vector_thumb   [12544-dim]
Segmento wrist  → CNN_wrist  → Flatten → feature_vector_wrist   [12544-dim]
                                                ↓
                              Concatenar todos [4×12544 = 50176-dim]
                              [+ género si USE_GENDER=True]
                                                ↓
                                       Dense(512, relu)
                                                ↓
                                          Dropout(0.5)
                                                ↓
                                       Dense(256, relu)
                                                ↓
                                          Dropout(0.3)
                                                ↓
                                     Dense(1, linear)  ── predicción final
```

El modelo de fusión **extrae las features del layer Flatten** de cada CNN de segmento (no la predicción final), y las combina en una sola capa de fusión profunda.

---

## 3. Ventajas frente a la arquitectura actual

| Aspecto | Arquitectura actual | Propuesta CNN Simple |
|---|---|---|
| Info. que llega a fusión | 4 escalares | 4 vectores completos (~12K-dim c/u) |
| Pérdida de info. espacial | Alta (GAP colapsa todo) | Ninguna (Flatten preserva posición) |
| Dependencia de ImageNet | Alta (backbones) | Ninguna (entrenamiento desde cero) |
| Parámetros por segmento | ~14M (VGG16) – ~7M (MobileNet) | ~3-5M (configurable) |
| Velocidad de entrenamiento | Lenta | Más rápida |
| Riesgo de overfitting | Alto (backbone grande, dataset pequeño) | Menor |
| Capacidad de fusión | Muy limitada (4 números) | Alta (vectores ricos) |

---

## 4. Plan de Implementación

### Fase 1 — Nueva función de arquitectura CNN

**Archivo:** `src/06_training.py` (o nuevo `src/utils/cnn_utils.py`)

Crear `create_simple_cnn_segment_model(cfg)`:
- Parámetros desde config: `CNN_FILTERS`, `CNN_BLOCKS`, `CNN_KERNEL_SIZE`, `CNN_DROPOUT`
- Retorna modelo Keras con cabeza de predicción individual (para entrenamiento aislado)
- El layer `Flatten` debe tener nombre fijo (`"flatten_features"`) para poder extraerlo en fusión

```python
def create_simple_cnn_segment_model(cfg):
    inputs = Input(shape=(*cfg.IMAGE_SIZE, 3))
    x = inputs
    for filters in cfg.CNN_FILTERS:  # e.g. [32, 64, 128, 256]
        x = Conv2D(filters, cfg.CNN_KERNEL_SIZE, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(2, 2)(x)
    x = Flatten(name="flatten_features")(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(cfg.CNN_DROPOUT)(x)
    output = Dense(1, activation='linear')(x)
    return Model(inputs, output)
```

### Fase 2 — Modificar la función de fusión

**Archivo:** `src/06_training.py` → `create_fusion_model_cnn(segment_paths, cfg, loss_fn)`

- Cargar cada modelo de segmento guardado
- Crear sub-modelo que termina en `"flatten_features"` (no en la predicción)
- Concatenar los 4 vectores Flatten
- Añadir dense layers de fusión más profundas

```python
def create_fusion_model_cnn(segment_paths, cfg, loss_fn):
    inputs = []
    feature_outputs = []
    for seg, path in zip(cfg.SEGMENTS_ORDER, segment_paths):
        seg_model = load_model(path, custom_objects=LOSS_MAP)
        feature_model = Model(
            inputs=seg_model.input,
            outputs=seg_model.get_layer("flatten_features").output,
            name=f"features_{seg}"
        )
        feature_model.trainable = False  # congelar durante fusión inicial
        inp = Input(shape=(*cfg.IMAGE_SIZE, 3), name=f"input_{seg}")
        inputs.append(inp)
        feature_outputs.append(feature_model(inp))

    x = Concatenate()(feature_outputs)
    # [+ género]
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)
    return Model(inputs=inputs, outputs=output)
```

### Fase 3 — Nuevos parámetros en config de experimento

Añadir al `experiments/NN/config.py`:

```python
# Arquitectura CNN simple
MODEL_TYPE = "simple_cnn"   # "simple_cnn" | "backbone"
CNN_FILTERS = [32, 64, 128, 256]
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.3
```

El switch `MODEL_TYPE` permite mantener compatibilidad con los experimentos anteriores basados en backbone.

### Fase 4 — Compatibilidad hacia atrás

- Usar `getattr(cfg, 'MODEL_TYPE', 'backbone')` para detectar qué arquitectura usar
- `create_segment_model(cfg)` → backbone (comportamiento actual)
- `create_simple_cnn_segment_model(cfg)` → nueva arquitectura CNN

Ambas funciones conviven en `06_training.py`; el `main()` selecciona según `MODEL_TYPE`.

### Fase 5 — Nuevo experimento de referencia

Crear `experiments/29/config.py` con:
- `MODEL_TYPE = "simple_cnn"`
- `CNN_FILTERS = [32, 64, 128, 256]`
- `IMAGE_SIZE = (112, 112)`
- Mismos `EPOCHS_SEGMENT`, `FUSION_EPOCHS`, `BATCH_SIZE`, `LOSS_FUNCTION_NAME` que exp. 26 (para comparación directa)

---

## 5. Consideraciones de Diseño

### Dimensión del vector Flatten

Con `IMAGE_SIZE = (112, 112)` y 4 bloques MaxPool(2×2):
- Mapa de características final: `(7, 7, 256)` → Flatten = **12544 valores** por segmento
- Fusión de 4 segmentos: **50176 valores** antes de las capas densas

Si este tamaño es excesivo (riesgo de overfitting), se puede añadir un `Dense(512)` antes de Flatten en el modelo de segmento para comprimir, o reducir `CNN_FILTERS`.

### Fine-tuning del modelo de fusión

Durante la segunda fase de entrenamiento (fine-tuning), se pueden descongelar las CNNs de segmento y entrenar el sistema completo end-to-end, ya que la arquitectura es mucho más liviana que los backbones actuales.

### Género

Si `USE_GENDER = True`, la entrada de género se concatena al vector fusionado (igual que en la arquitectura actual), antes de las capas Dense de fusión.

---

## 6. Experimentos Propuestos

| Exp | Filtros CNN | IMAGE_SIZE | Épocas seg. | Épocas fusión | Notas |
|---|---|---|---|---|---|
| 29 | [32,64,128,256] | 112×112 | 15 | 20 | Baseline CNN |
| 30 | [64,128,256,512] | 112×112 | 15 | 20 | CNN más ancha |
| 31 | [32,64,128,256] | 224×224 | 15 | 20 | Mayor resolución |
| 32 | [32,64,128,256] | 112×112 | 15 | 20 | Sin género |

Comparación directa con **Exp 26** (VGG16, sin imagenet) como baseline.

---

## 7. Archivos a Modificar / Crear

| Archivo | Cambio |
|---|---|
| `src/06_training.py` | Añadir `create_simple_cnn_segment_model()` y `create_fusion_model_cnn()` |
| `experiments/29/config.py` | Nuevo experimento CNN baseline |
| `config/experiment.py` | Añadir defaults para nuevos parámetros CNN (opcional) |

No se requiere modificar el pipeline de preprocesamiento (steps 01–04) ni los scripts de validación (07–08), ya que la interfaz de entrada/salida de los modelos es idéntica.
