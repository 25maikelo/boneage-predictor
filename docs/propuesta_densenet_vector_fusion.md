# Propuesta: DenseNet121 con Fusión por Vectores de Características

## Motivación

Los experimentos actuales muestran una diferencia marcada entre las dos arquitecturas:

| Experimento | Arquitectura | Fusión recibe | val_mae fusión |
|---|---|---|---|
| 33 | CNN simple | 4 vectores de 12,544 dims | ~50 m (no convergió) |
| 34 | DenseNet121 | 4 escalares (predicciones) | ~13 m |

El backbone de DenseNet121 produce mejores representaciones, pero la fusión solo recibe 4 escalares — descartando toda la información espacial aprendida por el backbone. La propuesta es extraer el vector de características intermedio del backbone (en lugar del escalar final) y usarlo como entrada a la fusión.

---

## Arquitectura propuesta

### Modelo de segmento

Sin cambios funcionales. Se agrega únicamente el nombre `"backbone_features"` al layer Dense intermedio para poder extraerlo después.

```
Entrada: (112, 112, 3)
    ↓
DenseNet121 (include_top=False, últimas 10 capas entrenables)
    ↓
GlobalAveragePooling2D                   → [1024]
    ↓
[Concatenate(género)]                    ← si USE_GENDER=True → [1025]
    ↓
Dense(256, relu, name="backbone_features")   → [256]
    ↓
Dropout(0.5)
    ↓
Dense(1, linear, name="boneage_output")
```

### Extractor de características para fusión

En lugar de usar la predicción escalar `boneage_output`, se construye un sub-modelo que termina en `backbone_features`:

```python
feature_extractor = tf.keras.models.Model(
    inputs=seg_model.inputs[0],   # solo imagen (igual que CNN simple)
    outputs=seg_model.get_layer("backbone_features").output,
    name=f"feature_extractor_{seg}",
)
feature_extractor.trainable = False
```

Cada segmento aporta un vector de **256 dims**. Con 4 segmentos: **1,024 dims** de entrada a la fusión (vs 4 escalares en exp 34, vs ~50K en exp 33).

### Modelo de fusión

```
feature_extractor_pinky  → [256] ──┐
feature_extractor_middle → [256] ──┤
feature_extractor_thumb  → [256] ──┼── Concatenate → [1,024]
feature_extractor_wrist  → [256] ──┘
                                        ↓
                           [Concatenate(género)]   ← si USE_GENDER=True → [1,025]
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

## Comparativa de las tres arquitecturas

| Aspecto | CNN simple (33) | Backbone escalar (34) | **Backbone vectores (propuesta)** |
|---|---|---|---|
| Extractor por segmento | CNN desde cero | DenseNet121 (ImageNet opt.) | DenseNet121 (ImageNet opt.) |
| Fusión recibe | 4 × Flatten(12,544) | 4 × escalar | 4 × Dense(256) |
| Dims de entrada a fusión | ~50,176 | 4 | **1,024** |
| Info espacial en fusión | Alta (sin comprimir) | Ninguna | Comprimida (256-dim) |
| Calidad de features | Baja (sin preentrenamiento) | Alta → colapsada a 1 valor | **Alta → preservada en 256** |
| Riesgo de sobreajuste en fusión | Alto (50K dims) | Bajo | Moderado |
| Parámetros fusión (aprox.) | ~26M | ~0.5K | **~530K** |

---

## Implementación

### Cambio en `build_backbone_segment_model` (06_training.py)

Agregar `name="backbone_features"` al Dense intermedio:

```python
# línea ~167 — actualmente:
x = tf.keras.layers.Dense(cfg.DENSE_UNITS, activation="relu")(x)

# propuesta:
x = tf.keras.layers.Dense(cfg.DENSE_UNITS, activation="relu",
                           name="backbone_features")(x)
```

### Nueva función `create_fusion_model_backbone_vectors`

Análoga a `create_fusion_model_cnn` pero extrayendo `backbone_features` en lugar de `flatten_features`:

```python
def create_fusion_model_backbone_vectors(segment_paths, cfg, loss_fn):
    feature_outputs = []
    inputs = []
    for seg, path in zip(cfg.SEGMENTS_ORDER, segment_paths):
        seg_model = tf.keras.models.load_model(path, ...)
        feature_extractor = tf.keras.models.Model(
            inputs=seg_model.inputs[0],
            outputs=seg_model.get_layer("backbone_features").output,
            name=f"feature_extractor_{seg}",
        )
        feature_extractor.trainable = False
        inp = tf.keras.layers.Input(shape=(*cfg.IMAGE_SIZE, 3), name=f"input_{seg}")
        inputs.append(inp)
        feature_outputs.append(feature_extractor(inp))

    combined = tf.keras.layers.Concatenate()(feature_outputs)  # [1,024]

    if cfg.USE_GENDER:
        gender_in = tf.keras.layers.Input(shape=(1,), name="gender_input")
        inputs.append(gender_in)
        combined = tf.keras.layers.Concatenate()([combined, gender_in])

    x = tf.keras.layers.Dense(512, activation="relu")(combined)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="boneage_output")(x)
    return tf.keras.models.Model(inputs=inputs, outputs=out,
                                  name="fusion_model_backbone_vectors")
```

### Nuevo `MODEL_TYPE`

Agregar `"backbone_vectors"` como tercer tipo en `config.py` y en el dispatch de `06_training.py`:

```python
MODEL_TYPE = "backbone_vectors"   # nuevo valor

# en 06_training.py (~línea 644):
if cfg.MODEL_TYPE == "simple_cnn":
    fusion = create_fusion_model_cnn(seg_paths, cfg, loss_fn)
elif cfg.MODEL_TYPE == "backbone_vectors":
    fusion = create_fusion_model_backbone_vectors(seg_paths, cfg, loss_fn)
else:
    fusion = create_fusion_model(seg_paths, cfg, loss_fn)
```

---

## Experimento sugerido

**Exp 35** — réplica de exp 34 con `MODEL_TYPE = "backbone_vectors"`:

```python
MODEL_TYPE          = "backbone_vectors"
BASE_MODEL_CHOICE   = "densenet121"
WEIGHTS             = None
DENSE_UNITS         = 256       # dimensión del vector extraído
DROPOUT_RATE        = 0.5
NUM_LAYERS_UNFREEZE = 10

EPOCHS_SEGMENT      = 15
FUSION_EPOCHS       = 20
FINE_TUNING_EPOCHS  = 10
LEARNING_RATE       = 0.001
USE_GENDER          = True
USE_AUGMENTATION    = True
LOSS_FUNCTION_NAME  = "attention_loss"
```

El único parámetro que vale la pena explorar adicionalmente es `DENSE_UNITS`: valores de 128 o 512 cambian el balance entre compresión y capacidad del vector extraído.

---

## Hipótesis

El modelo debería superar a exp 34 porque la fusión recibe 256 características semánticas por segmento en lugar de un único valor escalar, permitiéndole aprender combinaciones entre regiones anatómicas que un escalar no puede capturar. Al mismo tiempo, los 1,024 dims son manejables (vs los ~50K del CNN simple), reduciendo el riesgo de sobreajuste en la cabeza de fusión.
