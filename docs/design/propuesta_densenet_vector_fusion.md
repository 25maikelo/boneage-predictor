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

En lugar de usar la predicción escalar `boneage_output`, se construye un sub-modelo que termina en la capa `backbone_features` (ver implementación real en `create_fusion_model_backbone_vectors`, `src/06_training.py`). Cada segmento aporta un vector de **256 dims**. Con 4 segmentos: **1,024 dims** de entrada a la fusión (vs 4 escalares en exp 34, vs ~50K en exp 33).

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

Ya incorporada al pipeline como `MODEL_TYPE = "backbone_vectors"` (`src/06_training.py`:
`create_fusion_model_backbone_vectors`, dispatch en `main()`). Ver
[`arquitecturas.md`](arquitecturas.md#modo-3-backbone-vectors-model_type--backbone_vectors) para
el diagrama y parámetros actuales; el código de este documento queda solo como contexto histórico
de la propuesta original y puede haber divergido del código real.

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
