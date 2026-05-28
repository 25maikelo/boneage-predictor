# Plan de Exploración — Post Fase 5

## Objetivo

Una vez completados los experimentos 33–43, el objetivo es identificar la combinación óptima de arquitectura, dataset y hiperparámetros para maximizar el rendimiento en validación clínica (MAE en RSNA y MEX).

La exploración se divide en dos etapas secuenciales. No se avanza a la siguiente hasta tener conclusiones claras de la anterior.

---

## Etapa 1 — Selección del modelo base

### Criterio

Evaluar la matriz completa de experimentos 33–43 y seleccionar un único experimento ganador que sirva como referencia para todas las ablaciones posteriores.

### Métricas de decisión (en orden de prioridad)

1. MAE en validación RSNA (conjunto independiente)
2. MAE en validación MEX (generalización clínica)
3. MAE CV promedio de segmentos (calidad del extractor)
4. Estabilidad del entrenamiento (varianza entre folds)

### Matriz de comparación

| Arquitectura | Raw (24–216 m) | Completo (1–228 m) | Balanceado (≥50/edad) | Completo · extractor libre |
|---|---|---|---|---|
| `simple_cnn` | Exp 33 | Exp 36 | Exp 39 | Exp 42 |
| `backbone` DenseNet121 escalar | Exp 34 | Exp 37 | Exp 40 | — |
| `backbone_vectors` DenseNet121 | Exp 35 | Exp 38 | Exp 41 | Exp 43 |

### Resultado esperado

Un experimento de referencia (p.ej. "Exp 38") sobre el cual se realizarán todas las ablaciones de la Etapa 2.

---

## Etapa 2 — Ablaciones sobre el modelo ganador

Cada ablación modifica **una sola variable** respecto al experimento ganador. El orden está determinado por impacto esperado vs. costo computacional.

---

### 2.1 Ablación de género

**Variable:** `USE_GENDER`
**Valores:** `True` (baseline) vs. `False`
**Costo:** 1 experimento (~40 h)

**Motivación:** El exp 30 mostró colapso completo del MAE en 3 de 4 segmentos al remover el género en CNN simple. No se sabe si el mismo efecto ocurre en la arquitectura ganadora. Es la ablación más barata y su resultado puede descartar el género como variable necesaria.

**Resultado esperado:** Confirmar o descartar que el género es indispensable en la arquitectura ganadora.

---

### 2.2 Ablación de learning rate y épocas de fusión

**Variables:** `LEARNING_RATE` y `FUSION_EPOCHS`

**Grid:**

| LR \ Épocas fusión | 10 | 20 | 30 |
|---|---|---|---|
| 1e-4 | — | — | — |
| 5e-4 | — | baseline | — |
| 1e-3 | — | — | — |

**Costo:** hasta 8 experimentos adicionales (~320 h), pero se puede comenzar con los extremos (`1e-4/10` y `1e-3/30`) y añadir puntos intermedios solo si hay señal clara.

**Motivación:** Un LR mal calibrado introduce ruido en todas las ablaciones posteriores. Fijar el LR óptimo primero garantiza que las siguientes comparaciones sean válidas.

**Resultado esperado:** LR y número de épocas óptimos para la fase de fusión.

---

### 2.3 Ablación de fine-tuning depth

**Variable:** `NUM_LAYERS_UNFREEZE`
**Valores:** `0` (sin fine-tuning de backbone), `5`, `10` (baseline), `20`, `-1` (todas las capas)
**Costo:** 4 experimentos (~160 h)

**Motivación:** El fine-tuning parcial del backbone es una de las decisiones de diseño con mayor impacto en modelos de transferencia. Actualmente está fijo en 10 capas sin evidencia de que sea óptimo. Con `0` se mide si el fine-tuning aporta algo; con `all` se mide el riesgo de sobreajuste.

**Solo aplica si el ganador usa DenseNet121.** Si gana `simple_cnn`, esta ablación no es relevante.

**Resultado esperado:** Profundidad óptima de descongelamiento para el backbone.

---

### 2.4 Ablación de tamaño de imagen

**Variable:** `IMAGE_SIZE`
**Valores:** `(112, 112)` (baseline) vs. `(224, 224)`
**Costo:** 1 experimento (~80 h, aproximadamente el doble por mayor resolución)

**Motivación:** Los experimentos 24/25 evaluaron 224×224 con ResNet50 pero no con DenseNet121 ni con la nueva arquitectura de fusión. Imágenes más grandes pueden capturar detalles óseos finos que mejoran la predicción, especialmente en rangos de edad extremos.

**Condición para ejecutar:** Solo si las ablaciones 2.1–2.3 ya están saturadas (mejoras marginales < 1 mes MAE).

---

### 2.5 Incorporación de datos clínicos

**Variable:** Entradas adicionales al vector tabular de fusión
**Candidatos:** edad cronológica, talla, peso, z-score talla/edad, etnia

**Motivación:** La arquitectura actual recibe `[features_imagen, género]`. Agregar datos clínicos estructurados convierte el modelo en verdaderamente multimodal (imagen + datos clínicos), que es el estándar en radiología computacional de alto rendimiento. El impacto potencial es mayor que cualquier ajuste de hiperparámetro.

**Condición para ejecutar:** Requiere verificar disponibilidad de estos campos en los datasets RSNA y MEX. Si no están disponibles en RSNA, evaluar si el dataset MEX los tiene para una validación parcial.

**Costo:** Depende de disponibilidad de datos. Si están disponibles, 1–2 experimentos.

---

## Resumen de prioridades

| Prioridad | Ablación | Costo aprox. | Condición |
|---|---|---|---|
| 1 | Género (`USE_GENDER`) | ~40 h | Siempre |
| 2 | LR + épocas fusión | ~80–320 h | Siempre |
| 3 | Fine-tuning depth | ~160 h | Solo si ganador usa DenseNet |
| 4 | Tamaño de imagen | ~80 h | Solo si 1–3 están saturados |
| 5 | Datos clínicos adicionales | variable | Sujeto a disponibilidad |

---

## Principios de la exploración

- **Una variable a la vez.** Cambiar múltiples parámetros simultáneamente impide atribuir mejoras a causas específicas.
- **No grid search exhaustivo.** Comenzar con los extremos del rango y añadir puntos intermedios solo si hay señal.
- **Criterio de saturación.** Si una ablación muestra mejora < 0.5 meses MAE respecto al baseline, no justifica continuar en esa dirección.
- **Validación en MEX como criterio final.** El MAE en RSNA puede sobreestimar el rendimiento real. La validación en el dataset mexicano es el indicador clínico más relevante.
