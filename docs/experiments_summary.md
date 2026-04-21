# Resumen de Experimentos

> **Experimentos 0–16:** Las carpetas existen en el repositorio pero ninguna tiene `config.py`. Según el historial de git, los experimentos 17–28 fueron migrados desde `scripts/trainings/` (commit `7724a1d`, Feb 20), lo que indica que los experimentos anteriores existieron como scripts sueltos y nunca se formalizaron en la estructura actual.

**Leyenda de estado:**
- **Completado** — pipeline completo ejecutado (entrenamiento + validación + evaluación)
- **Incompleto** — entrenamiento iniciado pero pipeline sin terminar
- **Solo config** — configuración lista, sin ejecución
- **Ajuste de código** — réplica de un experimento previo motivada por corrección de bugs o cambios de infraestructura (no cambio conceptual)

---

## Fase 1 — Exploración de Backbones (Exps 17–28)

Arquitectura fija: `backbone` con segmentación en 4 regiones (pinky, middle, thumb, wrist) + fusión. Todas usan K-Fold CV=5, rango de edad variable, género activado salvo indicación.

| Exp | Estado | Backbone | Img | Rango edad (m) | Épocas (seg/fus/ft) | Warmup | Notas relevantes |
|-----|--------|----------|-----|----------------|----------------------|--------|-----------------|
| 17 | Completado | InceptionV3 | 299×299 | 94–168 | 5/5/5 | Sí | Loss `dynamic_attention_loss` (K=3.0); primera exploración |
| 18 | Completado | VGG16 | 224×224 | 94–168 | 7/5/5 | Sí | Loss `attention_loss`; LR 0.001 |
| 19 | Completado | ResNet50 | 112×112 | 94–168 | 7/20/10 | Sí | Fase de fusión extendida (20 épocas) |
| 20 | Completado | ResNet50 | 112×112 | 24–216 | 7/20/10 | Sí | Primer experimento con **rango completo** de edad |
| 21 | Completado | ResNet50 | 112×112 | 24–216 | 3/5/0 | No | Sin fine-tuning; configuración de prueba rápida |
| 22 | Completado | ResNet50 | 112×112 | 100–126 | 2/6/10 | No | Rango estrecho (middle range); prueba de especialización |
| 23 | Completado | ResNet50 | 112×112 | 24–216 | 15/20/10 | No | Entrenamiento pesado; primer exp con resultados completos (val + mex-val) |
| 24 | Completado | ResNet50 | 224×224 | 24–216 | 15/20/10 | No | Imágenes grandes con ResNet50 |
| 25 | Completado | ResNet50 | 224×224 | 24–216 | 15/20/10 | Sí | Igual a 24 + warmup; evalúa impacto del warmup |
| 26 | Completado | VGG16 | 112×112 | 24–216 | 15/20/10 | No | **Baseline oficial** de la Fase 1; suite completa de evaluación |
| 27 | Completado | DenseNet121 | 112×112 | 24–216 | 15/20/10 | No | Comparación DenseNet vs VGG16 (Exp 26) |
| 28 | Completado | InceptionV3 | 112×112 | 24–216 | 15/20/10 | No | Cierre de exploración de backbones con InceptionV3 |

---

## Fase 2 — CNN Simple vs Backbone, con/sin género (Exps 29–35)

Se introduce `simple_cnn` (4 bloques conv + Flatten + Dense). El foco es comparar arquitecturas y el impacto del género. Esta fase tuvo **dos iteraciones de código** motivadas por bugs en la fusión con género. Se añade también `backbone_vectors` (exp 35), nueva arquitectura que extrae vectores de características del backbone en lugar de escalares.

> **Nota sobre el dataset:** Todos los experimentos de esta fase usan el CSV raw (`boneage-training-dataset.csv`, 12,611 imágenes) filtrado por `AGE_RANGE=(24,216)`, resultando en ~12,499 imágenes con 160 clases de edad. El `balanced_dataset.csv` generado por el script 05 (11,783 imágenes, 36 edades con ≥50 muestras) no se usa en ningún experimento — la "restricción de edad" es únicamente por rango, no por frecuencia mínima.

| Exp | Estado | Tipo modelo | Género | Img | Épocas (seg/fus/ft) | MAE CV prom. | val_mae fusión | Notas relevantes |
|-----|--------|-------------|--------|-----|----------------------|--------------|----------------|-----------------|
| 29 | Completado | `simple_cnn` | Sí | 112×112 | 15/20/10 | — | — | **Baseline CNN**; réplica directa de Exp 26 con nueva arquitectura; múltiples runs de ajuste |
| 30 | Completado | `simple_cnn` | No | 112×112 | 15/20/10 | Pinky/Thumb/Wrist: ~123 m · Middle: ~36 m | — | Sin género; MAE anómalo en 3 segmentos sugiere artefacto de segmentación |
| 31 | Incompleto | `simple_cnn` | Sí | 112×112 | 15/20/10 | Pinky: 36.6 · Middle: 36.4 · Thumb: 36.6 · Wrist: 36.7 | — | Género restaura consistencia entre segmentos; entrenamiento no terminó (bug en fusión detectado después) |
| 32 | Completado | `backbone` (DenseNet121) | Sí | 112×112 | 15/20/10 | Pinky: 37.1 · Middle: 37.6 · Thumb: 37.0 · Wrist: 37.1 | — | Réplica de Exp 27 con infraestructura actualizada (CV + rutas segmentación); referencia backbone para comparar con CNN |
| 33 | Completado *(ajuste de código)* | `simple_cnn` | Sí | 112×112 | 15/20/10 | Pinky: 35.43 · Middle: 27.42 · Thumb: 25.55 · Wrist: 26.73 · **Prom: 28.78** | ~50 m (no convergió) | **Réplica corregida de Exp 31** — fix bug fusión `USE_GENDER`; fusión con vectores 50K-dim inestable |
| 34 | Completado *(ajuste de código)* | `backbone` (DenseNet121) | Sí | 112×112 | 15/20/10 | Pinky: 29.17 · Middle: 28.86 · Thumb: 23.53 · Wrist: 28.70 · **Prom: 27.57** | **~13.4 m** | **Réplica corregida de Exp 32** — fusión con 4 escalares converge muy bien |
| **35** | **Solo config** | `backbone_vectors` (DenseNet121) | Sí | 112×112 | 15/20/10 | — | — | **Nueva arquitectura** — fusión recibe 4×256 vectores del backbone; hipótesis: mejor que escalares (exp 34) |

---

---

## Fase 3 — Dataset Completo sin Filtro de Edad (Exps 36–38)

Réplicas exactas de los experimentos 33, 34 y 35 con la única diferencia de que `AGE_RANGE=(1,228)` incluye las 12,611 imágenes con las 160 clases de edad originales, incluyendo edades poco representadas (hasta 1 sola imagen). El objetivo es medir si el balanceo por rango de edad mejora o perjudica el rendimiento.

> **Diferencia clave vs Fase 2:** `AGE_RANGE=(24,216)` → `AGE_RANGE=(1,228)`. El resto de parámetros es idéntico.

| Exp | Estado | Tipo modelo | Imagen | Dataset | Épocas (seg/fus/ft) | MAE CV prom. | val_mae fusión | Comparar con |
|-----|--------|-------------|--------|---------|----------------------|--------------|----------------|--------------|
| **36** | **Solo config** | `simple_cnn` | 112×112 | 12,611 imgs · 160 edades | 15/20/10 | — | — | Exp 33 |
| **37** | **Solo config** | `backbone` (DenseNet121) | 112×112 | 12,611 imgs · 160 edades | 15/20/10 | — | — | Exp 34 |
| **38** | **Solo config** | `backbone_vectors` (DenseNet121) | 112×112 | 12,611 imgs · 160 edades | 15/20/10 | — | — | Exp 35 |

---

## Fase 4 — Dataset Balanceado (Exps 39–41)

Réplicas de los experimentos 33, 34 y 35 usando `balanced_dataset.csv` (11,783 imágenes · 36 edades · solo clases con ≥50 muestras). Es el único conjunto donde la distribución de edades es deliberadamente uniforme entre clases.

> **Diferencia clave vs Fase 2:** Se añade `DATASET_PATH = "data/training/dataset_analysis/balanced_dataset.csv"`. El resto de parámetros es idéntico a 33/34/35.

| Exp | Estado | Tipo modelo | Dataset | Épocas (seg/fus/ft) | MAE CV prom. | val_mae fusión | Comparar con |
|-----|--------|-------------|---------|----------------------|--------------|----------------|--------------|
| **39** | **Solo config** | `simple_cnn` | 11,783 imgs · 36 edades | 15/20/10 | — | — | Exp 33 (raw) · Exp 36 (completo) |
| **40** | **Solo config** | `backbone` (DenseNet121) | 11,783 imgs · 36 edades | 15/20/10 | — | — | Exp 34 (raw) · Exp 37 (completo) |
| **41** | **Solo config** | `backbone_vectors` (DenseNet121) | 11,783 imgs · 36 edades | 15/20/10 | — | — | Exp 35 (raw) · Exp 38 (completo) |

---

---

## Fase 5 — Extractores Descongelados (Exps 42–43)

Réplicas de los experimentos 36 y 38 con `FREEZE_EXTRACTORS = False`: los 4 extractores de segmento se entrenan junto con las capas de fusión desde el inicio de esa fase, en lugar de mantenerse congelados. El objetivo es evaluar si permitir que los extractores se adapten a la tarea de fusión mejora el rendimiento.

> **Diferencia clave vs Fase 3:** `FREEZE_EXTRACTORS = False` + `USE_WARMUP = True` (LR 1e-5 → 1e-3). Usan el dataset completo `AGE_RANGE=(1,228)` — mismo que exps 36–38.
>
> **Por qué no hay exp 4X para `backbone` escalar:** La fusión escalar recibe 4 valores numéricos; no hay "extractor" que descongelar — los modelos de segmento se usan completos (congelados durante fusión, descongelados parcialmente en fine-tuning). El concepto de descongelamiento de extractor aplica solo a `simple_cnn` y `backbone_vectors`.

| Exp | Estado | Tipo modelo | Dataset | Épocas (seg/fus/ft) | FREEZE_EXTRACTORS | MAE CV prom. | Comparar con |
|-----|--------|-------------|---------|----------------------|-------------------|--------------|--------------|
| **42** | **Solo config** | `simple_cnn` | Completo · 12,611 imgs (1–228 m) | 15/20/10 | `False` | — | Exp 36 (`True`) |
| **43** | **Solo config** | `backbone_vectors` (DenseNet121) | Completo · 12,611 imgs (1–228 m) | 15/20/10 | `False` | — | Exp 38 (`True`) |

---

## Matriz de comparación por arquitectura y dataset

| Arquitectura | Raw (24–216 m) | Completo (1–228 m) | Balanceado (≥50/edad) | Completo · extractor libre |
|---|---|---|---|---|
| `simple_cnn` | Exp 33 | Exp 36 | Exp 39 | Exp 42 |
| `backbone` DenseNet121 escalar | Exp 34 | Exp 37 | Exp 40 | — |
| `backbone_vectors` DenseNet121 | Exp 35 | Exp 38 | Exp 41 | Exp 43 |

---

## Exp 99 — Quick Test (Pipeline Completo)

| Exp | Estado | Tipo modelo | Img | Samples | Épocas | Notas relevantes |
|-----|--------|-------------|-----|---------|--------|-----------------|
| 99 | Solo config | `simple_cnn` | 64×64 | 80 | 2/2/1 | Validación rápida de todo el pipeline; filtros reducidos [16,32,64], K-Fold=2, sin género |

---

## Línea de tiempo de cambios de código relevantes

| Fecha | Commit | Cambio |
|-------|--------|--------|
| Feb 20 | `7724a1d` | Baseline Fase 1 consolidado (Exps 17–28) |
| Mar 11 | `5147fb7` | Introduce arquitectura `simple_cnn` |
| Mar 13 | `025bdd3` | Fix validación para modelos `simple_cnn` |
| Mar 14 | `aed3ca4` | Añade K-Fold CV y Exp 99 (quick test) |
| Apr 18 | `b60a66f` | Añade soporte de género en `simple_cnn` (Exps 31/32) |
| Apr 20 | `ca9e3a2` | **Fix bug en fusión con género** → Exps 33/34 (réplicas corregidas) |
| Apr 21 | `a7d1265` | Añade documentación MD (pipeline, arquitecturas, dataset report) |
| Apr 21 | `a7d1265` | Nueva arquitectura `backbone_vectors` + Exps 35–38 (dataset completo) |
| Apr 21 | *(actual)* | Flag `FREEZE_EXTRACTORS` en todos los configs + Exps 42–43 (extractores descongelados) |
