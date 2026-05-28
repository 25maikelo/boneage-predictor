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
| 33 | ✅ Completado *(ajuste de código)* | `simple_cnn` | Sí | 112×112 | 15/20/10 | Pinky: 35.43 · Middle: 27.42 · Thumb: 25.55 · Wrist: 26.73 · **Prom: 28.78** | ~50 m (no convergió) | **Réplica corregida de Exp 31** — fix bug fusión `USE_GENDER`; fusión con vectores 50K-dim inestable |
| 34 | ✅ Completado *(ajuste de código)* | `backbone` (DenseNet121) | Sí | 112×112 | 15/20/10 | Pinky: 29.17 · Middle: 28.86 · Thumb: 23.53 · Wrist: 28.70 · **Prom: 27.57** | **~13.4 m** | **Réplica corregida de Exp 32** — fusión con 4 escalares converge muy bien |
| **35** | ✅ Completado | `backbone_vectors` (DenseNet121) | Sí | 112×112 | 15/20/10 | Pinky: 23.42 · Middle: 27.15 · Thumb: 22.87 · Wrist: 32.84 · **Prom: 26.57** | 27.6 m | — |

---

---

## Fase 3 — Dataset Completo sin Filtro de Edad (Exps 36–38)

Réplicas exactas de los experimentos 33, 34 y 35 con la única diferencia de que `AGE_RANGE=(1,228)` incluye las 12,611 imágenes con las 160 clases de edad originales, incluyendo edades poco representadas (hasta 1 sola imagen). El objetivo es medir si el balanceo por rango de edad mejora o perjudica el rendimiento.

> **Diferencia clave vs Fase 2:** `AGE_RANGE=(24,216)` → `AGE_RANGE=(1,228)`. El resto de parámetros es idéntico.

| Exp | Estado | Tipo modelo | Imagen | Dataset | Épocas (seg/fus/ft) | MAE CV prom. | val_mae fusión | Comparar con |
|-----|--------|-------------|--------|---------|----------------------|--------------|----------------|--------------|
| 36 | ✅ Completado | `simple_cnn` | 112×112 | 12,611 imgs · 160 edades | 15/20/10 | Pinky: 36.07 · Middle: 34.10 · Thumb: 32.70 · Wrist: 36.09 · **Prom: 34.74** | **29.48 m** (fine-tuning) | Exp 33 |
| **37** | ✅ Completado | `backbone` (DenseNet121) | 112×112 | 12,611 imgs · 160 edades | 15/20/10 | Pinky: 28.99 · Middle: 30.96 · Thumb: 29.48 · Wrist: 29.30 · **Prom: 29.68** | **6.8 m** | Exp 34 |
| **38** | ✅ Completado | `backbone_vectors` (DenseNet121) | 112×112 | 12,611 imgs · 160 edades | 15/20/10 | Pinky: 33.07 · Middle: 32.56 · Thumb: 28.02 · Wrist: 31.85 · **Prom: 31.38** | 30.3 m | Exp 35 |

---

## Fase 4 — Dataset Balanceado (Exps 39–41)

Réplicas de los experimentos 33, 34 y 35 usando `balanced_dataset.csv` (11,783 imágenes · 36 edades · solo clases con ≥50 muestras). Es el único conjunto donde la distribución de edades es deliberadamente uniforme entre clases.

> **Diferencia clave vs Fase 2:** Se añade `DATASET_PATH = "data/training/dataset_analysis/balanced_dataset.csv"`. El resto de parámetros es idéntico a 33/34/35.

| Exp | Estado | Tipo modelo | Dataset | Épocas (seg/fus/ft) | MAE CV prom. | val_mae fusión | Comparar con |
|-----|--------|-------------|---------|----------------------|--------------|----------------|--------------|
| **39** | ✅ Completado | `simple_cnn` | 11,783 imgs · 36 edades | 15/20/10 | Pinky: 33.37 · Middle: 27.88 · Thumb: 22.93 · Wrist: 34.25 · **Prom: 29.61** | 41.8 m | Exp 33 (raw) · Exp 36 (completo) |
| **40** | ✅ Completado | `backbone` (DenseNet121) | 11,783 imgs · 36 edades | 15/20/10 | Pinky: 23.60 · Middle: 26.43 · Thumb: 21.09 · Wrist: 33.81 · **Prom: 26.23** | **9.0 m** | Exp 34 (raw) · Exp 37 (completo) |
| **41** | ✅ Completado | `backbone_vectors` (DenseNet121) | 11,783 imgs · 36 edades | 15/20/10 | Pinky: 28.90 · Middle: 27.19 · Thumb: 21.58 · Wrist: 29.29 · **Prom: 26.74** | 26.9 m | Exp 35 (raw) · Exp 38 (completo) |

---

---

## Fase 5 — Extractores Descongelados (Exps 42–43)

Réplicas de los experimentos 36 y 38 con `FREEZE_EXTRACTORS = False`: los 4 extractores de segmento se entrenan junto con las capas de fusión desde el inicio de esa fase, en lugar de mantenerse congelados. El objetivo es evaluar si permitir que los extractores se adapten a la tarea de fusión mejora el rendimiento.

> **Diferencia clave vs Fase 3:** `FREEZE_EXTRACTORS = False` + `USE_WARMUP = True` (LR 1e-5 → 1e-3). Usan el dataset completo `AGE_RANGE=(1,228)` — mismo que exps 36–38.
>
> **Por qué no hay exp 4X para `backbone` escalar:** La fusión escalar recibe 4 valores numéricos; no hay "extractor" que descongelar — los modelos de segmento se usan completos (congelados durante fusión, descongelados parcialmente en fine-tuning). El concepto de descongelamiento de extractor aplica solo a `simple_cnn` y `backbone_vectors`.

| Exp | Estado | Tipo modelo | Dataset | Épocas (seg/fus/ft) | FREEZE_EXTRACTORS | MAE CV prom. | Comparar con |
|-----|--------|-------------|---------|----------------------|-------------------|--------------|--------------|
| **42** | ✅ Completado | `simple_cnn` | Completo · 12,611 imgs (1–228 m) | 15/20/10 | `False` | Pinky: 35.72 · Middle: 33.99 · Thumb: 35.23 · Wrist: 27.49 · **Prom: 33.11** | Exp 36 (`True`) |
| **43** | ✅ Completado | `backbone_vectors` (DenseNet121) | Completo · 12,611 imgs (1–228 m) | 15/20/10 | `False` | Pinky: 34.32 · Middle: 31.47 · Thumb: 29.22 · Wrist: 31.16 · **Prom: 31.54** | Exp 38 (`True`) |

---

---

## Fase 6 — CNN Unificada end-to-end (Exps 44–46)

Nueva arquitectura `unified_cnn`: 4 ramas CNN separadas (una por segmento) entrenadas **end-to-end** sobre bone age directamente, sin el pipeline de dos etapas (entrenar segmentos → entrenar fusión). Cada rama tiene la misma estructura que `simple_cnn` (bloques Conv2D + BN + ReLU + MaxPool → Flatten). Las 4 salidas se concatenan junto con el género y se pasan por dos capas Dense antes de la predicción final.

> **Diferencia clave vs `simple_cnn`:** eliminación del pipeline de dos fases — la optimización es directamente sobre la tarea final desde el inicio. Los parámetros CNN son idénticos a exps 33/36/39 para aislar el efecto de la arquitectura.

| Exp | Estado | Dataset | Épocas (CV) | MAE CV prom. | Comparar con |
|-----|--------|---------|-------------|--------------|--------------|
| 44 | ✅ Completado | Raw · ~12,499 imgs (24–216 m) | 30 | **19.5 m** | Exp 33 (`simple_cnn` raw) |
| 45 | ✅ Completado | Completo · 12,611 imgs (1–228 m) | 30 | **30.02 ± 7.16 m** | Exp 36 (`simple_cnn` completo) |
| 46 | ✅ Completado | Balanceado · 11,783 imgs (≥50/edad) | 30 | **23.2 m** | Exp 39 (`simple_cnn` balanceado) |

---

## Resumen de resultados

**Tipos de modelo:**
- **simple_cnn** — 4 CNNs independientes por segmento + modelo de fusión
- **backbone** — 4 DenseNet121 entrenados desde cero + fusión escalar
- **backbone_vectors** — 4 DenseNet121 entrenados desde cero + fusión con vectores 256-dim
- **unified_cnn** — una sola CNN que recibe los 4 segmentos simultáneamente y predice directamente la edad ósea, sin etapa de fusión separada

| Exp | Tipo | Dataset | CV MAE | Fusión MAE | Val MAE | Mex MAE | Estado |
|-----|------|---------|--------|-----------|---------|---------|--------|
| 33 | `simple_cnn` | recortado (24–216 m) | 26.4 m | 34.5 m | 39.2 m | 35.9 m | ✅ |
| 36 | `simple_cnn` | completo (1–228 m) | 34.3 m | 25.5 m | 30.2 m | 22.2 m | ✅ |
| 39 | `simple_cnn` | balanceado (≥50/edad) | 29.6 m | 41.8 m | 43.5 m | 35.9 m | ✅ |
| **42** | **`simple_cnn`** | **completo · extractor libre** | **33.1 m** | **20.1 m** | **24.1 m** | **20.0 m** | ✅ |
| **34** | **`backbone`** | **recortado (24–216 m)** | **27.6 m** | **9.2 m** | **14.6 m** | **17.6 m** | ✅ |
| **37** | **`backbone`** | **completo (1–228 m)** | **29.7 m** | **6.8 m** | **15.4 m** | **16.7 m** | ✅ |
| **40** | **`backbone`** | **balanceado (≥50/edad)** | **26.2 m** | **9.0 m** | **15.1 m** | **13.9 m** | ✅ |
| 35 | `backbone_vectors` | recortado (24–216 m) | 26.6 m | 27.6 m | 36.7 m | 23.4 m | ✅ |
| 38 | `backbone_vectors` | completo (1–228 m) | 31.4 m | 30.3 m | 40.0 m | 28.0 m | ✅ |
| 41 | `backbone_vectors` | balanceado (≥50/edad) | 26.7 m | 26.9 m | 35.0 m | 27.3 m | ✅ |
| **43** | **`backbone_vectors`** | **completo · extractor libre** | **31.5 m** | **17.3 m** | **23.0 m** | **18.5 m** | ✅ |
| **44** | **`unified_cnn`** | **recortado (24–216 m)** | **19.5 m** | N/A | **19.0 m** | **16.9 m** | ✅ |
| 45 | `unified_cnn` | completo (1–228 m) | 30.0 m | N/A | 29.0 m | 21.0 m | ✅ |
| **46** | **`unified_cnn`** | **balanceado (≥50/edad)** | **23.2 m** | N/A | **21.0 m** | **21.9 m** | ✅ |

> CV MAE: validación cruzada (5 folds) · Fusión MAE: mejor val_mae del integrador final · Val MAE: dataset RSNA (1,393 imgs) · Mex MAE: dataset mexicano (98 imgs) · **Negrita** = mejor por arquitectura
> ✅ completado

---

### Script 07 — Validación estándar (1,393 imágenes · 32 fallidas)

| Exp | Tipo | Dataset | Val MAE |
|-----|------|---------|---------|
| 33 | `simple_cnn` | recortado (24–216 m) | 39.2 m |
| 36 | `simple_cnn` | completo (1–228 m) | 30.2 m |
| 39 | `simple_cnn` | balanceado | 43.5 m |
| **42** | **`simple_cnn`** | **completo · extractor libre** | **24.1 m** |
| 34 | `backbone` | recortado (24–216 m) | **14.6 m** |
| 37 | `backbone` | completo (1–228 m) | **15.4 m** |
| 40 | `backbone` | balanceado | **15.1 m** |
| 35 | `bbone_vec` | recortado (24–216 m) | 36.7 m |
| 38 | `bbone_vec` | completo (1–228 m) | 40.0 m |
| 41 | `bbone_vec` | balanceado | 35.0 m |
| **43** | **`bbone_vec`** | **completo · extractor libre** | **23.0 m** |
| 44 | `unified_cnn` | recortado (24–216 m) | **19.0 m** |
| 45 | `unified_cnn` | completo (1–228 m) | 29.0 m |
| 46 | `unified_cnn` | balanceado | **21.0 m** |

---

### Script 08 — Validación mexicana (98 imágenes · 2 fallidas)

| Exp | Tipo | Dataset | Mex MAE |
|-----|------|---------|---------|
| 33 | `simple_cnn` | recortado (24–216 m) | 35.9 m |
| 36 | `simple_cnn` | completo (1–228 m) | 22.2 m |
| 39 | `simple_cnn` | balanceado | 35.9 m |
| **42** | **`simple_cnn`** | **completo · extractor libre** | **24.1 m** |
| 34 | `backbone` | recortado (24–216 m) | 17.6 m |
| 37 | `backbone` | completo (1–228 m) | 16.7 m |
| 40 | `backbone` | balanceado | **13.9 m** |
| 35 | `bbone_vec` | recortado (24–216 m) | 23.4 m |
| 38 | `bbone_vec` | completo (1–228 m) | 28.0 m |
| 41 | `bbone_vec` | balanceado | 27.3 m |
| **43** | **`bbone_vec`** | **completo · extractor libre** | **23.0 m** |
| 44 | `unified_cnn` | recortado (24–216 m) | **16.9 m** |
| 45 | `unified_cnn` | completo (1–228 m) | 21.0 m |
| 46 | `unified_cnn` | balanceado | 21.9 m |

---

### Script 09 — Análisis de desempeño (MAE Val del modelo de fusión + segmentos individuales)

| Exp | Tipo | Dataset | Fusión Val MAE | Pinky | Middle | Thumb | Wrist |
|-----|------|---------|---------------|-------|--------|-------|-------|
| 33 | `simple_cnn` | recortado (24–216 m) | 34.5 m | 33.0 m | 21.4 m | 18.9 m | 21.0 m |
| 36 | `simple_cnn` | completo (1–228 m) | 25.5 m | 37.7 m | 35.0 m | 19.7 m | 26.0 m |
| 39 | `simple_cnn` | balanceado | 41.8 m | 31.1 m | 21.8 m | 30.3 m | 21.4 m |
| **42** | **`simple_cnn`** | **completo · extractor libre** | **20.1 m** | 35.6 m | 27.0 m | 19.6 m | 21.0 m |
| 34 | `backbone` | recortado (24–216 m) | **9.2 m** | 24.3 m | 15.9 m | 19.7 m | 21.2 m |
| 37 | `backbone` | completo (1–228 m) | **6.8 m** | 20.5 m | 20.5 m | 54.9 m | 22.6 m |
| 40 | `backbone` | balanceado | **9.0 m** | 25.1 m | 17.7 m | 18.3 m | 29.5 m |
| 35 | `bbone_vec` | recortado (24–216 m) | 27.5 m | 15.4 m | 15.7 m | 19.6 m | 23.2 m |
| 38 | `bbone_vec` | completo (1–228 m) | 30.3 m | 15.2 m | 34.9 m | 14.3 m | 22.9 m |
| 41 | `bbone_vec` | balanceado | 26.9 m | 19.4 m | 18.2 m | 17.2 m | 19.9 m |
| **43** | **`bbone_vec`** | **completo · extractor libre** | **17.3 m** | 27.4 m | 21.1 m | 19.8 m | 24.7 m |
| 44 | `unified_cnn` | recortado (24–216 m) | **16.0 m** | N/A | N/A | N/A | N/A |
| 45 | `unified_cnn` | completo (1–228 m) | 25.6 m | N/A | N/A | N/A | N/A |
| 46 | `unified_cnn` | balanceado | **16.2 m** | N/A | N/A | N/A | N/A |

> N/A = no aplica para esta arquitectura · ⏳ = pendiente · ⚠️ = error (retry pendiente)

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
| Apr 22 | *(actual)* | Nueva arquitectura `unified_cnn` (end-to-end, 4 ramas separadas) + Exps 44–46 |
| Apr 23 | *(actual)* | Fix bug `Graph disconnected` en fusión `backbone_vectors` (afecta exps 35/38/41/43) |
