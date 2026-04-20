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

## Fase 2 — CNN Simple vs Backbone, con/sin género (Exps 29–34)

Se introduce `simple_cnn` (4 bloques conv + Flatten + Dense). El foco es comparar arquitecturas y el impacto del género. Esta fase tuvo **dos iteraciones de código** motivadas por bugs en la fusión con género.

| Exp | Estado | Tipo modelo | Género | Img | Épocas (seg/fus/ft) | MAE CV prom. | Notas relevantes |
|-----|--------|-------------|--------|-----|----------------------|--------------|-----------------|
| 29 | Completado | `simple_cnn` | Sí | 112×112 | 15/20/10 | — | **Baseline CNN**; réplica directa de Exp 26 con nueva arquitectura; múltiples runs de ajuste |
| 30 | Completado | `simple_cnn` | No | 112×112 | 15/20/10 | Pinky/Thumb/Wrist: ~123 m · Middle: ~36 m | Sin género; MAE anómalo en 3 segmentos sugiere artefacto de segmentación |
| 31 | Incompleto | `simple_cnn` | Sí | 112×112 | 15/20/10 | Pinky: 36.6 · Middle: 36.4 · Thumb: 36.6 · Wrist: 36.7 | Género restaura consistencia entre segmentos; entrenamiento no terminó (bug en fusión detectado después) |
| 32 | Completado | `backbone` (DenseNet121) | Sí | 112×112 | 15/20/10 | Pinky: 37.1 · Middle: 37.6 · Thumb: 37.0 · Wrist: 37.1 | Réplica de Exp 27 con infraestructura actualizada (CV + rutas segmentación); referencia backbone para comparar con CNN |
| **33** | **Solo config** *(ajuste de código)* | `simple_cnn` | Sí | 112×112 | 15/20/10 | — | **Réplica corregida de Exp 31** — fix bug en fusión con `USE_GENDER`; pendiente de ejecución |
| **34** | **Solo config** *(ajuste de código)* | `backbone` (DenseNet121) | Sí | 112×112 | 15/20/10 | — | **Réplica corregida de Exp 32** — mismo fix; comparación backbone vs CNN con código limpio |

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
