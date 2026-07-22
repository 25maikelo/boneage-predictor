# Tabla Maestra de Experimentos — Boneage Predictor

> Última actualización: 2026-07-22  
> Reemplaza `2026-05-28_experiments_summary.md` (desactualizado desde exp 47 en adelante)

---

## Leyenda

### Estado del experimento

| Símbolo | Significado |
|---------|-------------|
| ✅ | Completo — entrenado + validación RSNA + validación MEX |
| 🔶 | Validación incompleta — modelos sí, val solo PNGs sin `plot_data.json` (pre-script actualizado) |
| 🟡 | Sin validar — modelos entrenados, validación nunca ejecutada |
| ⬜ | Sin entrenar — config lista, sin modelos |
| ❌ | Sin config — carpeta sin `config.py`, no ejecutable |
| ⚠️ | Resultado anómalo — completado pero MAE inválido o desbordado |

### ¿Puede correr con la config actual?

| Símbolo | Significado |
|---------|-------------|
| ✅ | Config completa — todos los campos requeridos presentes |
| ⚠️ | Config parcial — faltan `SEGMENT_MODE` y/o `DATASET_PATH` (usaría defaults del sistema, comportamiento no garantizado) |
| ❌ | No ejecutable — falta `MODEL_TYPE` u otros campos sin default, o sin `config.py` |

### Datasets

| Alias | Archivo | N | Edades |
|-------|---------|---|--------|
| **raw** | `boneage-training-dataset.csv` | 12,611 | 160 clases (1–228 m) |
| **recortado** | mismo raw filtrado por `AGE_RANGE=(24,216)` | ~12,499 | igual |
| **balanceado** | `balanced_dataset.csv` | 11,783 | 36 clases con ≥50 muestras |

---

## Experimentos 00–16 — Heredados sin estructura

No tienen `config.py`. Los modelos en 03–16 son artefactos de experimentos pre-restructuración. No es posible reproducirlos ni ejecutar la validación. Carpetas ignoradas en git.

| Exp | Modelos | Puede correr | Razón |
|-----|---------|:------------:|-------|
| 00–02 | No | ❌ | Sin `config.py`, solo PNGs y `.keras` sueltos |
| 03–16 | Sí (formato antiguo) | ❌ | Sin `config.py`; modelos en formato desconocido |

---

## Fase 1 — Exploración de backbones (17–22)

Primera etapa formal. Config existe pero en formato antiguo: faltan `MODEL_TYPE`, `SEGMENT_MODE`, `DATASET_PATH`, `FREEZE_EXTRACTORS`, `SEGMENTATION_MODEL`. Los modelos existen (mezcla de `.keras` ZIP y SavedModel), sin validación ejecutada.

Para poder re-correr hay que añadir los 5 campos al `config.py` de cada experimento.

| Exp | Estado | Backbone | Img | Dataset | Modelos | Puede correr | Campos faltantes |
|-----|--------|----------|-----|---------|---------|:------------:|-----------------|
| 17 | 🟡 | InceptionV3 | 299×299 | raw (94–168 m) | `.keras` (zip) | ❌ | MODEL_TYPE, SEGMENT_MODE, DATASET_PATH, FREEZE_EXTRACTORS, SEGMENTATION_MODEL |
| 18 | 🟡 | VGG16 | 224×224 | raw (94–168 m) | `.keras` (zip) | ❌ | ídem |
| 19 | 🟡 | ResNet50 | 112×112 | raw (94–168 m) | SavedModel | ❌ | ídem |
| 20 | 🟡 | ResNet50 | 112×112 | raw (24–216 m) | SavedModel | ❌ | ídem |
| 21 | 🟡 | ResNet50 | 112×112 | raw (24–216 m) | SavedModel | ❌ | ídem |
| 22 | 🟡 | ResNet50 | 112×112 | raw (100–126 m) | `.keras` (zip) | ❌ | ídem |

> Nota: los `.keras` de 17/18/22 son formato Keras 3.x (ZIP), incompatibles con el entorno `boneage_gpu` (Keras 2.10). Necesitarían reentrenarse.

---

## Ablación de Backbones con dataset balanceado (23–28)

Objetivo: comparar 4 backbones manteniendo todo constante (balanced_dataset, spatial, FREEZE=True, 112px).  
Documentación estadística detallada: [`ablacion_backbones/2026-07-22_ablacion_backbones.md`](ablacion_backbones/2026-07-22_ablacion_backbones.md)

| Exp | Estado | Backbone | Img | Seg. mode | Val RSNA | Val MEX | Puede correr |
|-----|--------|----------|-----|-----------|:--------:|:-------:|:------------:|
| 23 | ✅ | ResNet50 | 112×112 | spatial | 16.2 m | 17.4 m | ✅ |
| 24 | 🔶 | ResNet50 | 224×224 | spatial | — | — | ✅ |
| 25 | 🟡 | ResNet50 + warmup | 224×224 | spatial | — | — | ✅ |
| 26 | ✅ | VGG16 | 112×112 | spatial | 37.2 m | 28.9 m | ✅ |
| 27 | ✅ | DenseNet121 | 112×112 | spatial | 14.2 m | 17.9 m | ✅ |
| 28 | ✅ | InceptionV3 | 112×112 | spatial | **13.6 m** | **17.2 m** | ✅ |

> 24 — val corre con script antiguo (solo PNGs, sin `plot_data.json`): re-validar con `07_validation.py` actualizado para habilitar prueba pareada.  
> 25 — validación nunca ejecutada.  
> Conclusión: InceptionV3 ≈ DenseNet121 > ResNet50 >> VGG16. Ver doc de ablación.

---

## Fase 2 — CNN Simple vs Backbone, con/sin género (29–35)

Introduce `simple_cnn`. Experimentos 29–32 son exploraciones con bugs o configuración incompleta; 33–35 son las versiones corregidas y de referencia.

| Exp | Estado | Tipo | Género | Img | Dataset | Val RSNA | Val MEX | Puede correr | Notas |
|-----|--------|------|:------:|-----|---------|:--------:|:-------:|:------------:|-------|
| 29 | ⬜ | `simple_cnn` | Sí | 112×112 | — | — | — | ⚠️ | Sin modelos — entrenamiento nunca corrió; falta DATASET_PATH |
| 30 | ⚠️ | `simple_cnn` | No | 112×112 | raw (24–216 m) | ~62K m | ~79K m | ✅ | MAE desbordado — bug por ausencia de género (segmentos divergen) |
| 31 | 🟡 | `simple_cnn` | Sí | 112×112 | raw (24–216 m) | — | — | ⚠️ | Modelos CV, sin val; falta DATASET_PATH |
| 32 | 🟡 | `backbone` (DenseNet121) | Sí | 112×112 | raw (24–216 m) | — | — | ⚠️ | Modelos CV, sin val; falta DATASET_PATH |
| 33 | ✅ | `simple_cnn` | Sí | 112×112 | recortado (24–216 m) | 39.2 m | 35.9 m | ✅ | Fix bug fusión género; referencia simple_cnn |
| 34 | ✅ | `backbone` (DenseNet121) | Sí | 112×112 | recortado (24–216 m) | 14.6 m | 17.6 m | ✅ | Mejor backbone escalar |
| 35 | ✅ | `backbone_vectors` (DenseNet121) | Sí | 112×112 | recortado (24–216 m) | 36.6 m | 23.4 m | ✅ | Primera versión backbone_vectors |

---

## Fase 3 — Dataset completo sin filtro de edad (36–38)

Réplicas de 33/34/35 con `AGE_RANGE=(1,228)` — 12,611 imágenes, 160 clases.

| Exp | Estado | Tipo | Img | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|--------|------|-----|---------|:--------:|:-------:|:------------:|
| 36 | ✅ | `simple_cnn` | 112×112 | completo (1–228 m) | 30.2 m | 22.2 m | ✅ |
| 37 | ✅ | `backbone` (DenseNet121) | 112×112 | completo (1–228 m) | 15.3 m | 16.7 m | ✅ |
| 38 | ✅ | `backbone_vectors` (DenseNet121) | 112×112 | completo (1–228 m) | 40.0 m | 28.0 m | ✅ |

---

## Fase 4 — Dataset balanceado (39–41)

Réplicas de 33/34/35 con `balanced_dataset.csv` — 11,783 imágenes, 36 clases.

| Exp | Estado | Tipo | Img | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|--------|------|-----|---------|:--------:|:-------:|:------------:|
| 39 | ✅ | `simple_cnn` | 112×112 | balanceado | 43.5 m | 35.9 m | ✅ |
| 40 | ✅ | `backbone` (DenseNet121) | 112×112 | balanceado | 15.1 m | 13.9 m | ✅ |
| 41 | ✅ | `backbone_vectors` (DenseNet121) | 112×112 | balanceado | 35.0 m | 27.3 m | ✅ |

---

## Fase 5 — Extractores descongelados (42–43)

Réplicas de 36/38 con `FREEZE_EXTRACTORS=False`. Solo aplica a `simple_cnn` y `backbone_vectors` (en `backbone` escalar no hay extractor que descongelar).

| Exp | Estado | Tipo | FREEZE | Img | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|--------|------|:------:|-----|---------|:--------:|:-------:|:------------:|
| 42 | ✅ | `simple_cnn` | False | 112×112 | completo (1–228 m) | 24.1 m | 20.0 m | ✅ |
| 43 | ✅ | `backbone_vectors` (DenseNet121) | False | 112×112 | completo (1–228 m) | 23.0 m | 18.5 m | ✅ |

---

## Fase 6 — CNN Unificada end-to-end (44–46)

Nueva arquitectura `unified_cnn`: 4 ramas CNN entrenadas directamente sobre bone age, sin pipeline de dos fases.

| Exp | Estado | Tipo | Img | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|--------|------|-----|---------|:--------:|:-------:|:------------:|
| 44 | ✅ | `unified_cnn` | 112×112 | recortado (24–216 m) | 19.0 m | 16.9 m | ✅ |
| 45 | ✅ | `unified_cnn` | 112×112 | completo (1–228 m) | 29.0 m | 21.0 m | ✅ |
| 46 | ✅ | `unified_cnn` | 112×112 | balanceado | 21.0 m | 21.9 m | ✅ |

---

## Fase 7 — Ablación de hiperparámetros (47–53)

Base: Exp 37 (backbone) y Exp 43 (backbone_vectors). Dataset completo (1–228 m), 112px, spatial.  
Variables estudiadas: género, learning rate, épocas de fusión.

| Exp | Estado | Tipo | Género | LR | Fus. ep. | Warmup | Val RSNA | Val MEX | Puede correr | Pregunta |
|-----|--------|------|:------:|:--:|:--------:|:------:|:--------:|:-------:|:------------:|---------|
| 47 | ✅ | `backbone` | No | 1e-3 | 20 | No | 16.5 m | 16.9 m | ✅ | ¿Es el género indispensable en backbone? (base: Exp 37, 15.3 m) |
| 48 | ✅ | `backbone_vectors` | No | 1e-3 | 20 | Sí | 18.5 m | 16.2 m | ✅ | ¿Es el género indispensable en bbone_vec libre? (base: Exp 43, 23.0 m) |
| 49 | ✅ | `backbone` | Sí | **1e-4** | **10** | No | 17.4 m | 17.0 m | ✅ | ¿LR bajo + pocas épocas mejora backbone? |
| 50 | ✅ | `backbone` | Sí | 1e-3 | **30** | No | 16.8 m | 16.5 m | ✅ | ¿Más épocas de fusión mejoran backbone? |
| 51 | ✅ | `backbone_vectors` | Sí | **1e-4** | **10** | Sí | 19.0 m | 16.1 m | ✅ | ¿LR bajo estabiliza bbone_vec libre? |
| 52 | ✅ | `backbone_vectors` | Sí | 1e-3 | **30** | Sí | 26.2 m | 20.0 m | ✅ | ¿Más épocas mejoran bbone_vec libre? |
| 53 | ✅ | `backbone_vectors` | **No** | **1e-4** | **10** | Sí | 18.3 m | 17.1 m | ✅ | ¿Mejora aditiva de quitar género + LR bajo? |

> Exp 54 no existe (número reservado, nunca creado).  
> Conclusiones de fase 7: quitar género no ayuda en backbone (16.5 vs 15.3); LR y épocas de fusión tienen impacto marginal; backbone_vectors sigue siendo ~3 m peor que backbone en RSNA.

---

## Fase 8 — Imágenes grandes 224×224 (55–58)

Base: Exp 37/43 pero escalando a 224×224. Se comparan modos de segmentación: `spatial` (conserva posición) vs `cropped` (recorta al bounding box). `BATCH_SIZE=8` por limitación de VRAM con 4 DenseNet121 en 224px.

| Exp | Estado | Tipo | FREEZE | Img | Seg. mode | Val RSNA | Val MEX | Puede correr | Vs. referencia |
|-----|--------|------|:------:|-----|-----------|:--------:|:-------:|:------------:|---------------|
| 55 | ✅ | `backbone` | True | 224×224 | spatial | 13.4 m | 17.4 m | ✅ | vs Exp 37 (15.3 m / 16.7 m): +1.9 m RSNA |
| 56 | ✅ | `backbone_vectors` | False | 224×224 | spatial | 15.5 m | 18.3 m | ✅ | vs Exp 43 (23.0 m / 18.5 m): +7.5 m RSNA |
| 57 | ✅ | `backbone` | True | 224×224 | **cropped** | **10.7 m** | 17.0 m | ✅ | vs Exp 55 (13.4 m): **−2.7 m** — mejor RSNA global |
| 58 | ✅ | `backbone_vectors` | False | 224×224 | **cropped** | 16.8 m | **15.0 m** | ✅ | vs Exp 56 (15.5 m): +1.3 m RSNA; mejor MEX |

> 55/56: `SEGMENT_MODE` no declarado en config → usa default `spatial` del sistema (campo añadido implícitamente por paths.py). Para garantizar reproducibilidad se recomienda añadir el campo explícito.  
> Exp 57 es el mejor modelo global en RSNA (10.7 m). Exp 58 tiene el mejor MEX (15.0 m).

---

## Especiales (98, 99)

| Exp | Estado | Tipo | Img | Propósito | Val RSNA | Val MEX | Puede correr |
|-----|--------|------|-----|-----------|:--------:|:-------:|:------------:|
| 98 | 🟡 | `backbone_vectors` | 112×112 | Quick test mínimo de `create_fusion_model_backbone_vectors` | — | — | ✅ |
| 99 | ✅ | `simple_cnn` | 64×64 | Quick test pipeline completo (80 muestras, K-Fold=2) | ~30 m* | ~92 m* | ✅ |

> *MAE alto esperado — datos insuficientes por diseño.  
> 98: falta `SEGMENT_MODE` y `DATASET_PATH`.

---

## Resumen global de experimentos completos

Solo incluye experimentos con validación RSNA + MEX válidas (MAE < 500 m).

| Exp | Tipo | Backbone | Img | Seg. | Dataset | FREEZE | Género | Val RSNA | Val MEX |
|-----|------|----------|-----|------|---------|:------:|:------:|:--------:|:-------:|
| 23 | backbone | ResNet50 | 112 | spatial | balanceado | ✅ | Sí | 16.2 m | 17.4 m |
| 26 | backbone | VGG16 | 112 | spatial | balanceado | ✅ | Sí | 37.2 m | 28.9 m |
| 27 | backbone | DenseNet121 | 112 | spatial | balanceado | ✅ | Sí | 14.2 m | 17.9 m |
| 28 | backbone | InceptionV3 | 112 | spatial | balanceado | ✅ | Sí | 13.6 m | 17.2 m |
| 33 | simple_cnn | — | 112 | spatial | recortado | ✅ | Sí | 39.2 m | 35.9 m |
| 34 | backbone | DenseNet121 | 112 | spatial | recortado | ✅ | Sí | 14.6 m | 17.6 m |
| 35 | backbone_vectors | DenseNet121 | 112 | spatial | recortado | ✅ | Sí | 36.6 m | 23.4 m |
| 36 | simple_cnn | — | 112 | spatial | completo | ✅ | Sí | 30.2 m | 22.2 m |
| 37 | backbone | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 15.3 m | 16.7 m |
| 38 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 40.0 m | 28.0 m |
| 39 | simple_cnn | — | 112 | spatial | balanceado | ✅ | Sí | 43.5 m | 35.9 m |
| 40 | backbone | DenseNet121 | 112 | spatial | balanceado | ✅ | Sí | 15.1 m | 13.9 m |
| 41 | backbone_vectors | DenseNet121 | 112 | spatial | balanceado | ✅ | Sí | 35.0 m | 27.3 m |
| 42 | simple_cnn | — | 112 | spatial | completo | ❌ | Sí | 24.1 m | 20.0 m |
| 43 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | Sí | 23.0 m | 18.5 m |
| 44 | unified_cnn | — | 112 | spatial | recortado | ✅ | Sí | 19.0 m | 16.9 m |
| 45 | unified_cnn | — | 112 | spatial | completo | ✅ | Sí | 29.0 m | 21.0 m |
| 46 | unified_cnn | — | 112 | spatial | balanceado | ✅ | Sí | 21.0 m | 21.9 m |
| 47 | backbone | DenseNet121 | 112 | spatial | completo | ✅ | **No** | 16.5 m | 16.9 m |
| 48 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | **No** | 18.5 m | 16.2 m |
| 49 | backbone | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 17.4 m | 17.0 m |
| 50 | backbone | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 16.8 m | 16.5 m |
| 51 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | Sí | 19.0 m | 16.1 m |
| 52 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | Sí | 26.2 m | 20.0 m |
| 53 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | **No** | 18.3 m | 17.1 m |
| 55 | backbone | DenseNet121 | **224** | spatial | completo | ✅ | Sí | 13.4 m | 17.4 m |
| 56 | backbone_vectors | DenseNet121 | **224** | spatial | completo | ❌ | **No** | 15.5 m | 18.3 m |
| **57** | **backbone** | **DenseNet121** | **224** | **cropped** | **completo** | **✅** | **Sí** | **10.7 m** ★ | 17.0 m |
| **58** | **backbone_vectors** | **DenseNet121** | **224** | **cropped** | **completo** | **❌** | **Sí** | 16.8 m | **15.0 m** ★ |

> ★ Mejor valor en cada columna.  
> **Negrita** = experimentos con los mejores resultados absolutos.

---

## Experimentos pendientes o con deuda técnica

| Exp | Deuda | Acción recomendada |
|-----|-------|-------------------|
| 17–22 | 5 campos faltantes en config.py + modelos Keras 3.x incompatibles | Añadir campos + reentrenar si se necesitan resultados formales |
| 24 | Val solo PNGs, sin `plot_data.json` | Re-correr `07_validation.py` actualizado |
| 25 | Sin validación | Correr `07_validation.py` + `08_mex_validation.py` |
| 29 | Sin modelos; falta DATASET_PATH | Añadir DATASET_PATH → reentrenar |
| 31–32 | Modelos CV, sin val; falta DATASET_PATH | Añadir DATASET_PATH → correr validación |
| 30 | MAE desbordado (~62K m) | Investigar causa o descartar (bug sin género en simple_cnn) |
