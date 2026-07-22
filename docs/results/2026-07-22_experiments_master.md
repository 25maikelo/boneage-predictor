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
| ❌ | Sin config — carpeta sin `config.py`, no ejecutable |
| ⚠️ | Resultado anómalo — completado pero MAE inválido o desbordado |

### ¿Puede correr con la config actual?

| Símbolo | Significado |
|---------|-------------|
| ✅ | Config completa — todos los campos requeridos presentes |
| ❌ | No ejecutable — faltan campos críticos o sin `config.py` |

### Datasets

| Alias | Archivo | N | AGE_RANGE |
|-------|---------|---|-----------|
| **recortado** | `boneage-training-dataset.csv` | ~12,499 | (24, 216) m |
| **completo** | `boneage-training-dataset.csv` | 12,611 | (1, 228) m |
| **balanceado** | `balanced_dataset.csv` | 11,783 | (24, 216) m — solo clases con ≥50 muestras |

> Los tres usan el mismo CSV raíz (`boneage-training-dataset.csv`); la diferencia es el `AGE_RANGE` en config o el CSV preprocesado. Los experimentos 17–22 usaron rangos no estándar (ver tabla fase 1).

---

## Árbol de linaje

Muestra de qué experimento deriva cada uno y qué cambió. `★` = mejor resultado global.

```
── Fase 1: exploración de backbones ─────────────────────────────────────────
17 (InceptionV3, raw 94-168m, 299px)
18 (VGG16, raw 94-168m, 224px)
19 (ResNet50, raw 94-168m, 112px)
└── 20 (→ rango 24-216m)
    ├── 21 (− fine-tuning)
    └── 22 (→ rango estrecho 100-126m)

── Ablación de backbones ────────────────────────────────────────────────────
23 (ResNet50 | balanced | spatial | 112px)  ← BASE ablación
├── 24 (→ 224px)
│   └── 25 (+ warmup)
├── 26 (→ VGG16)
├── 27 (→ DenseNet121)
└── 28 (→ InceptionV3)

── Fase 2: simple_cnn vs backbone ──────────────────────────────────────────
29 (simple_cnn | recortado | con género)
└── 30 (− género)
    └── 31 (+ género de vuelta, intento fix)
        └── 33 [fix bug fusión]  ← BASE simple_cnn
            ├── 36 (→ completo)
            │   └── 42 (+ FREEZE=False)
            └── 39 (→ balanceado)

27 → 32 [+ CV + rutas actualizadas]
    └── 34 [fix bug fusión]  ← BASE backbone escalar
        ├── 37 (→ completo)  ← BASE backbone completo
        │   ├── 47 (− género)
        │   ├── 49 (→ LR=1e-4, fusión 10ep)
        │   ├── 50 (→ fusión 30ep)
        │   └── 55 (→ 224px)
        │       └── 57 (→ cropped)  ★ Mejor RSNA (10.7m)
        └── 40 (→ balanceado)

35 (backbone_vectors | recortado)  ← BASE backbone_vectors
├── 38 (→ completo)
│   └── 43 (+ FREEZE=False)  ← BASE bbone_vec libre
│       ├── 48 (− género)
│       │   └── 56 (→ 224px)
│       │       └── 58 (→ cropped)  ★ Mejor MEX (15.0m)
│       ├── 51 (→ LR=1e-4, fusión 10ep)
│       ├── 52 (→ fusión 30ep)
│       └── 53 (− género + LR=1e-4, fusión 10ep)
└── 41 (→ balanceado)

── Fase 6: unified_cnn ─────────────────────────────────────────────────────
44 (unified_cnn | recortado)  ← BASE unified_cnn
├── 45 (→ completo)
└── 46 (→ balanceado)
```

---

## Experimentos 00–16 — Heredados sin estructura

No tienen `config.py`. Los modelos en 03–16 son artefactos de experimentos pre-restructuración. Carpetas ignoradas en git.

| Exp | Réplica de | Modelos | Puede correr | Razón |
|-----|-----------|---------|:------------:|-------|
| 00–02 | — | No | ❌ | Sin `config.py`, solo PNGs y `.keras` sueltos |
| 03–16 | — | Sí (formato antiguo) | ❌ | Sin `config.py`; modelos en formato desconocido |

---

## Fase 1 — Exploración de backbones (17–22)

Config en formato antiguo: faltan `MODEL_TYPE`, `SEGMENT_MODE`, `DATASET_PATH`, `FREEZE_EXTRACTORS`, `SEGMENTATION_MODEL`. Modelos existen pero sin validación.

| Exp | Réplica de | Estado | Backbone | Img | AGE_RANGE | Modelos | Puede correr |
|-----|-----------|--------|----------|-----|-----------|---------|:------------:|
| 17 | Nuevo | 🟡 | InceptionV3 | 299×299 | (94, 168) m | `.keras` (zip) | ❌ |
| 18 | Nuevo | 🟡 | VGG16 | 224×224 | (94, 168) m | `.keras` (zip) | ❌ |
| 19 | Nuevo | 🟡 | ResNet50 | 112×112 | (94, 168) m | SavedModel | ❌ |
| 20 | 19 → rango completo | 🟡 | ResNet50 | 112×112 | (24, 216) m | SavedModel | ❌ |
| 21 | 20 − fine-tuning | 🟡 | ResNet50 | 112×112 | (24, 216) m | SavedModel | ❌ |
| 22 | 20 → rango estrecho | 🟡 | ResNet50 | 112×112 | (100, 126) m | `.keras` (zip) | ❌ |

> Los 5 campos faltantes deben añadirse a cada `config.py` para poder re-correr.  
> Los `.keras` de 17/18/22 son formato Keras 3.x (ZIP), incompatibles con `boneage_gpu` (Keras 2.10) — necesitarían reentrenarse.

---

## Ablación de Backbones con dataset balanceado (23–28)

Objetivo: comparar 4 backbones manteniendo todo constante (balanced_dataset, spatial, FREEZE=True, 112px).  
Documentación estadística detallada: [`ablacion_backbones/2026-07-22_ablacion_backbones.md`](ablacion_backbones/2026-07-22_ablacion_backbones.md)

| Exp | Réplica de | Estado | Backbone | Img | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|----------|-----|:--------:|:-------:|:------------:|
| 23 | Nuevo (BASE ablación) | ✅ | ResNet50 | 112×112 | 16.2 m | 17.4 m | ✅ |
| 24 | 23 → 224px | 🔶 | ResNet50 | 224×224 | — | — | ✅ |
| 25 | 24 + warmup | 🟡 | ResNet50 | 224×224 | — | — | ✅ |
| 26 | 23 → VGG16 | ✅ | VGG16 | 112×112 | 37.2 m | 28.9 m | ✅ |
| 27 | 23 → DenseNet121 | ✅ | DenseNet121 | 112×112 | 14.2 m | 17.9 m | ✅ |
| 28 | 23 → InceptionV3 | ✅ | InceptionV3 | 112×112 | **13.6 m** | **17.2 m** | ✅ |

> 24 — val corrió con script antiguo (solo PNGs, sin `plot_data.json`): re-validar para habilitar prueba pareada.  
> 25 — validación nunca ejecutada.  
> Conclusión: InceptionV3 ≈ DenseNet121 > ResNet50 >> VGG16. Ver doc de ablación.

---

## Fase 2 — CNN Simple vs Backbone, con/sin género (29–35)

Introduce `simple_cnn`. Experimentos 29–32 son exploraciones con bugs o infraestructura desactualizada; 33–35 son las versiones de referencia.

| Exp | Réplica de | Estado | Tipo | Género | Img | Dataset | Val RSNA | Val MEX | Puede correr | Notas |
|-----|-----------|--------|------|:------:|-----|---------|:--------:|:-------:|:------------:|-------|
| 29 | Nuevo (BASE simple_cnn) | 🔶 | `simple_cnn` | Sí | 112×112 | recortado | ~48 m† | ~31.6 m† | ✅ | Completado en Windows; modelos no migrados al clúster |
| 30 | 29 − género | ⚠️ | `simple_cnn` | No | 112×112 | recortado | ~62K m | ~79K m | ✅ | MAE desbordado — bug por ausencia de género |
| 31 | 30 + género (fix parcial) | 🟡 | `simple_cnn` | Sí | 112×112 | recortado | — | — | ✅ | CV K=5; val nunca ejecutada |
| 32 | 27 + CV + rutas actualizadas | 🟡 | `backbone` (DenseNet121) | Sí | 112×112 | recortado | — | — | ✅ | CV K=5; val nunca ejecutada |
| 33 | 31 [fix bug fusión género] | ✅ | `simple_cnn` | Sí | 112×112 | recortado | 39.2 m | 35.9 m | ✅ | Referencia `simple_cnn` |
| 34 | 32 [fix bug fusión género] | ✅ | `backbone` (DenseNet121) | Sí | 112×112 | recortado | 14.6 m | 17.6 m | ✅ | Referencia `backbone` escalar |
| 35 | Nuevo (BASE backbone_vectors) | ✅ | `backbone_vectors` (DenseNet121) | Sí | 112×112 | recortado | 36.6 m | 23.4 m | ✅ | Primera versión backbone_vectors |

> †Valores de los logs de Windows (script de validación antiguo, sin `plot_data.json`).

---

## Fase 3 — Dataset completo sin filtro de edad (36–38)

Réplicas de 33/34/35 con `AGE_RANGE=(1,228)`.

| Exp | Réplica de | Estado | Tipo | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|------|---------|:--------:|:-------:|:------------:|
| 36 | 33 → completo | ✅ | `simple_cnn` | completo (1–228 m) | 30.2 m | 22.2 m | ✅ |
| 37 | 34 → completo | ✅ | `backbone` (DenseNet121) | completo (1–228 m) | 15.3 m | 16.7 m | ✅ |
| 38 | 35 → completo | ✅ | `backbone_vectors` (DenseNet121) | completo (1–228 m) | 40.0 m | 28.0 m | ✅ |

---

## Fase 4 — Dataset balanceado (39–41)

Réplicas de 33/34/35 con `balanced_dataset.csv`.

| Exp | Réplica de | Estado | Tipo | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|------|---------|:--------:|:-------:|:------------:|
| 39 | 33 → balanceado | ✅ | `simple_cnn` | balanceado | 43.5 m | 35.9 m | ✅ |
| 40 | 34 → balanceado | ✅ | `backbone` (DenseNet121) | balanceado | 15.1 m | 13.9 m | ✅ |
| 41 | 35 → balanceado | ✅ | `backbone_vectors` (DenseNet121) | balanceado | 35.0 m | 27.3 m | ✅ |

---

## Fase 5 — Extractores descongelados (42–43)

Réplicas de 36/38 con `FREEZE_EXTRACTORS=False`. Solo aplica a `simple_cnn` y `backbone_vectors` (en `backbone` escalar no hay extractor que descongelar).

| Exp | Réplica de | Estado | Tipo | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|------|---------|:--------:|:-------:|:------------:|
| 42 | 36 + FREEZE=False | ✅ | `simple_cnn` | completo (1–228 m) | 24.1 m | 20.0 m | ✅ |
| 43 | 38 + FREEZE=False | ✅ | `backbone_vectors` (DenseNet121) | completo (1–228 m) | 23.0 m | 18.5 m | ✅ |

---

## Fase 6 — CNN Unificada end-to-end (44–46)

Nueva arquitectura `unified_cnn`: 4 ramas CNN entrenadas directamente sobre bone age, sin pipeline de dos fases.

| Exp | Réplica de | Estado | Tipo | Dataset | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|------|---------|:--------:|:-------:|:------------:|
| 44 | Nuevo (BASE unified_cnn) | ✅ | `unified_cnn` | recortado (24–216 m) | 19.0 m | 16.9 m | ✅ |
| 45 | 44 → completo | ✅ | `unified_cnn` | completo (1–228 m) | 29.0 m | 21.0 m | ✅ |
| 46 | 44 → balanceado | ✅ | `unified_cnn` | balanceado | 21.0 m | 21.9 m | ✅ |

---

## Fase 7 — Ablación de hiperparámetros (47–53)

Todas parten de Exp 37 (`backbone`) o Exp 43 (`backbone_vectors`). Dataset completo (1–228 m), 112px, spatial.

| Exp | Réplica de | Estado | Tipo | Género | LR | Fus. ep. | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|------|:------:|:--:|:--------:|:--------:|:-------:|:------------:|
| 47 | 37 − género | ✅ | `backbone` | No | 1e-3 | 20 | 16.5 m | 16.9 m | ✅ |
| 48 | 43 − género | ✅ | `backbone_vectors` | No | 1e-3 | 20 | 18.5 m | 16.2 m | ✅ |
| 49 | 37 → LR=1e-4, fusión 10ep | ✅ | `backbone` | Sí | 1e-4 | 10 | 17.4 m | 17.0 m | ✅ |
| 50 | 37 → fusión 30ep | ✅ | `backbone` | Sí | 1e-3 | 30 | 16.8 m | 16.5 m | ✅ |
| 51 | 43 → LR=1e-4, fusión 10ep | ✅ | `backbone_vectors` | Sí | 1e-4 | 10 | 19.0 m | 16.1 m | ✅ |
| 52 | 43 → fusión 30ep | ✅ | `backbone_vectors` | Sí | 1e-3 | 30 | 26.2 m | 20.0 m | ✅ |
| 53 | 43 − género + LR=1e-4 + fusión 10ep | ✅ | `backbone_vectors` | No | 1e-4 | 10 | 18.3 m | 17.1 m | ✅ |

> Exp 54 no existe (número reservado, nunca creado).  
> Conclusiones: quitar género no mejora backbone (47: 16.5 vs 37: 15.3 m); LR y épocas de fusión tienen impacto marginal; backbone_vectors sigue siendo ~3 m peor que backbone en RSNA.

---

## Fase 8 — Imágenes grandes 224×224 (55–58)

`BATCH_SIZE=8` por limitación de VRAM con 4 DenseNet121 a 224px.

| Exp | Réplica de | Estado | Tipo | Seg. mode | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|------|-----------|:--------:|:-------:|:------------:|
| 55 | 37 → 224px | ✅ | `backbone` | spatial | 13.4 m | 17.4 m | ✅ |
| 56 | 48 → 224px | ✅ | `backbone_vectors` | spatial | 15.5 m | 18.3 m | ✅ |
| 57 | 55 → cropped | ✅ | `backbone` | **cropped** | **10.7 m** ★ | 17.0 m | ✅ |
| 58 | 56 → cropped | ✅ | `backbone_vectors` | **cropped** | 16.8 m | **15.0 m** ★ | ✅ |

> Exp 57 es el mejor modelo global en RSNA (10.7 m). Exp 58 tiene el mejor MEX (15.0 m).

---

## Especiales (98, 99)

| Exp | Réplica de | Estado | Tipo | Img | Propósito | Val RSNA | Val MEX | Puede correr |
|-----|-----------|--------|------|-----|-----------|:--------:|:-------:|:------------:|
| 98 | Nuevo (quick test) | 🟡 | `backbone_vectors` | 112×112 | Prueba mínima de `create_fusion_model_backbone_vectors` | — | — | ✅ |
| 99 | Nuevo (quick test) | ✅ | `simple_cnn` | 64×64 | Pipeline completo con datos mínimos (80 muestras, K-Fold=2) | ~30 m† | ~92 m† | ✅ |

> †MAE alto esperado — datos insuficientes por diseño.

---

## Resumen global de experimentos completos

Solo incluye experimentos con validación RSNA + MEX válidas (MAE < 500 m).

| Exp | Réplica de | Tipo | Backbone | Img | Seg. | Dataset | FREEZE | Género | Val RSNA | Val MEX |
|-----|-----------|------|----------|-----|------|---------|:------:|:------:|:--------:|:-------:|
| 23 | Nuevo | backbone | ResNet50 | 112 | spatial | balanceado | ✅ | Sí | 16.2 m | 17.4 m |
| 26 | 23 → VGG16 | backbone | VGG16 | 112 | spatial | balanceado | ✅ | Sí | 37.2 m | 28.9 m |
| 27 | 23 → DenseNet121 | backbone | DenseNet121 | 112 | spatial | balanceado | ✅ | Sí | 14.2 m | 17.9 m |
| 28 | 23 → InceptionV3 | backbone | InceptionV3 | 112 | spatial | balanceado | ✅ | Sí | 13.6 m | 17.2 m |
| 33 | 31 [fix] | simple_cnn | — | 112 | spatial | recortado | ✅ | Sí | 39.2 m | 35.9 m |
| 34 | 32 [fix] | backbone | DenseNet121 | 112 | spatial | recortado | ✅ | Sí | 14.6 m | 17.6 m |
| 35 | Nuevo | backbone_vectors | DenseNet121 | 112 | spatial | recortado | ✅ | Sí | 36.6 m | 23.4 m |
| 36 | 33 → completo | simple_cnn | — | 112 | spatial | completo | ✅ | Sí | 30.2 m | 22.2 m |
| 37 | 34 → completo | backbone | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 15.3 m | 16.7 m |
| 38 | 35 → completo | backbone_vectors | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 40.0 m | 28.0 m |
| 39 | 33 → balanceado | simple_cnn | — | 112 | spatial | balanceado | ✅ | Sí | 43.5 m | 35.9 m |
| 40 | 34 → balanceado | backbone | DenseNet121 | 112 | spatial | balanceado | ✅ | Sí | 15.1 m | 13.9 m |
| 41 | 35 → balanceado | backbone_vectors | DenseNet121 | 112 | spatial | balanceado | ✅ | Sí | 35.0 m | 27.3 m |
| 42 | 36 + FREEZE=F | simple_cnn | — | 112 | spatial | completo | ❌ | Sí | 24.1 m | 20.0 m |
| 43 | 38 + FREEZE=F | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | Sí | 23.0 m | 18.5 m |
| 44 | Nuevo | unified_cnn | — | 112 | spatial | recortado | ✅ | Sí | 19.0 m | 16.9 m |
| 45 | 44 → completo | unified_cnn | — | 112 | spatial | completo | ✅ | Sí | 29.0 m | 21.0 m |
| 46 | 44 → balanceado | unified_cnn | — | 112 | spatial | balanceado | ✅ | Sí | 21.0 m | 21.9 m |
| 47 | 37 − género | backbone | DenseNet121 | 112 | spatial | completo | ✅ | No | 16.5 m | 16.9 m |
| 48 | 43 − género | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | No | 18.5 m | 16.2 m |
| 49 | 37 → LR=1e-4 | backbone | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 17.4 m | 17.0 m |
| 50 | 37 → fus. 30ep | backbone | DenseNet121 | 112 | spatial | completo | ✅ | Sí | 16.8 m | 16.5 m |
| 51 | 43 → LR=1e-4 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | Sí | 19.0 m | 16.1 m |
| 52 | 43 → fus. 30ep | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | Sí | 26.2 m | 20.0 m |
| 53 | 43 − género + LR=1e-4 | backbone_vectors | DenseNet121 | 112 | spatial | completo | ❌ | No | 18.3 m | 17.1 m |
| 55 | 37 → 224px | backbone | DenseNet121 | **224** | spatial | completo | ✅ | Sí | 13.4 m | 17.4 m |
| 56 | 48 → 224px | backbone_vectors | DenseNet121 | **224** | spatial | completo | ❌ | No | 15.5 m | 18.3 m |
| **57** | **55 → cropped** | **backbone** | **DenseNet121** | **224** | **cropped** | **completo** | **✅** | **Sí** | **10.7 m** ★ | 17.0 m |
| **58** | **56 → cropped** | **backbone_vectors** | **DenseNet121** | **224** | **cropped** | **completo** | **❌** | **Sí** | 16.8 m | **15.0 m** ★ |

> ★ Mejor valor en cada columna.  
> **Negrita** = experimentos con los mejores resultados absolutos.

---

## Experimentos pendientes o con deuda técnica

| Exp | Deuda | Acción recomendada |
|-----|-------|-------------------|
| 17–22 | 5 campos faltantes en config.py + modelos Keras 3.x incompatibles | Añadir campos + reentrenar si se necesitan resultados formales |
| 24 | Val solo PNGs, sin `plot_data.json` | Re-correr `07_validation.py` actualizado |
| 25 | Sin validación | Correr `07_validation.py` + `08_mex_validation.py` |
| 29 | Modelos solo en Windows, val sin `plot_data.json` | Migrar modelos al clúster o reentrenar → re-validar con script actualizado |
| 31–32 | CV K=5; validación nunca ejecutada | Correr `07_validation.py` + `08_mex_validation.py` |
| 30 | MAE desbordado (~62K m) | Investigar causa o descartar (bug sin género en simple_cnn) |
