# Tabla Maestra de Experimentos — Boneage Predictor

> Última actualización: 2026-07-22  
> Reemplaza `2026-05-28_experiments_summary.md`

---

## Leyenda

| Campo | Valores |
|-------|---------|
| **Estado** | 🟢 Completo y reproducible · 🟡 Incompleto pero se podría completar · 🔴 Terminó por error · ⬛ Sin configuración |
| **Tipo** | `simple_cnn` · `backbone` · `bbone_vec` (backbone_vectors) · `unified` (unified_cnn) · N/A |
| **Seg.** | `spatial` (conserva posición) · `cropped` (recorte al bbox) |
| **CSV** | `raw` (boneage-training-dataset.csv) · `balanced` (balanced_dataset.csv) |
| **Edad** | rango `AGE_RANGE` en meses |
| **FREEZE** | ✅ extractores congelados en fusión · ❌ libres |
| **Género** | ✅ `USE_GENDER=True` · ❌ `USE_GENDER=False` |

> N/A = no aplica por diseño (p.ej. Backbone en simple_cnn). — = dato desconocido o faltante.

---

## Árbol de linaje

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
29 (simple_cnn | raw recortado | con género)
└── 30 (− género)
    └── 31 (+ género, intento fix)
        └── 33 [fix entrenamiento fusión]  ← BASE simple_cnn
            ├── 36 (→ completo)
            │   └── 42 (+ FREEZE=False)
            └── 39 (→ balanceado)

27 → 32 [+ CV + rutas actualizadas]
    └── 34 [fix entrenamiento fusión]  ← BASE backbone escalar
        ├── 37 (→ completo)  ← BASE backbone completo
        │   ├── 47 (− género)
        │   ├── 49 (→ LR=1e-4, fusión 10ep)
        │   ├── 50 (→ fusión 30ep)
        │   └── 55 (→ 224px)
        │       └── 57 (→ cropped)  ★ Mejor RSNA y MEX globales (10.7m / 13.5m)
        └── 40 (→ balanceado)

35 (backbone_vectors | raw recortado)  ← BASE backbone_vectors
├── 38 (→ completo)
│   └── 43 (+ FREEZE=False)  ← BASE bbone_vec libre
│       ├── 48 (− género)
│       │   └── 56 (→ 224px)
│       │       └── 58 (→ cropped)
│       ├── 51 (→ LR=1e-4, fusión 10ep)
│       ├── 52 (→ fusión 30ep)
│       └── 53 (− género + LR=1e-4 + fusión 10ep)
└── 41 (→ balanceado)

── Fase 6: unified_cnn ─────────────────────────────────────────────────────
44 (unified_cnn | raw recortado)  ← BASE unified_cnn
├── 45 (→ completo)
└── 46 (→ balanceado)

── Fase 9: baseline whole-hand ─────────────────────────────────────────────
27 (comparación directa, mismo split/protocolo)
└── 60 (whole_hand | sin segmentación)  ← aísla el efecto de segmentar en 4 regiones
```

---

## Tabla unificada — todos los experimentos

### Sin configuración (00–16)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 00–16 | ⬛ | — | — | — | — | — | — | — | — | — | — |

### Fase 1 — Exploración de backbones (17–28)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 17 | 🟡 | `backbone` | InceptionV3 | 299×299 | — | raw | 94–168 | — | — | — | — |
| 18 | 🟡 | `backbone` | VGG16 | 224×224 | — | raw | 94–168 | — | — | — | — |
| 19 | 🟡 | `backbone` | ResNet50 | 112×112 | — | raw | 94–168 | — | — | — | — |
| 20 | 🟡 | `backbone` | ResNet50 | 112×112 | — | raw | 24–216 | — | — | — | — |
| 21 | 🟡 | `backbone` | ResNet50 | 112×112 | — | raw | 24–216 | — | — | — | — |
| 22 | 🟡 | `backbone` | ResNet50 | 112×112 | — | raw | 100–126 | — | — | — | — |
| 23 | 🟢 | `backbone` | ResNet50 | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 16.2 m | 19.3 m |
| 24 | 🟡 | `backbone` | ResNet50 | 224×224 | spatial | balanced | 24–216 | ✅ | ✅ | — | — |
| 25 | 🟡 | `backbone` | ResNet50 | 224×224 | spatial | balanced | 24–216 | ✅ | ✅ | — | — |
| 26 | 🟢 | `backbone` | VGG16 | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 37.2 m | 40.0 m |
| 27 | 🟢 | `backbone` | DenseNet121 | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 14.2 m | 16.4 m |
| 28 | 🟢 | `backbone` | InceptionV3 | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 13.6 m | 17.3 m |

### Fase 2 — simple_cnn vs backbone (29–35)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 29 | 🟡 | `simple_cnn` | N/A | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | ~48 m† | ~31.6 m† |
| 30 | 🔴 | `simple_cnn` | N/A | 112×112 | spatial | raw | 24–216 | ✅ | ❌ | ~62K m | ~79K m |
| 31 | 🟡 | `simple_cnn` | N/A | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | — | — |
| 32 | 🟡 | `backbone` | DenseNet121 | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | — | — |
| 33 | 🟢 | `simple_cnn` | N/A | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | 39.2 m | 38.2 m |
| 34 | 🟢 | `backbone` | DenseNet121 | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | 14.6 m | 16.8 m |
| 35 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | 36.6 m | 27.5 m |

### Fase 3 — Dataset completo (36–38)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 36 | 🟢 | `simple_cnn` | N/A | 112×112 | spatial | raw | 1–228 | ✅ | ✅ | 30.2 m | 24.3 m |
| 37 | 🟢 | `backbone` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ✅ | ✅ | 15.3 m | 18.4 m |
| 38 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ✅ | ✅ | 40.0 m | 31.7 m |

### Fase 4 — Dataset balanceado (39–41)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 39 | 🟢 | `simple_cnn` | N/A | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 43.5 m | 37.9 m |
| 40 | 🟢 | `backbone` | DenseNet121 | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 15.1 m | 16.4 m |
| 41 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 35.0 m | 30.2 m |

### Fase 5 — Extractores descongelados (42–43)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 42 | 🟢 | `simple_cnn` | N/A | 112×112 | spatial | raw | 1–228 | ❌ | ✅ | 24.1 m | 25.4 m |
| 43 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ❌ | ✅ | 23.0 m | 21.0 m |

### Fase 6 — unified_cnn end-to-end (44–46)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 44 | 🟢 | `unified` | N/A | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | 19.0 m | 20.0 m |
| 45 | 🟢 | `unified` | N/A | 112×112 | spatial | raw | 1–228 | ✅ | ✅ | 29.0 m | 26.3 m |
| 46 | 🟢 | `unified` | N/A | 112×112 | spatial | balanced | 24–216 | ✅ | ✅ | 21.0 m | 23.9 m |

### Fase 7 — Ablación de género e hiperparámetros (47–53)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 47 | 🟢 | `backbone` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ✅ | ❌ | 16.5 m | 18.3 m |
| 48 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ❌ | ❌ | 18.5 m | 19.0 m |
| 49 | 🟢 | `backbone` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ✅ | ✅ | 17.4 m | 18.6 m |
| 50 | 🟢 | `backbone` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ✅ | ✅ | 16.8 m | 18.5 m |
| 51 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ❌ | ✅ | 19.0 m | 19.5 m |
| 52 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ❌ | ✅ | 26.2 m | 25.0 m |
| 53 | 🟢 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 1–228 | ❌ | ❌ | 18.3 m | 17.5 m |

### Fase 8 — Resolución 224px y segmentación cropped (55–58)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 55 | 🟢 | `backbone` | DenseNet121 | 224×224 | spatial | raw | 1–228 | ✅ | ✅ | 13.4 m | 15.0 m |
| 56 | 🟢 | `bbone_vec` | DenseNet121 | 224×224 | spatial | raw | 1–228 | ❌ | ❌ | 15.5 m | 16.0 m |
| **57** | 🟢 | **`backbone`** | **DenseNet121** | **224×224** | **cropped** | **raw** | **1–228** | ✅ | ✅ | **10.7 m ★** | **13.5 m ★** |
| **58** | 🟢 | **`bbone_vec`** | **DenseNet121** | **224×224** | **cropped** | **raw** | **1–228** | ❌ | ✅ | 16.8 m | 16.4 m |

> **Exp 57 es el mejor resultado del proyecto**: segmentación `cropped` (recorte al bounding box, en vez de `spatial`) combinada con 224×224 gana en RSNA y en MEX a la vez, superando también a Exp 55 (misma resolución, segmentación `spatial`).

### Fase 9 — Baseline whole-hand, sin segmentación (60)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 60 | 🟢 | `whole_hand` | DenseNet121 | 112×112 | N/A (mano completa) | balanced | 24–216 | N/A | ✅ | 15.87 m | 23.98 m |

> Agregado 2026-08-07 para el Comentario 15 de la segunda ronda de revisión: mismo
> split/protocolo/hiperparámetros que el exp 27 (`backbone`, su comparación directa), pero sin
> segmentar en 4 regiones. Detalle completo, incluida la significancia estadística pareada contra
> el exp 27, en
> [`experimentos_adicionales/analisis.md`](experimentos_adicionales/analisis.md#15-ablación-insuficiente-falta-baseline-whole-hand).

### Tests rápidos (98–99)

| Exp | Estado | Tipo | Backbone | Img | Seg. | CSV | Edad (m) | FREEZE | Género | Val RSNA | Val MEX |
|-----|--------|------|----------|-----|------|-----|----------|:------:|:------:|:--------:|:-------:|
| 98 | 🟡 | `bbone_vec` | DenseNet121 | 112×112 | spatial | raw | 24–216 | ✅ | ✅ | — | — |
| 99 | 🟢 | `simple_cnn` | N/A | 64×64 | spatial | raw | 24–216 | ✅ | ❌ | ~30 m† | ~92 m† |

---

## Notas

★ Mejor MAE en cada columna entre experimentos con validación válida.  
† Valores aproximados: exp 29 desde logs de Windows (script antiguo); exp 99 con solo 80 muestras.  
Exps 00–16 sin `config.py` — no ejecutables; los modelos de 03–16 existen en formato desconocido.  
Exps 17–22 sin `MODEL_TYPE`, `SEGMENT_MODE`, `DATASET_PATH`, `FREEZE_EXTRACTORS`, `SEGMENTATION_MODEL` — no ejecutables sin añadir esos 5 campos.  
`.keras` de 17/18/22 son formato Keras 3.x (ZIP), incompatibles con `boneage_gpu` (Keras 2.10).

### Experimentos de exploración temprana, cerrados sin completar

Exps 17–32: fase exploratoria inicial, superada por las fases 1–9 (23–60), que cubren las mismas
preguntas con configuración completa y validación real. No se completará retroactivamente:

| Exp | Estado | Motivo |
|-----|--------|--------|
| 17–22 | Incompletos | Faltan 5 campos de config; modelos Keras 3.x incompatibles con el entorno actual |
| 24–25 | Sin validación | Nunca se corrió `07_validation.py`/`08_mex_validation.py` |
| 29 | Sin validación | Modelos solo en logs de Windows (script antiguo) |
| 31–32 | Sin validación | Modelos de CV existen, validación nunca ejecutada |
| 30 | Descartado | MAE ~62K m (desbordado) por ausencia de género en la config |
