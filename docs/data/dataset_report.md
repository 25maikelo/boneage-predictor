# Dataset Report — Bone Age Predictor

## Procesamiento de Imágenes

| Etapa | Imágenes | Notas |
|---|---:|---|
| RSNA descarga original | 13,014 | PNG en `data/images/raw/` |
| Eliminadas (calidad) | 13 | Revisión manual — baja calidad |
| Volteadas (orientación) | 190 | Mano izquierda volteada |
| **Dataset raw final** | **12,811** | Disponibles para procesamiento |
| Cropped (recorte + zoom) | 12,811 | Script 02 — 19 min |
| Equalized (CLAHE) | 12,811 | Script 03 — 11 min |
| Segmented (4 regiones) | 51,244 | Script 04 — 2h 05 min (GPU) |

---

## Construcción del Dataset de Entrenamiento

| Etapa | Imágenes | Notas |
|---|---:|---|
| CSV training dataset | 12,611 | 6,833 ♂ + 5,778 ♀ |
| Con 4 segmentos completos | 12,611 | 0 descartadas por segmentos faltantes |
| Filtro edad (< 50 imgs/mes) | −828 | 124 edades eliminadas de 160 posibles |
| **Dataset balanceado** | **11,783** | 6,313 ♂ + 5,470 ♀ · 36 edades |
| Split test (20 %) | ~2,357 | Reservado, no visto en entrenamiento |
| Split train + val (80 %) | ~9,426 | Base para cross-validation |

---

## Cross-Validation

| Parámetro | Valor |
|---|---|
| Estrategia | K-Fold estratificado |
| Número de folds | 5 |
| Tamaño aproximado por fold | ~7,540 train / ~1,883 val |
| Aplicado a | Modelos de segmento (pinky, middle, thumb, wrist) |

---

## Conjuntos de Evaluación

| Conjunto | Imágenes | Descripción |
|---|---:|---|
| Validación estándar (RSNA) | 1,425 | Dataset independiente del entrenamiento |
| Validación mexicana (IMSS) | 100 | Pacientes mexicanos — evaluación de generalización |

---

## Distribución de Edades por Conjunto

### Training — dataset raw (12,611 imágenes)

**160 edades únicas · rango 1–228 meses.** La mayoría de las edades fuera del rango 24–216m son
valores aislados con muy pocas muestras (frecuentemente 1–10 imágenes); las pocas excepciones con
conteo alto están cerca del centro de la distribución (p.ej. 120m: 992, 132m: 1,084, 156m: 1,113).
Las 124 edades con <50 muestras quedan excluidas del dataset balanceado (ver abajo); distribución
completa reproducible desde `data/training/boneage-training-dataset.csv`.

---

### Training — dataset balanceado (11,783 imágenes)

**36 edades · rango 24–216 meses · criterio: ≥ 50 imágenes por mes de edad.** Conteo por edad
reproducible desde `data/training/dataset_analysis/balanced_dataset.csv`; picos en 120/132/156
meses (~1,000+ imágenes cada uno), resto entre 55 y 500 imágenes por edad.

---

### Validación estándar — RSNA (1,425 imágenes)

**82 edades únicas · rango 3–228 meses.** De estas, **36 edades coinciden** con el training
balanceado y **46 no tienen ejemplos en entrenamiento** (56% de las edades presentes en
validación). El modelo produce salida continua (regresión lineal), por lo que puede predecir
cualquier valor, pero en esas 46 edades nunca vio ejemplos etiquetados durante el entrenamiento.
Las que además caen fuera del rango 24–216m (p.ej. 3, 6, 12, 228 meses) representan extrapolación
real, no solo interpolación entre clases no vistas. Lista completa de edades dentro/fuera del
training y distribución detallada reproducibles desde `data/validation/validation_dataset.csv`.

---

### Validación mexicana — IMSS (100 imágenes)

**26 edades únicas · rango 19–216 meses.** Las edades corresponden a la edad ósea radiológica
(`bone_age`, estándar TW3) en años, convertida a meses. Varias edades intermedias (p.ej. 71, 73,
85, 97 m) no existen en el training balanceado, igual que en la validación RSNA. Distribución
completa reproducible desde `data/mex-validation/mex_dataset.csv`.
