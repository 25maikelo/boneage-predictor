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

**Rango de edad conservado (meses):**
24, 36, 42, 48, 50, 54, 60, 69, 72, 78, 82, 84, 88, 94, 96, 100, 106, 108, 114, 120, 126, 132, 138, 144, 150, 156, 159, 162, 165, 168, 174, 180, 186, 192, 204, 216

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
