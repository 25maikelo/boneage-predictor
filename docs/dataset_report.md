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

**160 edades únicas · rango 1–228 meses**

La mayoría son edades aisladas con muy pocas muestras. Las edades con ≥ 50 muestras son las que pasan al dataset balanceado.

| Edad (m) | n | | Edad (m) | n | | Edad (m) | n | | Edad (m) | n |
|---:|---:|-|---:|---:|-|---:|---:|-|---:|---:|
| 60 | 278 | | 94 | 492 | | 120 | 992 | | 156 | 1,113 |
| 69 | 193 | | 96 | 302 | | 126 | 198 | | 159 | 69 |
| 72 | 254 | | 100 | 60 | | 132 | 1,084 | | 162 | 682 |
| 78 | 55 | | 106 | 478 | | 138 | 529 | | 165 | 64 |
| 82 | 385 | | 108 | 312 | | 144 | 657 | | 168 | 892 |
| 84 | 274 | | 114 | 63 | | 150 | 678 | | 174 | 97 |
| 88 | 55 | | | | | | | | 180 | 418 |
| 24 | 77 | | 36 | 106 | | 42 | 89 | | 48 | 71 |
| 50 | 95 | | 54 | 89 | | 186 | 138 | | 192 | 172 |
| 204 | 200 | | 216 | 72 | | | | | | |

**Edades eliminadas (< 50 muestras) — 124 edades:**

| Edad (m) | n | | Edad (m) | n | | Edad (m) | n | | Edad (m) | n |
|---:|---:|-|---:|---:|-|---:|---:|-|---:|---:|
| 1 | 1 | | 40 | 2 | | 91 | 2 | | 140 | 4 |
| 4 | 1 | | 43 | 1 | | 93 | 1 | | 141 | 2 |
| 6 | 2 | | 45 | 10 | | 101 | 4 | | 142 | 1 |
| 9 | 5 | | 46 | 5 | | 102 | 46 | | 143 | 1 |
| 10 | 4 | | 49 | 1 | | 103 | 1 | | 146 | 3 |
| 12 | 11 | | 51 | 11 | | 104 | 3 | | 147 | 8 |
| 13 | 2 | | 52 | 1 | | 105 | 1 | | 148 | 1 |
| 14 | 1 | | 55 | 12 | | 107 | 1 | | 149 | 2 |
| 15 | 16 | | 56 | 2 | | 109 | 1 | | 151 | 1 |
| 16 | 3 | | 57 | 18 | | 110 | 1 | | 152 | 5 |
| 17 | 2 | | 58 | 4 | | 111 | 2 | | 153 | 42 |
| 18 | 27 | | 62 | 3 | | 112 | 8 | | 154 | 3 |
| 20 | 1 | | 63 | 2 | | 113 | 48 | | 158 | 4 |
| 21 | 15 | | 64 | 7 | | 115 | 5 | | 160 | 3 |
| 27 | 9 | | 65 | 2 | | 116 | 1 | | 161 | 1 |
| 28 | 10 | | 66 | 39 | | 117 | 1 | | 163 | 2 |
| 29 | 1 | | 67 | 3 | | 118 | 2 | | 164 | 6 |
| 30 | 38 | | 70 | 4 | | 121 | 2 | | 166 | 2 |
| 32 | 24 | | 74 | 1 | | 123 | 2 | | 167 | 1 |
| 33 | 13 | | 75 | 15 | | 124 | 4 | | 169 | 2 |
| 34 | 3 | | 76 | 25 | | 125 | 6 | | 170 | 3 |
| 37 | 1 | | 77 | 1 | | 128 | 3 | | 171 | 1 |
| 38 | 1 | | 80 | 1 | | 129 | 1 | | 172 | 1 |
| 39 | 16 | | 81 | 2 | | 130 | 2 | | 173 | 1 |
| | | | 86 | 1 | | 133 | 1 | | 176 | 3 |
| | | | 87 | 3 | | 134 | 3 | | 177 | 1 |
| | | | 90 | 49 | | 135 | 27 | | 179 | 1 |
| | | | | | | 136 | 6 | | 182 | 1 |
| | | | | | | 137 | 4 | | 183 | 5 |
| | | | | | | 139 | 3 | | 184 | 3 |
| | | | | | | | | | 188 | 1 |
| | | | | | | | | | 189 | 15 |
| | | | | | | | | | 194 | 1 |
| | | | | | | | | | 196 | 2 |
| | | | | | | | | | 197 | 1 |
| | | | | | | | | | 198 | 31 |
| | | | | | | | | | 200 | 2 |
| | | | | | | | | | 206 | 1 |
| | | | | | | | | | 210 | 12 |
| | | | | | | | | | 212 | 2 |
| | | | | | | | | | 214 | 1 |
| | | | | | | | | | 222 | 2 |
| | | | | | | | | | 228 | 19 |

---

### Training — dataset balanceado (11,783 imágenes)

**36 edades · rango 24–216 meses · criterio: ≥ 50 imágenes por mes de edad**

| Edad (m) | n | | Edad (m) | n | | Edad (m) | n | | Edad (m) | n |
|---:|---:|-|---:|---:|-|---:|---:|-|---:|---:|
| 24 | 77 | | 60 | 278 | | 108 | 312 | | 156 | 1,113 |
| 36 | 106 | | 69 | 193 | | 114 | 63 | | 159 | 69 |
| 42 | 89 | | 72 | 254 | | 120 | 992 | | 162 | 682 |
| 48 | 71 | | 78 | 55 | | 126 | 198 | | 165 | 64 |
| 50 | 95 | | 82 | 385 | | 132 | 1,084 | | 168 | 892 |
| 54 | 89 | | 84 | 274 | | 138 | 529 | | 174 | 97 |
| | | | 88 | 55 | | 144 | 657 | | 180 | 418 |
| | | | 94 | 492 | | 150 | 678 | | 186 | 138 |
| | | | 96 | 302 | | | | | 192 | 172 |
| | | | 100 | 60 | | | | | 204 | 200 |
| | | | 106 | 478 | | | | | 216 | 72 |

---

### Validación estándar — RSNA (1,425 imágenes)

**82 edades únicas · rango 3–228 meses**

De estas, **36 edades coinciden** con el training balanceado y **46 edades no tienen ejemplos en entrenamiento**. El modelo produce salida continua (regresión lineal), por lo que puede predecir cualquier valor, pero en esas 46 edades nunca vio ejemplos etiquetados. Las que además caen fuera del rango 24–216 m (p.ej. 3, 6, 228 m) representan extrapolación real.

**Edades dentro del training balanceado (36):**
24, 36, 42, 48, 50, 54, 60, 69, 72, 78, 82, 84, 88, 94, 96, 100, 106, 108, 114, 120, 126, 132, 138, 144, 150, 156, 159, 162, 165, 168, 174, 180, 186, 192, 204, 216

**Edades fuera del training balanceado (46):**
3, 6, 12, 14, 15, 16, 18, 21, 26, 28, 30, 32, 34, 38, 39, 43, 51, 53, 55, 57, 63, 64, 65, 66, 76, 90, 102, 107, 113, 116, 135, 141, 147, 148, 152, 153, 160, 166, 167, 176, 183, 184, 188, 189, 210, 228

Distribución detallada:

| Edad (m) | n | | Edad (m) | n | | Edad (m) | n | | Edad (m) | n |
|---:|---:|-|---:|---:|-|---:|---:|-|---:|---:|
| 3 | 1 | | 60 | 31 | | 120 | 114 | | 168 | 105 |
| 6 | 1 | | 63 | 1 | | 126 | 21 | | 174 | 5 |
| 12 | 1 | | 64 | 1 | | 132 | 125 | | 176 | 1 |
| 14 | 1 | | 65 | 1 | | 135 | 2 | | 180 | 45 |
| 15 | 3 | | 66 | 3 | | 138 | 55 | | 183 | 2 |
| 16 | 1 | | 69 | 21 | | 141 | 1 | | 184 | 1 |
| 18 | 4 | | 72 | 31 | | 144 | 74 | | 186 | 16 |
| 21 | 1 | | 76 | 6 | | 147 | 2 | | 188 | 1 |
| 24 | 11 | | 78 | 6 | | 148 | 1 | | 189 | 1 |
| 26 | 1 | | 82 | 39 | | 150 | 78 | | 192 | 23 |
| 28 | 2 | | 84 | 32 | | 152 | 1 | | 204 | 24 |
| 30 | 1 | | 88 | 11 | | 153 | 8 | | 210 | 5 |
| 32 | 1 | | 90 | 6 | | 156 | 116 | | 216 | 6 |
| 34 | 1 | | 94 | 59 | | 159 | 11 | | 228 | 4 |
| 36 | 13 | | 96 | 27 | | 160 | 3 | | | |
| 38 | 1 | | 100 | 6 | | 162 | 80 | | | |
| 39 | 4 | | 102 | 9 | | 165 | 4 | | | |
| 42 | 7 | | 106 | 54 | | 166 | 1 | | | |
| 43 | 1 | | 107 | 1 | | 167 | 1 | | | |
| 48 | 16 | | 108 | 37 | | | | | | |
| 50 | 6 | | 113 | 4 | | | | | | |
| 51 | 2 | | 114 | 7 | | | | | | |
| 53 | 1 | | 116 | 1 | | | | | | |
| 54 | 10 | | | | | | | | | |
| 55 | 1 | | | | | | | | | |
| 57 | 2 | | | | | | | | | |

---

### Validación mexicana — IMSS (100 imágenes)

**26 edades únicas · rango 19–216 meses**

Las edades corresponden a la edad ósea radiológica (`bone_age`) en años, convertida a meses. Las edades intermedias (p.ej. 71, 73, 85, 97 m) no existen en el training balanceado.

| Edad (m) | n | | Edad (m) | n | | Edad (m) | n |
|---:|---:|-|---:|---:|-|---:|---:|
| 19 | 1 | | 85 | 4 | | 151 | 1 |
| 24 | 1 | | 96 | 1 | | 156 | 10 |
| 36 | 4 | | 97 | 3 | | 163 | 10 |
| 43 | 1 | | 101 | 1 | | 168 | 8 |
| 60 | 1 | | 108 | 1 | | 180 | 2 |
| 71 | 2 | | 110 | 1 | | 192 | 2 |
| 72 | 5 | | 120 | 13 | | 204 | 2 |
| 73 | 1 | | 132 | 7 | | 216 | 1 |
| 84 | 7 | | 144 | 10 | | | |
