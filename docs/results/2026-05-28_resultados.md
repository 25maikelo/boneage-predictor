# Resultados — Bone Age Predictor

> 14 experimentos completados · 4 arquitecturas · 3 datasets · 2026-04-29

---

## Arquitecturas evaluadas

| Arquitectura | Descripción |
|---|---|
| `simple_cnn` | 4 CNNs independientes (una por segmento) + modelo de fusión |
| `backbone` | 4 DenseNet121 desde cero + fusión escalar |
| `backbone_vectors` | 4 DenseNet121 desde cero + fusión con vectores 256-dim |
| `unified_cnn` | Una sola CNN que recibe los 4 segmentos y predice directamente |

---

## Datasets utilizados

| Dataset | Rango de edad | Imágenes | Descripción |
|---------|--------------|----------|-------------|
| **recortado** | 24–216 m | ~12,499 | Dataset raw filtrado eliminando las colas de la distribución (edades extremas con pocos casos) |
| **completo** | 1–228 m | 12,611 | Dataset raw sin ningún filtro; incluye todas las edades aunque estén poco representadas |
| **balanceado** | variable | 11,783 | Solo clases con ≥50 imágenes (36 edades); distribución más uniforme entre grupos de edad |
| **completo+libre** | 1–228 m | 12,611 | Igual que completo, pero con los extractores de segmento **descongelados** durante la fusión |

**Extractores congelados vs descongelados:** en el pipeline de dos etapas, los modelos de segmento se entrenan primero y luego se usan como extractores de características para el modelo de fusión. En la variante **congelada** (default), los pesos de los extractores no se modifican durante la fusión — el modelo de fusión solo aprende a combinar representaciones fijas. En la variante **libre** (`completo+libre`), los pesos de los extractores se actualizan junto con la fusión, permitiendo que toda la red se adapte end-to-end a la tarea final de predicción de edad ósea.

---

## Tabla de resultados

🥇🥈🥉 = podio global · **negrita** = mejor por arquitectura · **±12m** = porcentaje de predicciones con error menor a 1 año (12 meses)

| Exp | Arquitectura | Dataset | Val MAE | Mex MAE | Fusión MAE | ±12m | Sesgo |
|-----|-------------|---------|:-------:|:-------:|:----------:|:----:|:-----:|
| 🥇 **34** | `backbone` | recortado | **14.6 m** | 17.6 m | **9.2 m** | 50.9% | +0.8 m |
| 🥈 **40** | `backbone` | balanceado | **15.1 m** | **13.9 m** | 9.0 m | 47.1% | −0.7 m |
| 🥉 **37** | `backbone` | completo | 15.4 m | 16.7 m | **6.8 m** | 48.9% | +1.3 m |
| **44** | `unified_cnn` | recortado | **19.0 m** | **16.9 m** | N/A | 35.9% | −1.8 m |
| **46** | `unified_cnn` | balanceado | 21.0 m | 21.9 m | N/A | 35.3% | +0.9 m |
| **43** | `backbone_vectors` | completo+libre | 23.0 m | 18.5 m | **17.3 m** | 28.8% | −12.8 m |
| **42** | `simple_cnn` | completo+libre | 24.1 m | 20.0 m | 20.1 m | 30.7% | −11.0 m |
| 45 | `unified_cnn` | completo | 29.0 m | 21.0 m | N/A | 24.3% | −18.5 m |
| 36 | `simple_cnn` | completo | 30.2 m | 22.2 m | 25.5 m | 22.5% | −20.2 m |
| 35 | `backbone_vectors` | recortado | 36.7 m | 23.4 m | 27.6 m | 14.7% | −28.0 m |
| 41 | `backbone_vectors` | balanceado | 35.0 m | 27.3 m | 26.9 m | 15.5% | −26.5 m |
| 33 | `simple_cnn` | recortado | 39.2 m | 35.9 m | 34.5 m | 11.8% | −31.6 m |
| 38 | `backbone_vectors` | completo | 40.0 m | 28.0 m | 30.3 m | 13.4% | −31.6 m |
| 39 | `simple_cnn` | balanceado | 43.5 m | 35.9 m | 41.8 m | 9.2% | −35.8 m |

> **Val MAE:** dataset RSNA (1,393 imgs) · **Mex MAE:** dataset mexicano (98 imgs) · **Fusión MAE:** integrador final · **±12m:** % predicciones dentro de 1 año · **Sesgo:** + sobreestima, − subestima

---

## Hallazgos principales

### 1. `backbone` es la arquitectura más precisa
Val MAE consistente de **14–15 m** en los tres datasets. Sesgo prácticamente nulo (±1 m). La fusión escalar (6–9 m) es la más efectiva de todas las estrategias de integración.

### 2. Descongelar los extractores es el mayor salto de rendimiento
| | Congelado | Libre | Mejora |
|--|--|--|--|
| `simple_cnn` (completo) | 30.2 m (exp 36) | **24.1 m** (exp 42) | **−6.1 m** |
| `backbone_vectors` (completo) | 40.0 m (exp 38) | **23.0 m** (exp 43) | **−17.0 m** |

Permitir que los extractores se ajusten durante la fusión reduce el error hasta en 17 meses.

### 3. `unified_cnn` es el segundo mejor sin pipeline de dos etapas
Val MAE de **19–21 m** en recortado y balanceado, con sesgo casi nulo (±2 m). Demuestra que el entrenamiento end-to-end es una alternativa eficiente al esquema segmento+fusión.

### 4. El dataset no define el ganador, la arquitectura sí
`backbone` obtiene resultados similares en los tres datasets (14–15 m). `simple_cnn` y `backbone_vectors` congelados varían mucho más según el dataset.

### 5. Zonas críticas por rango de edad

| Arquitectura | Mejor rango | Peor rango |
|---|---|---|
| `backbone` | 96–108 m (8–9 a) | 228–240 m (19–20 a) — adolescentes tardíos |
| `unified_cnn` | 96–108 m (8–9 a) | 228–240 m (19–20 a) — adolescentes tardíos |
| `simple_cnn` / `bbone_vec` (congelado) | 36–60 m (3–5 a) | 228–240 m (19–20 a) — adolescentes tardíos |
| `simple_cnn` libre | 72–84 m (6–7 a) | 228–240 m (19–20 a) — adolescentes tardíos |
| `bbone_vec` libre | 12–24 m (1–2 a)* | 228–240 m (19–20 a) — adolescentes tardíos |

> El extremo **228–240 m es universalmente el peor rango en todas las arquitecturas** — las edades próximas al cierre de epífisis son las más difíciles de predecir.
> *Bin 12–24 m tiene solo 11 imágenes — muestra pequeña.

