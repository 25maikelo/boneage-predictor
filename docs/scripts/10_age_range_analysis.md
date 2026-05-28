# Script 10 — Análisis de Efectividad por Rango de Edad

## Descripción

`src/10_age_range_analysis.py` evalúa en qué rangos de edad el modelo predice con mayor y menor precisión. A diferencia de los scripts 07/08 (que reportan un MAE global), este script descompone el error por intervalo de edad e identifica sesgos sistemáticos.

### Uso

```bash
python src/10_age_range_analysis.py --experiment <n> [--bin-size <m>] [--dataset <val|mex|both>]
# o via SLURM:
sbatch slurm/10_age_range_analysis.slurm <exp> <bin-size> <dataset>
```

### Salidas (`experiments/<n>/age_range_analysis/`)

| Archivo | Contenido |
|---------|-----------|
| `per_image_results.csv` | Predicción, error, edad real y género por imagen |
| `bin_stats.json` | MAE, RMSE, sesgo, ±12 m/±24 m por intervalo |
| `pediatric_stats.json` | Mismas métricas por grupo etario clínico |
| `summary.json` | Resumen global + mejor y peor rango identificados |
| `mae_por_rango.png` | Barras coloreadas de MAE por intervalo |
| `sesgo_por_rango.png` | Sesgo sistemático (+ sobreestima, − subestima) |
| `scatter_error.png` | Dispersión real vs pred + error vs edad real |
| `violin_grupos_etarios.png` | Distribución del error por grupo pediátrico |
| `precision_por_umbral.png` | % predicciones dentro de ±12 m y ±24 m |
| `heatmap_real_vs_pred.png` | Matriz de confusión continua de edades |
| `distribucion_error.png` | Histograma de error y error absoluto |
| `resumen_rangos.png` | Tabla: top 3 mejor / peor / grupos pediátricos |

---

## Configuración aplicable a todos los experimentos

```bash
--bin-size 12  --dataset both
```

---

## Justificación estadística

### 1. Tamaño del intervalo: 12 meses

El dataset de validación contiene **1,425 imágenes** con edades entre 3 y 228 meses (media 127.2 m, std 41.7 m). Se evaluaron tres tamaños de intervalo:

| Bin | N bins | Min n/bin | Bins con n < 10 | Bins con n < 30 |
|-----|--------|-----------|-----------------|-----------------|
| 6 m | 36 | 1 | **9** | 22 |
| **12 m** | **19** | 3 | **2** | 7 |
| 24 m | 9 | 24 | 0 | 1 |

**Bin de 6 m:** 9 de 36 intervalos tienen menos de 10 muestras — insuficiente para estimar MAE con confianza (IC del 95% muy amplio). No recomendado.

**Bin de 12 m:** Solo 2 intervalos con n < 10 (edades extremas, 0–12 m y 216–228 m), el resto tiene poder estadístico aceptable. Coincide con la **unidad clínica estándar** de evaluación de edad ósea en radiología pediátrica.

**Bin de 24 m:** Mayor poder estadístico pero demasiado grueso — enmascara patrones dentro de cada año de desarrollo esquelético.

> **Regla de mínimos:** para estimar un MAE con un IC del 95% de ±3 meses asumiendo std ≈ 15 m, se necesitan al menos **n ≈ (1.96 × 15 / 3)² ≈ 96 muestras** por bin. Con 12 m la mayoría de bins centrales (24–216 m) superan este umbral; los extremos se reportan con advertencia.

---

### 2. Dataset: ambos (`--dataset both`)

| Dataset | N | Rango | Población |
|---------|---|-------|-----------|
| Validación estándar | 1,425 | 3–228 m | RSNA (EE.UU.) |
| Validación mexicana | 100 | — | Hospital mexicano |

Usar ambos permite:
- **Detectar sesgos de generalización:** si el modelo funciona bien en validación estándar pero mal en mexicana para ciertos rangos, hay un sesgo poblacional.
- **Aumentar la muestra en rangos extremos** donde la validación estándar tiene pocos casos.

---

### 3. Métricas reportadas y por qué

| Métrica | Fórmula | Por qué es relevante |
|---------|---------|---------------------|
| **MAE** | mean(\|pred − real\|) | Misma unidad que la medición; fácil de interpretar clínicamente |
| **RMSE** | √mean((pred − real)²) | Penaliza errores grandes; útil para detectar outliers por rango |
| **Sesgo** | mean(pred − real) | Error sistemático: el modelo sobreestima o subestima en ciertos rangos |
| **% dentro de ±12 m** | % con \|error\| ≤ 12 | Umbral clínico de 1 año — estándar en literatura de bone age |
| **% dentro de ±24 m** | % con \|error\| ≤ 24 | Umbral de aceptabilidad clínica extendido |
| **P10 / P90** | percentiles del \|error\| | Comportamiento en casos fáciles y difíciles respectivamente |

El **sesgo** es especialmente importante: un modelo con MAE=15 m pero sesgo=+14 m sistemáticamente sobreestima la edad — en contexto clínico esto tiene implicaciones diagnósticas distintas a un error aleatorio.

---

### 4. Grupos pediátricos clínicos

El script agrupa adicionalmente por rangos con significado clínico estandarizado:

| Grupo | Rango | Relevancia clínica |
|-------|-------|-------------------|
| Lactante | 0–24 m | Osificación primaria activa; muy pocos casos en el dataset |
| Preescolar | 24–60 m | Aparición de centros de osificación carpal |
| Escolar | 60–120 m | Período de crecimiento estable; mayor densidad de muestras |
| Adolescente temprano | 120–168 m | Pico de velocidad de crecimiento; mayor variabilidad |
| Adolescente tardío | 168–228 m | Cierre de epífisis; predicción más difícil |

---

## Cómo correr en todos los experimentos completados

```bash
cd /lustre/home/mlozano/boneage-predictor
for exp in 33 34 35 36 37 38 39 40 41 42 43 44 45 46; do
  sbatch slurm/10_age_range_analysis.slurm $exp 12 both
done
```

Los resultados de cada experimento se guardarán en `experiments/<n>/age_range_analysis/` y pueden compararse entre arquitecturas para identificar en qué rangos de edad cada modelo tiene ventaja comparativa.
