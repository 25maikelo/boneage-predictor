# Script 05: Análisis y Balanceo del Dataset

## Descripción

`src/05_dataset_analysis.py` filtra el CSV de entrenamiento para conservar solo las imágenes que tienen los 4 segmentos generados, y aplica un umbral mínimo de muestras por edad para construir el dataset balanceado.

### Uso

```bash
python src/05_dataset_analysis.py --experiment N
```

| Argumento | Default | Descripción |
|---|---|---|
| `--experiment` | `26` | Número de experimento, solo para leer `SEGMENTS_ORDER` de su config (no afecta el resultado salvo el orden de verificación) |

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `TRAINING_CSV` (`data/training/boneage-training-dataset.csv`, 12,611 filas) |
| Salida | `DATASET_ANALYSIS_DIR/balanced_dataset.csv` (11,783 filas · 36 edades · 24–216 meses) |

Salidas adicionales en `DATASET_ANALYSIS_DIR/`:

| Archivo | Contenido |
|---|---|
| `dataset_statistics.json` | Conteos (total, con 4 segmentos, balanceado), rango de edad, edades descartadas/consideradas |
| `plot_data.json` | Datos de histogramas para regeneración multiidioma |
| `age_distribution.png` / `filtered_age_distribution.png` / `balanced_age_distribution.png` | Histogramas de edad: original, filtrado (con 4 segmentos), balanceado |
| `dataset_proportion.png` | Pastel de imágenes completas vs. con segmentos faltantes |

### Criterio de balanceo

`UMBRAL_MIN_IMAGENES = 50`: se descartan edades (en meses) con menos de 50 imágenes. Con el dataset actual esto excluye 124 de 160 edades (−828 imágenes), concentradas en los extremos de la distribución. Ver [`dataset_report.md`](../data/dataset_report.md) para el detalle numérico completo.

### Notas

- Tiempo aproximado: ~18 s.
- Este script no entrena nada; solo produce el CSV que luego usan los experimentos "balanceado" (39–41, 46, vía `DATASET_PATH` en su config).
