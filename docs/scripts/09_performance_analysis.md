# Script 09: Análisis de Desempeño

## Descripción

`src/09_performance_analysis.py` genera la tabla comparativa de métricas (train/val loss y MAE) por modelo de segmento y por fusión, además de mapas de saliencia ilustrativos sobre 3 muestras fijas del dataset balanceado.

### Uso

```bash
python src/09_performance_analysis.py --experiment N
# o via SLURM:
sbatch slurm/09_performance_analysis.slurm N
```

Único argumento: `--experiment` (requerido).

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `experiments/N/models/`, 3 muestras fijas (`random_state=42`) de `BALANCED_DATASET_CSV` |
| Salida | `experiments/N/evaluation/` |

| Archivo | Contenido |
|---|---|
| `comparative_table_data.json` / `tabla_comparativa.png` | Filas: Fusionado + cada segmento (Meñique/Medio/Pulgar/Muñeca); columnas: parámetros, loss train/val, MAE train/val |
| `sample_result_<id>.png` | Imagen + predicción + edad real + mapa de saliencia superpuesto por segmento, para 3 casos fijos |

### Notas

- Las 3 muestras son siempre las mismas (semilla fija) para poder comparar visualmente entre experimentos.
- Para `MODEL_TYPE="unified_cnn"` no hay modelos de segmento independientes (`unified_model` en vez de `fusion_model` + `{seg}_model`); la tabla comparativa en ese caso solo tiene la fila del modelo unificado.
- Este script reutiliza directamente los pesos ya entrenados; no reentrena nada.
