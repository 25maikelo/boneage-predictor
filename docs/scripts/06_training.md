# Script 06: Entrenamiento

## Descripción

`src/06_training.py` es el script central del pipeline: entrena los 4 modelos de segmento (uno por región anatómica) con K-Fold CV y luego construye y entrena el modelo de fusión, en cualquiera de los cinco modos soportados vía `MODEL_TYPE`. Ver [`arquitecturas.md`](../design/arquitecturas.md) para el diagrama y los parámetros de cada modo; este documento cubre solo el flujo de ejecución y las salidas.

### Uso

```bash
python src/06_training.py --experiment N
# o via SLURM:
sbatch slurm/06_training.slurm N
```

Único argumento: `--experiment` (requerido). Todo lo demás (arquitectura, dataset, hiperparámetros) viene del config del experimento (`config/experiments/experiment_N.py` o equivalente, vía `load_experiment_config`).

### Flujo según `MODEL_TYPE`

| `MODEL_TYPE` | Función de entrenamiento | Fases |
|---|---|---|
| `backbone` / `simple_cnn` / `backbone_vectors` | `train_one_segment` (×4, con reanudación: salta segmentos ya entrenados) + `train_fusion` | Segmentos (CV) → Fusión → Fine-tuning |
| `unified_cnn` | `train_unified_cnn` | Una sola fase end-to-end, sin segmento+fusión separados |
| `whole_hand` | `train_whole_hand` | Mismo esquema de 2 fases que fusión normal, pero sobre la mano completa sin segmentar |

Si `USE_CROSS_VALIDATION=True`, al terminar los segmentos se genera un resumen de CV (`report_cv_results`) antes de pasar a la fusión.

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `data/images/segmented/` (o `whole_hand`/`unified_cnn`: imagen completa), CSV de `DATASET_PATH` |
| Salida | `experiments/N/models/` (`{segmento}_model`, `fusion_model` o `unified_model`), `experiments/N/training_history/` |

### Notas

- `train_one_segment` es reanudable: si `experiments/N/models/{seg}_model` ya existe, ese segmento se omite al reejecutar el script.
- `SaveModelCallback` guarda el mejor modelo por época según la métrica de validación; `WarmupLR` aplica warmup de learning rate en las primeras épocas.
- Tiempos y fases varían fuertemente según `MODEL_TYPE` y tamaño del dataset; ver [`pipeline.md`](../planning/pipeline.md#06--entrenamiento) para el detalle de fases y [`2026-07-22_experiments_master.md`](../results/2026-07-22_experiments_master.md) para tiempos reales por experimento.
