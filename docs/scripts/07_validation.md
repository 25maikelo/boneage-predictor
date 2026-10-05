# Script 07: Validación Estándar

## Descripción

`src/07_validation.py` evalúa el modelo de fusión (o unificado) de un experimento sobre el dataset de validación RSNA, generando mapas de saliencia por segmento, gráficos de dispersión y estadísticas resumen.

### Uso

```bash
python src/07_validation.py --experiment N
# o via SLURM:
sbatch slurm/07_validation.slurm N
```

Único argumento: `--experiment` (requerido).

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `VALIDATION_CSV` + imágenes de `data/validation/` (1,425 imágenes), `experiments/N/models/` |
| Salida | `experiments/N/validation/` |

Contenido de la salida:

| Archivo | Contenido |
|---|---|
| `plot_data.json` | `ages`, `gender`, `scatter` (`ids`/`trues`/`preds`, usado por el script 11 para pruebas pareadas), `summary` (MAE, n procesadas/fallidas) |
| `histograma_edad_validacion.png` / `sexo_pastel_validacion.png` | Distribución del dataset de validación |
| `scatter_pred_vs_real.png` / `validation_summary.png` | Dispersión predicción vs. edad real |
| `sample_result_<id>.png` | Muestras individuales con mapa de saliencia superpuesto (reservoir sampling, `NUM_SAMPLE_RESULTS` casos) |

### Pipeline por imagen

Si la imagen de validación no viene pre-segmentada, el script reproduce in-line el preprocesamiento completo: `frame_and_zoom` → `clahe_equalize` → segmentación (`segment_spatial` o `segment_crop_bbox`, según `cfg`) antes de pasar por los modelos de segmento/fusión. Esto permite validar directamente contra radiografías crudas, no solo contra la carpeta ya segmentada.

### Notas

- `load_all_models` resuelve rutas de modelo con compatibilidad hacia atrás (`_resolve_model_path`, `_load_seg_model_compat`) para experimentos antiguos con convenciones de nombre distintas.
- El campo `scatter.ids` en `plot_data.json` es el que permite emparejar predicciones entre experimentos por paciente (script 11); experimentos muy antiguos sin ese campo requieren re-ejecutar este script para habilitar pruebas pareadas.
