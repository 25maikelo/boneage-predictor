# Script 08: Validación Mexicana

## Descripción

`src/08_mex_validation.py` evalúa el modelo de un experimento sobre el dataset de pacientes mexicanos (IMSS), para medir generalización geográfica. Misma lógica que el script 07, aplicada a `MEX_CSV`/`MEX_IMAGES_DIR` en vez del set RSNA. La comparación es contra `bone_age` (edad ósea TW3, lo que el modelo predice), no contra `real_age` (edad cronológica).

### Uso

```bash
python src/08_mex_validation.py --experiment N
# o via SLURM:
sbatch slurm/08_mex_validation.slurm N
```

Único argumento: `--experiment` (requerido). Corre en la partición `q1` (CPU), ya que el dataset de 100 imágenes no justifica GPU.

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `MEX_CSV` + `MEX_IMAGES_DIR` (`data/mex-validation/`, 100 imágenes), `experiments/N/models/` |
| Salida | `experiments/N/mex-validation/` |

Mismo formato de salida que el script 07 (`plot_data.json` con `ages`/`gender`/`scatter`/`summary`, histogramas, scatter, muestras con saliencia). De las 100 imágenes, típicamente 99 se procesan y 1 falla (caso recurrente, no específico de un experimento).

### Notas

Como el dataset mexicano reporta edad ósea directamente (estándar TW3 de origen), no requiere la reconstrucción de pipeline in-line que hace el script 07 para RSNA si las imágenes ya vienen segmentadas.
