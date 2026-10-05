# Script 08: Validación Mexicana

## Descripción

`src/08_mex_validation.py` evalúa el modelo de un experimento sobre el dataset de pacientes mexicanos (IMSS), para medir generalización geográfica. Misma lógica que el script 07, aplicada a `MEX_CSV`/`MEX_IMAGES_DIR` en vez del set RSNA.

> ⚠️ **Historial de bug corregido:** hasta el commit `9ccd9a6`, este script comparaba las predicciones contra `real_age` (edad cronológica) en vez de `bone_age` (edad ósea TW3, lo que el modelo realmente predice), inflando artificialmente el MAE reportado. Ya está corregido. Experimentos con números de MEX citados en otros documentos de `docs/results/` **anteriores a esa corrección** pueden estar desactualizados; ver banners de corrección en esos archivos y los números ya recalculados en [`results/experimentos_adicionales/analisis.md`](../results/experimentos_adicionales/analisis.md).

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

- El campo clave para comparar contra el bug histórico es `summary.mae` en `plot_data.json`: debe calcularse contra `bone_age`, no `real_age`.
- Como el dataset mexicano reporta edad ósea directamente (estándar TW3 de origen), no requiere la reconstrucción de pipeline in-line que hace el script 07 para RSNA si las imágenes ya vienen segmentadas.
