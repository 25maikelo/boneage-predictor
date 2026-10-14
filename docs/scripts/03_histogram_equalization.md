# Script 03: Ecualización de Histograma

## Descripción

`src/preprocessing/03_histogram_equalization.py` mejora el contraste de las radiografías recortadas usando CLAHE (Contrast Limited Adaptive Histogram Equalization).

### Uso

```bash
python src/preprocessing/03_histogram_equalization.py
```

Sin argumentos CLI.

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `CROPPED_IMAGES_DIR` |
| Salida | `EQUALIZED_IMAGES_DIR`, misma cantidad de imágenes |

### Procesamiento (`ecualizacion_clahe`)

1. CLAHE con `clipLimit=4.0`, `tileGridSize=(8,8)`.
2. Mezcla 60% imagen original + 40% imagen ecualizada (`cv2.addWeighted`), para evitar sobre-realce.
3. Ajuste final de contraste/brillo (`alpha=0.9, beta=-10`), aplicado con `np.clip` en vez de `cv2.convertScaleAbs` para que los valores que quedarían negativos saturen a 0 en vez de reflejarse (ver diagnostico.md, hallazgo 6).

### Parámetros configurables

Todos los parámetros numéricos viven en [`config/preprocessing.py`](../../config/preprocessing.py), no hardcodeados en el script: `CLAHE_CLIP_LIMIT`, `CLAHE_TILE_GRID_SIZE`, `BLEND_CLAHE_WEIGHT`, `FINAL_CONTRAST_ALPHA`, `FINAL_CONTRAST_BETA`.

### Notas

- Tiempo aproximado: ~11 min (CPU).
- `clipLimit=4.0` fue validado contra el default de OpenCV (2.0) sobre las 12,811 imágenes raw. Ver [`docs/results/preprocesamiento_diagnostico/diagnostico.md`](../results/preprocesamiento_diagnostico/diagnostico.md) para el diagnóstico completo.
