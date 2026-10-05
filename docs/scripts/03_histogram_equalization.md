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

1. CLAHE con `clipLimit=2.0`, `tileGridSize=(8,8)`.
2. Mezcla 60% imagen original + 40% imagen ecualizada (`cv2.addWeighted`), para evitar sobre-realce.
3. Ajuste final de contraste/brillo (`alpha=0.9, beta=-10`).

### Notas

- Tiempo aproximado: ~11 min (CPU).
- Los parámetros de mezcla están fijos en el código, no son configurables por archivo de config.
