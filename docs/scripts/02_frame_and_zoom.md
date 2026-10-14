# Script 02: Recorte y Zoom

## Descripción

`src/preprocessing/02_frame_and_zoom.py` alinea horizontalmente el eje mayor de la mano en cada radiografía y recorta el fondo sobrante, usando umbralización Otsu + contorno mínimo rotado (sin red neuronal).

### Uso

```bash
python src/preprocessing/02_frame_and_zoom.py
```

Sin argumentos CLI.

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `RAW_IMAGES_DIR` (12,811 imágenes tras revisión manual) |
| Salida | `CROPPED_IMAGES_DIR`, misma cantidad de imágenes |

### Algoritmo (`rotar_y_recortar`)

1. Blur gaussiano + umbral Otsu para separar mano de fondo.
2. Corrección de inversión si el fondo quedó marcado como primer plano (`verificar_inversion`).
3. Dilatación + contorno externo más grande → rectángulo rotado mínimo (`cv2.minAreaRect`).
4. Rotación hacia un canvas expandido (no el tamaño original) que contiene la imagen rotada completa, para no perder contenido cuando la mano está lejos del centro.
5. Recorte al bounding box del contorno ya rotado, acotado a los límites del canvas.

**Salvaguardas:** si no se detecta contorno, si el recorte resulta degenerado (`x0>=x1` o `y0>=y1`), o si el área recortada es menor a `MIN_CROP_AREA_RATIO` de la imagen original, se guarda la imagen sin modificar en vez de un recorte inválido. El estado de cada imagen (`ok`/`no_contour`/`degenerate`/`discard_area`) se guarda en `data/images/crop_fallbacks.json`.

### Parámetros configurables

Todos los parámetros numéricos viven en [`config/preprocessing.py`](../../config/preprocessing.py), no hardcodeados en el script: `GAUSSIAN_BLUR_KERNEL`, `DILATION_KERNEL_SIZE`, `DILATION_ITERATIONS`, `ANGLE_BOUNDARY_TOLERANCE`, `MIN_CROP_AREA_RATIO`.

### Notas

- Tiempo aproximado: ~19 min (CPU).
- No usa el modelo de segmentación (ese es el paso 04); es un recorte geométrico simple basado en contraste.
- Ver [`docs/results/preprocesamiento_diagnostico/diagnostico.md`](../results/preprocesamiento_diagnostico/diagnostico.md) para el diagnóstico de fallos (bajo contraste, corrimiento de píxeles, rotación espuria) y los fixes aplicados.
