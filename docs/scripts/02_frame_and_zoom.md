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
4. Rotación de la imagen completa para alinear el eje mayor de la mano.
5. Recorte al bounding box del contorno ya rotado.

**Salvaguardas:** si no se detecta contorno, si el recorte resulta degenerado (`x0>=x1` o `y0>=y1`), o si el área recortada es menor al 20% de la imagen original, se guarda la imagen sin modificar en vez de un recorte inválido.

### Notas

- Tiempo aproximado: ~19 min (CPU).
- No usa el modelo de segmentación (ese es el paso 04); es un recorte geométrico simple basado en contraste.
