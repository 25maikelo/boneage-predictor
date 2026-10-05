# Script 04: Segmentación de Imágenes

## Descripción

`src/preprocessing/04_segment_images.py` aplica el modelo de segmentación entrenado en el paso 01 para extraer las 4 regiones anatómicas (`pinky`, `middle`, `thumb`, `wrist`) de cada radiografía ecualizada, además de guardar la máscara de segmentación completa.

### Uso

```bash
python src/preprocessing/04_segment_images.py
python src/preprocessing/04_segment_images.py --mode cropped --padding 0.15
python src/preprocessing/04_segment_images.py --mode both
```

| Argumento | Default | Descripción |
|---|---|---|
| `--mode` | `spatial` | `spatial` mantiene posición/tamaño original enmascarando el fondo (comportamiento histórico); `cropped` recorta al bounding box del segmento + padding; `both` genera ambas variantes en una sola pasada (una sola inferencia del modelo por imagen) |
| `--padding` | `0.15` | Padding relativo al bounding box, solo aplica a `cropped` |

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `EQUALIZED_IMAGES_DIR` (12,811 imágenes) |
| Salida (spatial) | `SEGMENTED_IMAGES_DIR/{pinky,middle,thumb,wrist}/`, 51,244 imágenes |
| Salida (cropped) | `SEGMENTED_CROPPED_IMAGES_DIR/{pinky,middle,thumb,wrist}/` |
| Máscaras | `MASKS_DIR/`, 12,811 PNG (una por imagen original, independiente del modo) |

### Notas

- El modelo usado es el de `HAND_DETECTOR_RUN` (`config/segmentation.py`); por defecto el último run disponible (`get_segmentation_model_path`).
- Si un segmento no se detecta en modo `cropped` (máscara vacía), se guarda una imagen mínima 1×1 del color de fondo en vez de fallar.
- Tiempo aproximado: ~2h 05 min (GPU), modo `spatial` sobre el dataset completo.
