# Script 01: Entrenamiento del Detector de Mano

## Descripción

`src/preprocessing/01_train_hand_detector.py` entrena el modelo de segmentación semántica que identifica las 4 regiones anatómicas de la mano (meñique, medio, pulgar, muñeca) más el fondo, a partir de anotaciones manuales en formato LabelMe.

Clases de salida: `0=fondo, 1=pinky, 2=middle, 3=thumb, 4=wrist`.

### Uso

```bash
python src/preprocessing/01_train_hand_detector.py
# o via SLURM:
sbatch slurm/01_train_hand_detector.slurm
```

Todos los hiperparámetros (arquitectura, tamaño de imagen, batch size, épocas, augmentation) se configuran en `config/segmentation.py`, no por CLI.

### Arquitecturas disponibles (`ARCHITECTURE` en `config/segmentation.py`)

| Valor | Descripción |
|---|---|
| `unet` | U-Net puro, sin backbone preentrenado |
| `unet_mobilenetv2` | U-Net con encoder MobileNetV2 + skip connections (default) |
| `mobilenetv2_sym` | Encoder MobileNetV2 + decoder simétrico inverted-residual (espeja la estructura del encoder) |
| `unet_resnet50` | U-Net con encoder ResNet50; soporta pesos `imagenet`, `radimagenet` o `None` |
| `unet_densenet121` | U-Net con encoder DenseNet121; soporta pesos `imagenet`, `radimagenet` o `None` |

Para `radimagenet` se requiere el archivo de pesos correspondiente en `models/pretrained/`.

### Entrada / Salida

| | Ruta |
|---|---|
| Entrada | `HAND_DETECTOR_IMAGES_DIR` + `HAND_DETECTOR_ANNOTATIONS_DIR` (imágenes + JSON LabelMe) |
| Salida | `models/hand-detector/hand-detector_NN/` (run numerado automáticamente, o fijo via `HAND_DETECTOR_RUN_DIR`) |

Contenido de cada run: `models/modelo_segmentacion.h5`, `training_history/` (curvas de loss/IoU/Dice), `evaluation/` (tabla de métricas, muestras de predicción), `config.json` (snapshot de la config usada).

### Métricas

IoU y Dice por clase, calculadas en `iou_metric`/`dice_metric` sobre el set de validación (split interno vía `train_test_split`).

### Notas

- El modelo activo para el pipeline (`src/preprocessing/04_segment_images.py`) se selecciona con `HAND_DETECTOR_RUN` en `config/segmentation.py` (`None` = último run disponible).
- Tiempo aproximado: ~29 min (GPU).
