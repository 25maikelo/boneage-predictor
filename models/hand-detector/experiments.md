# Experimentos — Hand Detector

Configuración fija: `IMAGE_SIZE=224×224` · `NUM_CLASSES=5` · `BATCH_SIZE=8` · `optimizer=adam` · `loss=categorical_crossentropy`

`—` indica que el parámetro no aplica a esa arquitectura y se ignora.

## Configuraciones

| # | Arquitectura | Encoder weights | Base model trainable | Canales | Data aug |
|---|--------------|-----------------|:--------------------:|:-------:|:--------:|
| 00 | `unet_mobilenetv2` | `imagenet` | No | 3 | No |
| 01 | `unet_mobilenetv2` | `imagenet` | No | 3 | Sí |
| 02 | `unet_mobilenetv2` | `imagenet` | Sí | 3 | No |
| 03 | `unet_mobilenetv2` | `imagenet` | Sí | 3 | Sí |
| 04 | `unet_mobilenetv2` | `None` | Sí | 3 | No |
| 05 | `unet_mobilenetv2` | `None` | Sí | 3 | Sí |
| 06 | `unet_mobilenetv2` | `None` | No | 3 | No |
| 07 | `unet_mobilenetv2` | `None` | No | 3 | Sí |
| 08 | `unet_mobilenetv2` | `imagenet` | Sí | 1 | No |
| 09 | `unet_mobilenetv2` | `imagenet` | Sí | 1 | Sí |
| 10 | `unet` | — | — | 3 | No |
| 11 | `unet` | — | — | 3 | Sí |
| 12 | `unet` | — | — | 1 | No |
| 13 | `unet` | — | — | 1 | Sí |
| 14 | `mobilenetv2_sym` | `imagenet` | No | 3 | No |
| 15 | `mobilenetv2_sym` | `imagenet` | No | 3 | Sí |
| 16 | `mobilenetv2_sym` | `imagenet` | Sí | 3 | No |
| 17 | `mobilenetv2_sym` | `imagenet` | Sí | 3 | Sí |
| 18 | `mobilenetv2_sym` | `None` | Sí | 3 | No |
| 19 | `mobilenetv2_sym` | `None` | Sí | 3 | Sí |
| 20 | `mobilenetv2_sym` | `imagenet` | Sí | 1 | No |
| 21 | `mobilenetv2_sym` | `imagenet` | Sí | 1 | Sí |
| 22 | `unet_resnet50` | `radimagenet` | No | 3 | No |
| 23 | `unet_resnet50` | `radimagenet` | No | 3 | Sí |
| 24 | `unet_resnet50` | `radimagenet` | Sí | 3 | No |
| 25 | `unet_resnet50` | `radimagenet` | Sí | 3 | Sí |
| 26 | `unet_resnet50` | `imagenet` | Sí | 3 | No |
| 27 | `unet_resnet50` | `imagenet` | Sí | 3 | Sí |
| 28 | `unet_densenet121` | `radimagenet` | No | 3 | No |
| 29 | `unet_densenet121` | `radimagenet` | No | 3 | Sí |
| 30 | `unet_densenet121` | `radimagenet` | Sí | 3 | No |
| 31 | `unet_densenet121` | `radimagenet` | Sí | 3 | Sí |
| 32 | `unet_densenet121` | `imagenet` | Sí | 3 | No |
| 33 | `unet_densenet121` | `imagenet` | Sí | 3 | Sí |

## Notas

- Los experimentos con `encoder_weights=None` entrenan completamente desde cero.
- `unet_mobilenetv2` con `imagenet + trainable=No` congela el encoder — útil como baseline rápido.
- Las combinaciones con `canales=1` replican el canal gris 3 veces antes del backbone (para compatibilidad con imagenet).
- Los experimentos 22–33 usan `unet_resnet50` y `unet_densenet121` con pesos **RadImageNet** (Mei et al. 2022).
  Requieren descargar los pesos `.h5` desde https://github.com/BMEII-AI/RadImageNet y guardarlos en `models/pretrained/`.
- Comparativa RadImageNet vs ImageNet: experimentos 22–25 vs 26–27 (ResNet50), 28–31 vs 32–33 (DenseNet121).
