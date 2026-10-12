# Auditoría de preprocesamiento: `02_frame_and_zoom.py` y `03_histogram_equalization.py`

> Etapa 4 de [`fase_segmentacion.md`](../../planning/fase_segmentacion.md). Evalúa la calidad
> del preprocesamiento con métricas propias de la imagen (geometría del recorte, contraste,
> entropía, saturación), sin depender de ningún modelo de segmentación ni de sus métricas.

## Metodología

Muestra aleatoria de 800 de las 12,813 imágenes raw (`random_state=42`). Script:
[`scripts/auditoria_preprocesamiento.py`](scripts/auditoria_preprocesamiento.py). Resultados
crudos en `results_json/`, figuras en `figures/`.

- **Recorte (`02_frame_and_zoom.py`)**: se replica la lógica exacta del script (Otsu + contorno
  + `minAreaRect`) y se mide, para cada imagen, qué salvaguarda se activó, y un **fill ratio**:
  qué fracción del recorte final sigue siendo mano (primer plano) al re-umbralizar el propio
  recorte. Es la métrica objetiva de "recorte flojo" (queda borde/fondo residual) sin ningún
  modelo de por medio.
- **Ecualización (`03_histogram_equalization.py`)**: se comparan los parámetros actuales
  (`clipLimit=2.0`, `tileGridSize=8×8`) contra variantes, midiendo contraste (desviación
  estándar de intensidad), entropía y tasa de saturación (píxeles en 0 o 255) de cada resultado.

## Hallazgo 1: el recorte deja en promedio 27% de fondo/borde residual

| Métrica | Valor |
|---|---:|
| Fill ratio medio | 0.730 |
| Fill ratio mediana | 0.765 |
| Fill ratio p10 | 0.634 |
| Salvaguarda activada (`<20%` del área) | 42/800 (5.25%) |

El histograma (`figures/fill_ratio_histograma.png`) muestra una distribución con un cuerpo
principal alrededor de 0.75–0.80, pero con una cola notable cerca de 0.0: ~15–20 imágenes (de
800) donde el recorte final es casi todo fondo.

## Hallazgo 2: la causa de los peores recortes es contraste global bajo, no rotación

Las imágenes con fill ratio más bajo (`figures/peor_recorte_*.png`) no son casos de rotación mal
calculada: son radiografías con **brillo/contraste global bajo**, donde Otsu no logra separar
claramente la mano del fondo (ambos quedan oscuros), y el script cae en la salvaguarda de
descarte (se queda con la imagen completa sin recortar, que por diseño tiene mucho fondo).

Esto importa para el orden del pipeline: hoy el recorte (02) corre **antes** de la ecualización
de contraste (03). Si una imagen tiene bajo contraste global, el recorte ya falló antes de que
CLAHE tuviera oportunidad de mejorarla. Invertir el orden (o aplicar una normalización de
contraste ligera antes del umbral de Otsu, solo para detectar el contorno, sin alterar la imagen
que se recorta) es una hipótesis concreta a probar.

## Hallazgo 3: los parámetros actuales de CLAHE no son los mejores de los probados

| Variante | Contraste (std) | Entropía | % saturados |
|---|---:|---:|---:|
| Sin CLAHE | 42.59 | 5.798 | 5.49% |
| **Actual** (`clip=2.0`, `tile=8×8`) | 38.66 | 5.615 | 8.50% |
| `clip=1.0`, `tile=8×8` | 37.86 | 5.483 | 8.38% |
| `clip=4.0`, `tile=8×8` | **39.48** | **5.838** | **7.40%** |
| `clip=2.0`, `tile=4×4` | 39.77 | 5.705 | 7.95% |
| `clip=2.0`, `tile=16×16` | 37.78 | 5.534 | 8.85% |

Los parámetros actuales **reducen el contraste** respecto a no aplicar CLAHE (42.59 → 38.66) y
**aumentan la saturación** (5.49% → 8.50% de píxeles en 0 o 255). De las variantes probadas,
`clipLimit=4.0` con el mismo tamaño de tile (`8×8`) conserva más contraste y satura menos píxeles
que la configuración actual, y de hecho su entropía (5.838) supera incluso a la imagen sin
ecualizar. `tileGridSize=16×16` es la peor variante en ambos criterios.

## Conclusiones

1. El recorte de `02_frame_and_zoom.py` funciona bien en la mayoría de los casos (mediana de
   fill ratio 0.765), pero tiene una cola de fallos (~2–5% del dataset) concentrada en imágenes
   de bajo contraste global, no en casos de rotación compleja.
2. Los parámetros actuales de CLAHE (`clipLimit=2.0`) no son óptimos frente a las alternativas
   probadas en estas métricas: `clipLimit=4.0` preserva más contraste y satura menos píxeles.
3. Ambos hallazgos son independientes entre sí pero sugieren la misma hipótesis: adelantar o
   reforzar el control de contraste antes/durante el recorte podría reducir la cola de fallos
   del hallazgo 1, además de la mejora directa en calidad de imagen del hallazgo 3.

## Validación de las propuestas sobre el dataset completo (12,811 imágenes)

Script: [`scripts/validar_propuestas.py`](scripts/validar_propuestas.py). Resultados crudos en
`results_json/validacion_clahe_clip4.json` y `results_json/validacion_otsu_contraste_previo.json`.

**`clipLimit=4.0` se confirma**: 85.4% de las imágenes muestran más contraste que con
`clipLimit=2.0` (delta medio +0.85 de desviación estándar); la reducción de saturación es más
débil de lo que sugería la muestra de 800 (51.7% de las imágenes satura menos, prácticamente
parejo). Dirección consistente con el hallazgo 3, confirmado a escala completa.

**La normalización de contraste antes de Otsu (`equalizeHist`) NO funciona**: el fill ratio
promedio bajó (0.735 → 0.715, empeora en vez de mejorar), 18.2% de las imágenes empeoran, y de
los 583 peores casos (fill ratio <0.3) solo el 48.5% mejora con este método, prácticamente una
moneda al aire. La tasa de descarte sí bajó un poco (642 → 549 de 12,811), pero a costa de
recortes peores en otros casos. **Esta propuesta queda descartada**, no se adopta.

## Segunda alternativa: sustracción de fondo morfológica

Tras revisar literatura sobre segmentación de radiografías con iluminación/contraste desparejo
([`revision_literatura.md`](revision_literatura.md), sección 3), se identificó la sustracción de
fondo morfológica ("rolling ball"/top-hat) como el método documentado para este problema
específico, mejor fundamentado que una ecualización genérica. En validación sobre el dataset
completo (resultado pendiente al momento de escribir esto, ver
`results_json/validacion_background_subtraction.json` cuando esté disponible).

## Próximos pasos

- Confirmar o descartar la sustracción de fondo morfológica con los resultados completos.
- Adoptar `clipLimit=4.0` en `03_histogram_equalization.py` está respaldado por los datos; falta
  decidir con el usuario si se aplica al script de producción.
- La normalización vía `equalizeHist` antes de Otsu queda descartada por los resultados.
