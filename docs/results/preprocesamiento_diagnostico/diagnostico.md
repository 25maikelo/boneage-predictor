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

## Hallazgo 4: bug de corrimiento de píxeles en la rotación, causa independiente de fallos

El recorte rota la imagen completa alrededor del **centro de la imagen** (`cv2.getRotationMatrix2D`
con `(w//2, h//2)`), no del centro de la mano detectada, y mantiene el **mismo tamaño de canvas**
en `cv2.warpAffine(img, M, (w, h))`. Cuando la mano está lejos del centro de la imagen, rotarla
alrededor de un punto que no es el suyo puede sacar parte de su contenido fuera del cuadro fijo,
contenido que `warpAffine` descarta silenciosamente (no hay forma de recuperarlo de `rotated`
después).

Esto produce coordenadas de recorte **negativas** en una fracción medible de las imágenes
(21/600 en una muestra, 3.5%). En Python/NumPy, un índice negativo en un slice no se trata como
"0": se interpreta como "contar desde el final del arreglo", así que
`rotated[y0:y1, x0:x1]` con `y0` negativo produce un recorte **equivocado** (una tira angosta de
una zona arbitraria de la imagen rotada) en vez de un error visible. Ejemplo verificado
(`1748.png`): el recorte real queda en **(264, 2040)** píxeles, una tira sin nada de la mano,
visible en [`figures/bug_corrimiento_1748.png`](figures/bug_corrimiento_1748.png).

**Por qué no se había detectado antes**: la salvaguarda existente (`<20%` del área) sí atrapa
estos 21 casos verificados y cae de vuelta a la imagen completa sin recortar, no hay corrupción
silenciosa confirmada hasta ahora, pero caen en la misma categoría "descarte" que el hallazgo 2,
con una causa completamente distinta (geometría de la rotación, no contraste). Las imágenes
afectadas tienen buen contraste; el problema es puramente que la mano está lejos del centro.

**Fix validado**: expandir el canvas antes de rotar (técnica estándar de "rotar sin recortar
esquinas": calcular el tamaño mínimo que contiene la imagen rotada completa, ajustar la matriz de
traslación, rotar hacia ese canvas más grande, y solo entonces calcular el recorte). Sobre una
muestra de 1,500 imágenes completas (no solo los casos conocidos):

| Métrica | Actual | Canvas expandido |
|---|---:|---:|
| Estado `discard_area` | 84/1500 (5.6%) | 31/1500 (2.1%) |
| Estado `ok` | 1416/1500 | 1469/1500 |

**63% menos descartes**, recuperando ~53 de cada 1,500 imágenes (≈3.5%, consistente con la
muestra de 600). El histograma de fill ratio del cuerpo principal de la distribución (0.6–0.85)
es prácticamente idéntico entre ambos métodos, el fix no degrada los casos que ya funcionaban,
solo repara los que fallaban por esta causa geométrica
([`figures/validacion_canvas_expandido.png`](figures/validacion_canvas_expandido.png)).

## Hallazgo 5: rotación espuria de 90° en casos ya alineados

Detectado al revisar visualmente los casos del hallazgo 4: algunas imágenes con el fix de canvas
expandido aplicado quedaban con la mano rotada de lado (horizontal) en vez de mantener su
orientación original (vertical, dedos hacia arriba), pese a que el original ya estaba
razonablemente alineado.

Causa: `cv2.minAreaRect` devuelve `angle=90.0` como valor de frontera cuando el rectángulo
detectado ya está alineado a los ejes. La lógica de corrección existente
(`if w_rect < h_rect: angle += 90`) interpreta ese caso límite como "necesita rotar 90°", cuando
en realidad significa "ya está alineado, no rotar". Confirmado en **2.5–3.5% de una muestra de
600–800 imágenes** (ángulo crudo en el rango [89°, 91°] combinado con `w_rect < h_rect`).

**Fix validado**: no aplicar la corrección `+90` cuando el ángulo crudo ya está en ese rango de
frontera (tolerancia de 1°). Verificado visualmente en los 4 casos del hallazgo 4: con el fix,
las 4 manos quedan en su orientación original en vez de rotar 90°
([`figures/comparacion_angulo.png`](figures/comparacion_angulo.png)). Sobre una muestra de 800
imágenes, solo cambia el ángulo final del 3.0% de los casos, todos dentro del rango de frontera
identificado, sin afectar la lógica de corrección para inclinaciones reales.

**Aplicado a producción** en `02_frame_and_zoom.py`, junto con el fix de canvas expandido
(hallazgo 4). Con ambos fixes activos, una muestra de 300 imágenes da 97.7% `ok` / 2.3%
`discard_area`, consistente con que el único problema restante es el de contraste (hallazgo 2,
sin fix encontrado).

## Conclusiones

1. El recorte de `02_frame_and_zoom.py` tenía una cola de fallos/errores (~5% del dataset) con
   **tres causas independientes**: bajo contraste global que rompe Otsu (hallazgo 2, sin fix
   encontrado todavía), pérdida de contenido al rotar manos lejos del centro (hallazgo 4, **fix
   aplicado**), y rotación espuria de 90° en casos ya alineados (hallazgo 5, **fix aplicado**).
2. Los parámetros actuales de CLAHE (`clipLimit=2.0`) no eran óptimos frente a las alternativas
   probadas: `clipLimit=4.0` preserva más contraste y satura menos píxeles (**aplicado**).
3. Los fixes de los hallazgos 4 y 5 son independientes de CLAHE y del problema de contraste;
   atacan causas geométricas distintas, con evidencia cuantitativa y visual clara y sin
   regresiones detectadas.

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
específico, mejor fundamentado que una ecualización genérica.

**Primer intento, kernel fijo de 51px, descartado**: sobre el dataset completo
(`results_json/validacion_background_subtraction.json`), la tasa de descarte se disparó de 642 a
**6,413 de 12,811 imágenes** (el kernel de 51px es minúsculo frente al tamaño real de las
imágenes raw, ~900–2200px por lado, muy por debajo de lo que recomienda la literatura: "al menos
tan grande como el objeto de interés más grande"). El fill ratio promedio también empeoró
(0.735 → 0.725). Sin embargo, para el subconjunto específico de los 583 peores casos actuales
(fill ratio <0.3), el fill ratio promedio **más que se duplicó** (0.139 → 0.303) y 52.3% de ellos
mejoraron, mejor que `equalizeHist` (48.5%) y con menos imágenes dañadas en general (8.7% vs.
18.2%). Esto sugiere que la técnica tiene mérito pero el kernel está mal calibrado, no que el
método en sí no sirva.

**Segundo intento, kernel proporcional al tamaño de imagen**: `kernel = min(alto, ancho) /
divisor` en vez de un tamaño fijo, sobre una muestra de 405 imágenes (400 al azar + los 5 peores
casos conocidos). Mejora la tasa de descarte frente al kernel fijo (10.1–11.6% según el divisor,
vs. el 50% catastrófico de antes), pero sigue siendo **peor que el 5% actual** en esta muestra.

**Verificación dirigida a los 5 peores casos conocidos (`2615`, `3172`, `2176`, `3016`, `2721`)**:
con el mejor divisor (3, kernel≈495px), **ninguno de los 5 mejora**: mismo estado
(`discard_area`) y mismo fill ratio hasta la tercera decimal que con Otsu directo. La mejora
agregada del 52.3% en el conjunto de 583 "peores casos" (kernel fijo) no viene de estos casos
extremos, sino de casos menos severos dentro de ese grupo. Estas 5 imágenes en particular son
tan uniformemente oscuras/bajo contraste que ninguna normalización de contraste previa a Otsu
(ni `equalizeHist` ni sustracción de fondo, con ningún tamaño de kernel probado) les da
suficiente separación bimodal para que Otsu encuentre la mano.

**Conclusión de esta línea de trabajo**: se descartan ambas alternativas a Otsu probadas como
reemplazo general. Ninguna resuelve de forma barata el núcleo duro de casos extremos. Siguiendo
el precedente de BoneXpert ([`revision_literatura.md`](revision_literatura.md), sección 4): en
vez de seguir buscando un fix automático para ~2–5% del dataset, lo más defendible es **detectar
y registrar explícitamente** estos casos (hoy `02_frame_and_zoom.py` cae a la imagen completa en
silencio, sin dejar rastro) para poder decidir caso por caso, en vez de asumir que el pipeline
debe producir un recorte automático para el 100% de las imágenes.

## Próximos pasos

- ~~Adoptar `clipLimit=4.0`~~ **hecho**: aplicado en `03_histogram_equalization.py`.
- ~~Agregar logging explícito de las salvaguardas~~ **hecho**: `02_frame_and_zoom.py` ahora
  guarda `data/images/crop_fallbacks.json` con el detalle por imagen.
- `equalizeHist` y sustracción de fondo morfológica (fija o proporcional) quedan **descartadas**
  como reemplazo de Otsu: ninguna resuelve los casos extremos y ambas introducen regresiones en
  otras imágenes.
- ~~Aplicar el fix de canvas expandido~~ **hecho**: aplicado en `02_frame_and_zoom.py`.
- ~~Aplicar el fix de rotación espuria de 90°~~ **hecho**: aplicado en `02_frame_and_zoom.py`.
- **Pendiente**: re-correr `02_frame_and_zoom.py` y `03_histogram_equalization.py` sobre el
  dataset completo (12,813 imágenes) para que `data/images/cropped/`/`equalized/` reflejen los
  tres fixes ya aplicados; el código está listo pero los datos en disco siguen siendo los
  generados con el código anterior hasta que se regeneren.
