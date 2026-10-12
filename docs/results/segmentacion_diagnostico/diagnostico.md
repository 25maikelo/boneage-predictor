# Diagnóstico de fallos: segmentación en producción (`hand-detector_00`)

> Etapa 2 de [`fase_segmentacion.md`](../../planning/fase_segmentacion.md). Objetivo: caracterizar
> dónde y por qué falla el modelo de segmentación realmente usado en el pipeline, antes de decidir
> si vale la pena rehacer la grilla de arquitecturas o ablacionar el preprocesamiento.

## Metodología

Mismo split de validación que `per_class_dice.py` (76 de 379 imágenes anotadas,
`random_state=42`). Para cada imagen y clase se calculó Dice/IoU individual (no solo el
agregado), se construyó una matriz de confusión a nivel de píxel sobre las 76 imágenes, y se
generaron comparativas visuales (entrada, máscara real, predicción) para los 5 peores casos por
clase. Script: [`scripts/diagnostico_fallos.py`](scripts/diagnostico_fallos.py). Resultados
crudos en `results_json/`, figuras en `figures/`.

## Hallazgo 1: no hay confusión entre clases, solo pérdida hacia fondo

La matriz de confusión a nivel de píxel (76 imágenes) muestra confusión cruzada entre dedos
**prácticamente nula** (0.000 en todos los pares pinky/middle/thumb/wrist):

| Real \ Predicha | fondo | pinky | middle | thumb | wrist |
|---|:---:|:---:|:---:|:---:|:---:|
| pinky | 0.152 | **0.848** | 0.000 | 0.000 | 0.000 |
| middle | 0.117 | 0.000 | **0.883** | 0.000 | 0.000 |
| thumb | 0.131 | 0.000 | 0.000 | **0.869** | 0.000 |
| wrist | 0.147 | 0.000 | 0.000 | 0.000 | **0.852** |

El único error sistemático es que 12–15% de los píxeles de cada clase se predicen como fondo.
El modelo nunca confunde un dedo con otro: cuando falla, deja de detectar, no detecta mal.
Esto ya descarta una hipótesis (que el problema fuera "el modelo no distingue bien entre
regiones anatómicas"); el patrón real es erosión de borde hacia el fondo, más marcada en
estructuras delgadas (pinky, wrist tienen menor área, así que el mismo error de borde en
píxeles absolutos les cuesta más Dice proporcionalmente, consistente con el hallazgo previo de
Dice bajo en esas dos clases).

## Hallazgo 2: un subconjunto pequeño de imágenes con recorte/rotación defectuoso arrastra a
## varias clases a la vez

Al ordenar las 76 imágenes por Dice promedio, 3 casos (`1702`, `1609`, `1692`) se separan
claramente del resto (0.60–0.76 vs. ≥0.80 para las siguientes). Las tres aparecen como "peor
caso" en **dos o más clases simultáneamente**: no es que una clase sea mala en esa imagen, es
la imagen completa la que falla. Inspección visual de las tres:

- **`1702`** (`figures/peor_pinky_1_1702.png`): recorte suelto, se ve el borde de la placa
  radiográfica alrededor de la mano. El modelo predice dice=0.000 en pinky: la región
  desaparece, absorbida visualmente por la predicción de middle (su mancha verde se extiende
  sobre el área que debería ser roja).
- **`1609`** (`figures/peor_pinky_2_1609.png`): mismo patrón, borde de placa visible, mano
  rotada dentro del cuadro. Pinky se reduce a un punto minúsculo (dice=0.369) en vez de la
  franja alargada real.
- **`1692`** (`figures/peor_pinky_4_1692.png`): recorte con una esquina de la placa visible en
  la parte superior izquierda. El modelo **alucina una región azul (thumb) sobre esa esquina**,
  sin relación con ninguna anotación real, un falso positivo directamente atribuible al
  artefacto de encuadre, no a la anatomía.

Las otras imágenes revisadas (`10534`, `10552`, bien encuadradas) no muestran este patrón; ahí
el error es más leve y consistente con el hallazgo 1 (erosión de borde / fragmentación de la
región en piezas separadas, ej. el dedo medio partido en dos en `10534`), no recorte defectuoso.

## Conclusión: dos causas distintas, no una

1. **Recorte/rotación defectuoso de `02_frame_and_zoom.py`** explica los peores casos (≈4–7% de
   las imágenes de validación revisadas) y produce el error más severo: pérdida completa de
   clases delgadas o falsos positivos sobre artefactos de la placa. Esto **sí** justifica
   auditar `02_frame_and_zoom.py` (Etapa 4 del plan) con evidencia concreta, no solo porque "los
   parámetros nunca se validaron".
2. **Erosión de borde en estructuras delgadas** en imágenes bien encuadradas es un patrón más
   leve y extendido, consistente con que Dice penaliza más a regiones con peor relación
   perímetro/área (pinky, wrist). No hay evidencia de que un backbone distinto lo resuelva por
   sí solo: la matriz de confusión no muestra un problema de "la arquitectura no distingue las
   clases", sino de precisión de borde. Rehacer la grilla de arquitecturas (Etapa 3) no está
   claramente justificada por este diagnóstico; técnicas más dirigidas (loss sensible a bordes,
   post-procesamiento para quedarse con el componente conexo más grande por clase, o mayor
   resolución efectiva) atacan más directamente lo observado.

## Recomendación para las siguientes etapas

- **Priorizar Etapa 4** (auditoría/ablación de `02_frame_and_zoom.py`): cuantificar qué fracción
  del dataset completo (no solo las 76 de validación) tiene un recorte deficiente, usando el
  mismo tipo de heurística vista aquí (borde de placa visible / mano no centrada).
- **Degradar la prioridad de la Etapa 3** (grilla de arquitecturas) tal como estaba planteada:
  no hay evidencia de que el problema sea de arquitectura. Si se retoma, debería enfocarse en
  técnicas dirigidas al hallazgo 1 (loss con término de borde, post-procesamiento de componente
  conexo) en vez de un barrido genérico de backbones.
- Pendiente de decidir con el usuario antes de ejecutar cualquiera de las dos.
