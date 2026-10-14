# Fase: Preprocesamiento y Segmentación

> **Estado:** en curso, iniciada 2026-10-11. Etapas 1, 2 y 4 completas. Etapa 3 (grilla de
> arquitecturas) en pausa, sin justificación clara tras el diagnóstico de la Etapa 2.
>
> **Corrección de rumbo (2026-10-11):** el orden original saltaba directo a rehacer la grilla de
> arquitecturas sin antes diagnosticar qué falla realmente en el modelo que está en producción
> (`hand-detector_00`). Se reordena: primero diagnóstico de fallos sobre el modelo actual, y solo
> si el diagnóstico apunta a un problema de arquitectura/preprocesamiento se decide qué ablación
> vale la pena correr. El modelo en producción (`hand-detector_00`, sin augmentation) nunca
> estuvo afectado por el bug del hallazgo #2: eso solo contaminó corridas de la grilla que
> nadie adoptó.

## Contexto

Manuscrito publicado, ciclo de revisión cerrado. Nueva fase: análisis sistemático de
preprocesamiento (`src/preprocessing/02_frame_and_zoom.py`, `03_histogram_equalization.py`) y
segmentación (`01_train_hand_detector.py`, `04_segment_images.py`), la parte menos auditada del
pipeline. Objetivo: rehacer la base experimental de segmentación con metodología correcta y
explorar el preprocesamiento de forma justificada y reproducible, no con mejoras puntuales.

## Hallazgos del diagnóstico inicial

1. Ya existe una ablación de 34 configuraciones de segmentación (`models/hand-detector/00`–`33`),
   **contaminada por un bug** (#2) en la mitad de las corridas; nunca se cruzó con el modo de
   recorte (`spatial`/`cropped`).
2. **Bug confirmado**: `DATA_AUGMENTATION=True` rompe el entrenamiento (colapsa a predecir solo
   fondo) en toda arquitectura probada. Causa: dos `ImageDataGenerator.flow()` independientes
   (imagen/máscara) con el mismo seed, pero `AUG_BRIGHTNESS_RANGE` solo aplica a la imagen,
   desincronizando las transformaciones geométricas subsecuentes. Confirmado con repro aislado
   (dos generadores con parámetros idénticos sí sincronizan, diff=0.0).
3. `ENCODER_WEIGHTS=None` también colapsa en arquitecturas con backbone, incluso sin
   augmentation (runs 04/06). Diagnóstico refinado tras inspeccionar las curvas por época:
   **no es un bug de código**, son dos causas distintas de un mismo síntoma.
   `trainable=False` (run 06) congela un backbone con pesos aleatorios desde la época 1: el
   decoder nunca recibe features útiles y el loss converge suavemente al óptimo trivial
   ("predecir solo fondo"), comportamiento correcto para una configuración sin sentido
   (nadie debería congelar un backbone sin preentrenar). `trainable=True` (run 04) sí aprende
   en el set de entrenamiento (loss de train baja de forma monótona, 1.66→0.19), pero el
   accuracy de validación oscila violentamente entre épocas (0.878 → 0.106 → 0.848 → 0.164):
   inestabilidad de optimización al entrenar backbone+BatchNorm desde cero con el
   LR/presupuesto de épocas actual, no un defecto de sincronización como el hallazgo #2.
   `EarlyStopping`/`restore_best_weights` termina quedándose con una época mala por azar.
   Baja prioridad (nadie usa `ENCODER_WEIGHTS=None` con backbone en la práctica); sin fix de
   código, documentado como limitación conocida.
4. Clases más débiles del mejor run no contaminado: meñique (dice=0.778) y muñeca (dice=0.769)
   vs. medio (0.890) y pulgar (0.864).
5. El modo `cropped` parece superar a `spatial` en el resultado final (Exp 57 vs. 55: RSNA
   10.7m vs 13.4m, MEX 13.5m vs 15.0m), pero solo 2 pares confundidos con otras configuraciones,
   nunca aislado como variable deliberada.
6. Preprocesamiento (02/03) usa parámetros fijos sin justificación registrada ni ablación.
7. Segmentación nunca validada sobre imágenes mexicanas (Comentario #10 del revisor, cerrado sin
   hacerse por restricción de tiempo).

## Principios de reproducibilidad

- Semilla fija (`random_state=42`, ya usado en todo el pipeline) en cualquier split, muestreo o
  inicialización aleatoria.
- Cada corrida en su propio directorio versionado con `config.json`, igual que ya hace
  `models/hand-detector/hand-detector_NN/`.
- Comando exacto de SLURM documentado por experimento, en el documento de resultados
  correspondiente.
- Datasets de comparación pineados: mismas imágenes de entrada para todas las variantes de una
  misma comparación.
- Métricas crudas (`metrics.json`/`plot_data.json`) como fuente de verdad; las tablas
  comparativas se derivan de ahí, nunca se transcriben a mano.
- Mismo entorno conda `boneage_gpu` en todas las corridas.

## Plan de trabajo

Etapas secuenciales: cada una es condición de la siguiente (no tiene sentido comparar
arquitecturas con un augmentation roto, ni elegir modo de recorte sin cruzarlo con el resto de
la grilla, ni ablacionar preprocesamiento sin un segmentador de referencia confiable).

### Etapa 1: Corregir defectos de entrenamiento del segmentador
Arreglar el bug de sincronización imagen/máscara (#2) y el colapso de `ENCODER_WEIGHTS=None`
(#3) en `01_train_hand_detector.py`. Sin esto, ninguna comparación de la Etapa 2 es válida.

**Verificación:** re-entrenar 2–3 configuraciones previamente colapsadas (mismo `config.json`,
mismo split `random_state=42`) y confirmar en `evaluation/metrics.json` que el accuracy ya no se
estanca en ~0.878.

### Etapa 2: Diagnóstico de fallos del modelo en producción (`hand-detector_00`)
Antes de ablacionar nada, caracterizar dónde y por qué falla el modelo que realmente se usa.
- Extender `per_class_dice.py` (ya calcula Dice/IoU por imagen, pero solo guarda el agregado)
  para conservar el Dice por imagen y por clase, con el ID de cada imagen.
- Identificar las imágenes con peor Dice, en particular para meñique y muñeca (hallazgo #4).
- Generar comparativas visuales (imagen de entrada, máscara real, máscara predicha) para los
  peores casos de cada clase.
- Clasificar el tipo de fallo observado: ¿imprecisión de borde, confusión entre clases
  adyacentes, región completamente perdida, correlación con pose/ángulo de la mano o calidad de
  imagen? Esto determina si el problema es de arquitectura (→ Etapa 3), de preprocesamiento
  (→ Etapa 4), de la anotación misma, o ninguno de los anteriores.
- Dataset: las mismas 76 imágenes de validación (`random_state=42`) usadas por
  `per_class_dice.py`, más el subconjunto mexicano (~20–30 imágenes, retoma el Comentario #10)
  para ver si el patrón de fallo es el mismo o distinto fuera de RSNA.

**Decisión de alcance:** las Etapas 3 y 4 solo se ejecutan, y solo en la medida en que el
diagnóstico de esta etapa las justifique. Si los fallos son de borde/anotación y no de
arquitectura, rehacer la grilla completa no tendría sentido.

### Etapa 3 (condicional): Rehacer la grilla de segmentación
Solo si el diagnóstico de la Etapa 2 apunta a una limitación de arquitectura. Re-ejecutar la
grilla (arquitecturas/pesos/augmentation ya justificados en
`models/hand-detector/experiments.md`, por consistencia con `ablacion_backbones.md`) con el
código corregido, agregando `spatial`/`cropped` como eje adicional. Evaluar cada configuración
por tres criterios: Dice/IoU por clase, impacto en MAE downstream (experimento de fusión de
referencia), y Dice sobre el subconjunto mexicano. Selección por los tres criterios combinados,
no solo Dice en RSNA.

### Etapa 4: Preprocesamiento (02/03), justificado y ablacionado

**Alcance explícito:** esta etapa evalúa `02_frame_and_zoom.py`/`03_histogram_equalization.py`
con métricas propias de la imagen (geometría del recorte, contraste, entropía, saturación), sin
depender del segmentador ni de ningún modelo entrenado. Es deliberadamente independiente de las
Etapas 2/3.

| Script | Parámetro actual | Justificación actual | Alternativa a explorar |
|---|---|---|---|
| `02_frame_and_zoom.py` | Otsu + blur gaussiano 5×5 | Histograma bimodal (mano clara sobre fondo oscuro); asume iluminación uniforme | Umbral adaptativo (`adaptiveThreshold`) |
| `02_frame_and_zoom.py` | Kernel dilatación 9×9, 4 iteraciones | Cierra huecos del contorno antes de buscar el contorno externo; sin derivación documentada | Barrer tamaño (5×5–15×15) e iteraciones (2–6) |
| `02_frame_and_zoom.py` | Descarte si recorte <20% del área | Salvaguarda contra contorno mal detectado; umbral arbitrario | Medir tasa real de activación y auditar visualmente antes de ajustar |
| `03_histogram_equalization.py` | CLAHE `clipLimit=2.0`, `tileGridSize=8×8` | Defaults de OpenCV, no ajustados al dominio | Barrer `clipLimit` (1.0–4.0) y `tileGridSize` (4×4/8×8/16×16) |
| `03_histogram_equalization.py` | Mezcla 60% original + 40% CLAHE | Evita sobre-realce; proporción sin criterio medible | Barrer proporción de mezcla (0–100% CLAHE) |
| `03_histogram_equalization.py` | Ajuste final `alpha=0.9, beta=-10` | Corrección cosmética sin justificación registrada | Evaluar si sigue siendo necesaria tras optimizar la mezcla |

Cada variante se mide con métricas propias de la imagen (fill ratio post-recorte, contraste,
entropía, tasa de saturación), nunca por inspección visual ni contra un modelo de segmentación.

### Etapa 5: Síntesis y adopción
Experimentos de fusión end-to-end (mismo split/seed que experimentos previos) confirmando la
mejora acumulada en RSNA y MEX. Actualizar `config/segmentation.py`, scripts 02/03 y la
documentación del pipeline con la configuración ganadora y su justificación final.

## Verificación por etapa

| Etapa | Criterio |
|---|---|
| 1 | `metrics.json`: accuracy ya no colapsa a ~0.878 |
| 2 | Reporte de fallos por imagen/clase con evidencia visual, y una conclusión explícita de si el problema es de arquitectura, preprocesamiento, anotación o ninguno |
| 3 | Tabla comparativa (arquitectura × modo de recorte) con Dice/IoU por clase, MAE downstream, Dice en MEX |
| 4 | Fill ratio/contraste/entropía/saturación antes y después de cada variación, mismo subconjunto de imágenes |
| 5 | Prueba pareada (`11_paired_validation.py`) config nueva vs. actual, en RSNA y MEX |

## Bitácora de avance

- **2026-10-11:** plan aprobado. Inicia Etapa 1.
- **2026-10-11:** corregido el bug de augmentation (hallazgo #2) en
  `01_train_hand_detector.py`: generador único sobre imagen+máscara combinadas en vez de dos
  generadores independientes. **Verificado con datos reales**: re-entrenando la config exacta
  del run colapsado 01 (`unet_mobilenetv2`, `imagenet`, `trainable=False`, `aug=True`,
  `brightness=(0.8,1.2)`) con el código corregido (`hand-detector_34`), el resultado pasa de
  dice=0.1867/iou=0.1751/acc=0.8778 (colapsado) a **dice=0.9073/iou=0.8397/acc=0.9764**. Fix
  confirmado, Etapa 1 cerrada para el hallazgo #2.
- Hallazgo #3 (`ENCODER_WEIGHTS=None`) diagnosticado sin aplicar fix de código (ver detalle en
  la sección de hallazgos): baja prioridad, no bloquea el resto del plan.
- Etapa 1 completa.
- **Corrección de rumbo:** se reordena el plan (ver nota al inicio). Antes de rehacer la grilla
  de arquitecturas, se diagnostican los fallos reales del modelo en producción
  (`hand-detector_00`, nunca afectado por el bug del hallazgo #2). Rehacer la grilla queda
  condicional a lo que ese diagnóstico encuentre.
- Inicia Etapa 2 (diagnóstico de fallos de `hand-detector_00`).
- **Etapa 2 completa.** Diagnóstico en
  [`docs/results/segmentacion_diagnostico/diagnostico.md`](../results/segmentacion_diagnostico/diagnostico.md).
  Dos causas distintas, no una: (a) un subconjunto pequeño de imágenes (≈4–7% de las 76 de
  validación) con recorte/rotación defectuoso de `02_frame_and_zoom.py` (borde de placa
  visible) causa los fallos más severos, incluyendo un falso positivo alucinado sobre un
  artefacto de encuadre; (b) erosión de borde en estructuras delgadas (pinky, wrist) en
  imágenes bien encuadradas, sin evidencia de confusión entre clases (matriz de confusión por
  píxel: ~0% de confusión cruzada entre dedos, el único error es perder píxeles hacia fondo).
  **Esto cambia la prioridad de las etapas condicionales**: la Etapa 4 (preprocesamiento) queda
  priorizada con evidencia directa; la Etapa 3 (grilla de arquitecturas) pierde justificación
  tal como estaba planteada, porque el diagnóstico no apunta a un problema de arquitectura.
- **Corrección de alcance:** la Etapa 4 se evalúa exclusivamente con métricas propias de la
  imagen (fill ratio, contraste, entropía, saturación), sin ninguna dependencia del segmentador
  ni de Dice/IoU. El preprocesamiento (02/03) y la segmentación (01/04) se tratan como líneas de
  trabajo separadas; no se retoma la idea de reentrenar el segmentador en esta etapa.
- **Etapa 4 completa.** Diagnóstico en
  [`docs/results/preprocesamiento_diagnostico/diagnostico.md`](../results/preprocesamiento_diagnostico/diagnostico.md).
  Hallazgos: (a) el recorte deja en promedio 27% de fondo/borde residual (fill ratio medio
  0.730), con una cola de ~2–5% de imágenes donde el recorte falla casi por completo; (b) esos
  peores casos no son de rotación compleja, son de **contraste global bajo** que rompe el
  umbral de Otsu, lo cual importa porque el recorte (02) corre antes de la ecualización (03) en
  el pipeline; (c) los parámetros actuales de CLAHE (`clipLimit=2.0`) no son los mejores de los
  probados: reducen el contraste respecto a no aplicar CLAHE y aumentan la saturación más que
  `clipLimit=4.0`. Próximos pasos propuestos (sin ejecutar todavía): probar `clipLimit=4.0` como
  nuevo default, y evaluar normalizar contraste antes del umbral de Otsu en el recorte.
- **Revisión de literatura** sobre mejora de calidad radiográfica en
  [`docs/results/preprocesamiento_diagnostico/revision_literatura.md`](../results/preprocesamiento_diagnostico/revision_literatura.md).
  Identificó sustracción de fondo morfológica como alternativa a Otsu mejor fundamentada que
  normalizar contraste genérico, y el precedente de BoneXpert (rechazar imágenes de mala calidad
  en vez de forzarlas, tasa real de rechazo 2.3%, mismo orden de magnitud que la nuestra).
- **Validación de las tres propuestas sobre el dataset completo (12,811 imágenes), cerrada**:
  `clipLimit=4.0` se confirma (85.4% de las imágenes con más contraste) y queda recomendado para
  adoptar. `equalizeHist` antes de Otsu y sustracción de fondo morfológica (kernel fijo y
  proporcional) quedan **descartadas** como reemplazo de Otsu: ninguna resuelve los 5 casos más
  extremos verificados uno por uno (imágenes tan uniformemente oscuras que ninguna normalización
  de contraste previa les da separación bimodal suficiente), y ambas dañan otras imágenes en el
  proceso. Recomendación final: adoptar `clipLimit=4.0`, y agregar logging explícito de las
  imágenes que activan las salvaguardas de `02_frame_and_zoom.py` (hoy es silencioso) en vez de
  seguir buscando un reemplazo automático de Otsu.
- **`clipLimit=4.0` y el logging de salvaguardas ya se aplicaron a producción.**
- **Nuevo hallazgo (hallazgo 4 de
  [`preprocesamiento_diagnostico/diagnostico.md`](../results/preprocesamiento_diagnostico/diagnostico.md))**:
  bug de corrimiento de píxeles en la rotación de `02_frame_and_zoom.py` (rota alrededor del
  centro de la imagen, no de la mano, con el mismo tamaño de canvas; produce coordenadas de
  recorte negativas y pierde contenido). Causa independiente del bajo contraste (hallazgo 2):
  afecta imágenes bien contrastadas donde la mano está lejos del centro. Fix de "canvas
  expandido" validado sobre 1,500 imágenes: reduce la tasa de `discard_area` de 5.6% a 2.1%
  (63% menos descartes), sin regresiones detectadas en el resto de la distribución. **Aplicado a
  producción.**
- **Hallazgo 5**: al revisar visualmente el hallazgo 4, se detectó que algunos casos quedaban
  con la mano rotada de lado pese a estar bien alineados en el original. Causa: `cv2.minAreaRect`
  devuelve `angle=90` como valor de frontera para rectángulos ya alineados; la lógica de
  corrección lo interpretaba como "rotar 90°". Confirmado en 2.5–3.5% de muestras de 600–800
  imágenes, verificado visualmente en los 4 casos conocidos. **Aplicado a producción.**
- Reporte PDF generado y actualizado con ambos fixes:
  [`docs/results/preprocesamiento_diagnostico/reporte_corrimiento_pixeles.pdf`](../results/preprocesamiento_diagnostico/reporte_corrimiento_pixeles.pdf).
- **Revisión adicional a pedido del usuario** ("revisa si hay más cosas que mejorar"): se
  investigaron casos con fill ratio ≈0 que pasaban como `ok` sin activar ninguna salvaguarda;
  resultó ser una limitación de la métrica de diagnóstico (re-umbralizar con Otsu un recorte ya
  ajustado y de bajo contraste), no un bug del recorte en sí. Se revisó `04_segment_images.py`
  con el mismo criterio que `02_frame_and_zoom.py` (recorte geométrico al bounding box del
  segmento): sin bugs equivalentes, los límites ya estaban correctamente acotados.
- **Hallazgo 6** (en `03_histogram_equalization.py`): `cv2.convertScaleAbs` usa valor absoluto en
  vez de saturar, reflejando hacia arriba los tonos más oscuros (el fondo de la radiografía) en
  vez de dejarlos en 0. Verificado en una imagen real; impacto sutil pero real (~3.3% de píxeles
  difieren >3 niveles de gris). **Aplicado a producción.**
- **Pendiente**: re-correr `02_frame_and_zoom.py` y `03_histogram_equalization.py` sobre el
  dataset completo para que los datos en disco reflejen los cuatro fixes ya aplicados en código.
