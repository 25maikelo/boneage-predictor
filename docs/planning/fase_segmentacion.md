# Fase: Preprocesamiento y Segmentación

> **Estado:** en curso, iniciada 2026-10-11. Etapa 1 completa, Etapa 2 pendiente de iniciar.

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

### Etapa 2: Rehacer la grilla de segmentación
Re-ejecutar la grilla (arquitecturas/pesos/augmentation ya justificados en
`models/hand-detector/experiments.md`, por consistencia con `ablacion_backbones.md`) con el
código corregido, agregando `spatial`/`cropped` como eje adicional. Evaluar cada configuración
por tres criterios: Dice/IoU por clase (`per_class_dice.py`), impacto en MAE downstream
(experimento de fusión de referencia), y Dice sobre un subconjunto mexicano anotado (~20–30
imágenes, retoma el Comentario #10). Selección por los tres criterios combinados, no solo Dice
en RSNA.

### Etapa 3: Preprocesamiento (02/03), justificado y ablacionado

| Script | Parámetro actual | Justificación actual | Alternativa a explorar |
|---|---|---|---|
| `02_frame_and_zoom.py` | Otsu + blur gaussiano 5×5 | Histograma bimodal (mano clara sobre fondo oscuro); asume iluminación uniforme | Umbral adaptativo (`adaptiveThreshold`) |
| `02_frame_and_zoom.py` | Kernel dilatación 9×9, 4 iteraciones | Cierra huecos del contorno antes de buscar el contorno externo; sin derivación documentada | Barrer tamaño (5×5–15×15) e iteraciones (2–6) |
| `02_frame_and_zoom.py` | Descarte si recorte <20% del área | Salvaguarda contra contorno mal detectado; umbral arbitrario | Medir tasa real de activación y auditar visualmente antes de ajustar |
| `03_histogram_equalization.py` | CLAHE `clipLimit=2.0`, `tileGridSize=8×8` | Defaults de OpenCV, no ajustados al dominio | Barrer `clipLimit` (1.0–4.0) y `tileGridSize` (4×4/8×8/16×16) |
| `03_histogram_equalization.py` | Mezcla 60% original + 40% CLAHE | Evita sobre-realce; proporción sin criterio medible | Barrer proporción de mezcla (0–100% CLAHE) |
| `03_histogram_equalization.py` | Ajuste final `alpha=0.9, beta=-10` | Corrección cosmética sin justificación registrada | Evaluar si sigue siendo necesaria tras optimizar la mezcla |

Cada variante se mide contra Dice/IoU del segmentador ganador de la Etapa 2, mismo subconjunto
de validación, nunca por inspección visual.

### Etapa 4: Síntesis y adopción
Experimentos de fusión end-to-end (mismo split/seed que experimentos previos) confirmando la
mejora acumulada en RSNA y MEX. Actualizar `config/segmentation.py`, scripts 02/03 y la
documentación del pipeline con la configuración ganadora y su justificación final.

## Verificación por etapa

| Etapa | Criterio |
|---|---|
| 1 | `metrics.json`: accuracy ya no colapsa a ~0.878 |
| 2 | Tabla comparativa (arquitectura × modo de recorte) con Dice/IoU por clase, MAE downstream, Dice en MEX |
| 3 | Dice/IoU antes/después de cada variación, mismo subconjunto de validación |
| 4 | Prueba pareada (`11_paired_validation.py`) config nueva vs. actual, en RSNA y MEX |

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
- Etapa 1 completa. Sigue la Etapa 2 (rehacer la grilla de segmentación con el código
  corregido, cruzada con el modo de recorte `spatial`/`cropped`).
