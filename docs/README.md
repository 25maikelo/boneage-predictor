# Documentación: Bone Age Predictor

> Mapa de archivos, actualizado 2026-10-27. La documentación está organizada por **era** del
> proyecto donde aplica: `maestria/` (cerrada, trabajo histórico de referencia) vs. `doctorado/`
> (activa). Las carpetas sin esa distinción (`data/`, `arquitectura/`, `scripts/`,
> `planning/pipeline.md`) son recursos compartidos que usan ambas eras. El estado de cada
> documento se indica explícitamente (vigente, histórico/completado, o superado) para que no
> haya que adivinar cuál es la fuente de verdad actual.

## planning/: Flujo del proyecto y próximos pasos

| Archivo | Contenido | Estado |
|---|---|---|
| [doctorado/](planning/doctorado/README.md) | Mapa de más alto nivel del doctorado, estilo Kiro (`requirements.md`/`design.md`/`tasks.md`/`referencias.md`): objetivos A–K, su diseño y la siguiente tarea concreta. **Leer primero** | **Vigente: punto de entrada** |
| [doctorado/fase_segmentacion.md](planning/doctorado/fase_segmentacion.md) | Plan y bitácora técnica de los objetivos C/D1 del doctorado: análisis sistemático de preprocesamiento y segmentación (bugs encontrados, grilla a rehacer, ablación de parámetros) | **En curso** |
| [pipeline.md](planning/pipeline.md) | Descripción de cada script (00–11), entradas/salidas, tiempos y comandos SLURM. Recurso compartido, no es específico de una era | Vigente |
| [maestria/plan_exploracion.md](planning/maestria/plan_exploracion.md) | Criterios de priorización para las ablaciones de género/LR/tamaño de imagen/datos clínicos | Histórico (completado, ver `results/maestria/2026-06-18_optimizacion.md`) |

---

## arquitectura/: Arquitecturas y decisiones de diseño

> Antes `design/`; renombrada para no confundirse con `planning/doctorado/design.md` (el
> documento de diseño del sistema, estilo Kiro). Este recurso es compartido: describe las
> arquitecturas del pipeline que tanto la maestría como el doctorado usan/extienden.

| Archivo | Contenido | Estado |
|---|---|---|
| [arquitecturas.md](arquitectura/arquitecturas.md) | Diagramas y parámetros de los 5 modos: `backbone`, `simple_cnn`, `backbone_vectors`, `unified_cnn`, `whole_hand` | Vigente |
| [propuesta_densenet_vector_fusion.md](arquitectura/propuesta_densenet_vector_fusion.md) | Propuesta original de `backbone_vectors` | Histórico (contexto, ya implementado en exps 35/38/41/43) |

---

## data/: Dataset

> Recurso compartido: describe el dataset RSNA+MEX ya en uso por ambas eras. Cuando el objetivo
> F del doctorado (base de datos mexicana ampliada) tenga datos, su reporte va aquí también.

| Archivo | Contenido | Estado |
|---|---|---|
| [dataset_report.md](data/dataset_report.md) | Estadísticas de procesamiento: imágenes raw → cropped → equalized → segmented, splits, distribución de edades | Vigente |

---

## results/: Resultados y evaluación

### doctorado/ (evidencia activa de los objetivos C/D)

| Archivo | Contenido | Estado |
|---|---|---|
| [segmentacion_diagnostico/](results/doctorado/segmentacion_diagnostico/diagnostico.md) | Diagnóstico de fallos de `hand-detector_00`: matriz de confusión por píxel, peores casos por clase, causas identificadas | Vigente |
| [preprocesamiento_diagnostico/](results/doctorado/preprocesamiento_diagnostico/diagnostico.md) | Auditoría de `02_frame_and_zoom.py`/`03_histogram_equalization.py` con métricas propias (fill ratio, contraste, saturación), sin depender de ningún modelo. Incluye `reporte_resumen_preprocesamiento.pdf` para presentar a asesores | **Vigente: más reciente** |

### maestria/ (cerrado, resultados de referencia/baseline)

| Archivo | Contenido | Estado |
|---|---|---|
| [2026-07-22_experiments_master.md](results/maestria/2026-07-22_experiments_master.md) | Tabla maestra de todos los experimentos de la maestría: configuración, estado, línea de tiempo de código | **Vigente: fuente de verdad del historial de experimentos de la maestría, y baseline candidato para el Requirement 5 del doctorado** |
| [2026-06-18_optimizacion.md](results/maestria/2026-06-18_optimizacion.md) | Fase 6 (exps 47–56): ablaciones de género, LR/épocas, tamaño de imagen. Mejor de esta fase: Exp 55 (224×224); el mejor global de la maestría es Exp 57 (Fase 8, ver `2026-07-22_experiments_master.md`) | Histórico |
| [2026-05-28_resultados.md](results/maestria/2026-05-28_resultados.md) | Tabla resumen con podio y hallazgos de los primeros 14 experimentos | Histórico |
| [2026-05-28_resumen_evaluacion.md](results/maestria/2026-05-28_resumen_evaluacion.md) | Resultados detallados por experimento: ranking global, análisis por segmento y rango de edad | Histórico |
| [ablacion_backbones/](results/maestria/ablacion_backbones/2026-07-22_ablacion_backbones.md) | Comparación formal de los 4 backbones (ResNet50/VGG16/DenseNet121/InceptionV3) con pruebas pareadas | Histórico |
| [experimentos_adicionales/](results/maestria/experimentos_adicionales/analisis.md) | Experimentos y respuestas a la segunda ronda de revisión del artículo publicado (*Technologies*, MDPI): baseline whole-hand, Dice/IoU por clase, benchmark de latencia, significancia estadística | Histórico, registro cerrado |

---

## scripts/: Documentación de scripts individuales

Los 12 scripts del pipeline (`src/00`–`src/11`), cada uno con descripción, uso, entradas/salidas.
Recurso compartido, no específico de una era:

| Script | Documento |
|---|---|
| `00_download_dataset.py` | [00_download_dataset.md](scripts/00_download_dataset.md) |
| `01_train_hand_detector.py` | [01_train_hand_detector.md](scripts/01_train_hand_detector.md) |
| `02_frame_and_zoom.py` | [02_frame_and_zoom.md](scripts/02_frame_and_zoom.md) |
| `03_histogram_equalization.py` | [03_histogram_equalization.md](scripts/03_histogram_equalization.md) |
| `04_segment_images.py` | [04_segment_images.md](scripts/04_segment_images.md) |
| `05_dataset_analysis.py` | [05_dataset_analysis.md](scripts/05_dataset_analysis.md) |
| `06_training.py` | [06_training.md](scripts/06_training.md) |
| `07_validation.py` | [07_validation.md](scripts/07_validation.md) |
| `08_mex_validation.py` | [08_mex_validation.md](scripts/08_mex_validation.md) |
| `09_performance_analysis.py` | [09_performance_analysis.md](scripts/09_performance_analysis.md) |
| `10_age_range_analysis.py` | [10_age_range_analysis.md](scripts/10_age_range_analysis.md) |
| `11_paired_validation.py` | [11_paired_validation.md](scripts/11_paired_validation.md) |
