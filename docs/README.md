# Documentación: Bone Age Predictor

> Mapa de archivos, actualizado 2026-10-05. El estado de cada documento se indica explícitamente
> (vigente, histórico/completado, o superado) para que no haya que adivinar cuál es la fuente de
> verdad actual.

## planning/: Flujo del proyecto y próximos pasos

| Archivo | Contenido | Estado |
|---|---|---|
| [pipeline.md](planning/pipeline.md) | Descripción de cada script (00–10), entradas/salidas, tiempos y comandos SLURM | Vigente |
| [plan_exploracion.md](planning/plan_exploracion.md) | Criterios de priorización para las ablaciones de género/LR/tamaño de imagen/datos clínicos | Histórico (completado, ver `results/2026-06-18_optimizacion.md`) |

---

## design/: Arquitecturas y decisiones de diseño

| Archivo | Contenido | Estado |
|---|---|---|
| [arquitecturas.md](design/arquitecturas.md) | Diagramas y parámetros de los 5 modos: `backbone`, `simple_cnn`, `backbone_vectors`, `unified_cnn`, `whole_hand` | Vigente |
| [propuesta_densenet_vector_fusion.md](design/propuesta_densenet_vector_fusion.md) | Propuesta original de `backbone_vectors` | Histórico (contexto, ya implementado en exps 35/38/41/43) |

---

## data/: Dataset

| Archivo | Contenido | Estado |
|---|---|---|
| [dataset_report.md](data/dataset_report.md) | Estadísticas de procesamiento: imágenes raw → cropped → equalized → segmented, splits, distribución de edades | Vigente |

---

## results/: Resultados y evaluación

| Archivo | Contenido | Estado |
|---|---|---|
| [2026-07-22_experiments_master.md](results/2026-07-22_experiments_master.md) | Tabla maestra de todos los experimentos: configuración, estado, línea de tiempo de código | **Vigente: fuente de verdad para el historial de experimentos** |
| [2026-06-18_optimizacion.md](results/2026-06-18_optimizacion.md) | Fase 6 (exps 47–56): ablaciones de género, LR/épocas, tamaño de imagen. Mejor de esta fase: Exp 55 (224×224); el mejor global del proyecto es Exp 57 (Fase 8, ver `2026-07-22_experiments_master.md`) | Vigente |
| [2026-05-28_resultados.md](results/2026-05-28_resultados.md) | Tabla resumen con podio y hallazgos de los primeros 14 experimentos | Vigente |
| [2026-05-28_resumen_evaluacion.md](results/2026-05-28_resumen_evaluacion.md) | Resultados detallados por experimento: ranking global, análisis por segmento y rango de edad | Vigente |
| [ablacion_backbones/](results/ablacion_backbones/2026-07-22_ablacion_backbones.md) | Comparación formal de los 4 backbones (ResNet50/VGG16/DenseNet121/InceptionV3) con pruebas pareadas | Vigente |
| [experimentos_adicionales/](results/experimentos_adicionales/analisis.md) | Experimentos y análisis adicionales para el manuscrito publicado: baseline whole-hand, Dice/IoU por clase, benchmark de latencia, significancia estadística | **Vigente: más reciente** |

---

## revisiones/: Respuestas a comentarios de revisores

El artículo ya fue publicado; estos documentos son registro histórico cerrado del proceso de revisión, no trabajo pendiente.

| Archivo | Contenido |
|---|---|
| [ronda1_analisis.md](revisiones/ronda1_analisis.md) | Respuestas a la primera ronda de revisión |
| [../results/experimentos_adicionales/analisis.md](results/experimentos_adicionales/analisis.md) | Respuestas y experimentos adicionales de la segunda ronda (vive en `results/` por la cantidad de datos/figuras/scripts de respaldo que genera) |

---

## scripts/: Documentación de scripts individuales

Los 12 scripts del pipeline (`src/00`–`src/11`), cada uno con descripción, uso, entradas/salidas:

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
