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
| [2026-06-18_optimizacion.md](results/2026-06-18_optimizacion.md) | Fase 6 (exps 47–56): ablaciones de género, LR/épocas, tamaño de imagen | Histórico (completado) |
| [2026-05-28_resultados.md](results/2026-05-28_resultados.md) | Tabla resumen con podio y hallazgos de los primeros 14 experimentos | Histórico, con nota de corrección (columnas Mex MAE afectadas por el bug `real_age`/`bone_age`, ver abajo) |
| [2026-05-28_resumen_evaluacion.md](results/2026-05-28_resumen_evaluacion.md) | Resultados detallados por experimento: ranking global, análisis por segmento y rango de edad | Histórico, con la misma nota de corrección |
| [ablacion_backbones/](results/ablacion_backbones/2026-07-22_ablacion_backbones.md) | Comparación formal de los 4 backbones (ResNet50/VGG16/DenseNet121/InceptionV3) con pruebas pareadas | Vigente |
| [experimentos_adicionales/](results/experimentos_adicionales/analisis.md) | Experimentos y análisis adicionales en respuesta a la segunda ronda de revisión: baseline whole-hand, corrección del bug MEX, Dice/IoU por clase, benchmark de latencia, significancia estadística | **Vigente: más reciente** |

> El bug corregido en `src/08_mex_validation.py` (comparaba contra `real_age` en vez de
> `bone_age`) afecta las columnas "Mex MAE" de los documentos de 2026-04-29 y de
> `2026-06-18_optimizacion.md`. Los números recalculados para los experimentos 23/26/27/28 y
> 33/34/36/37/39/40 están en `experimentos_adicionales/analisis.md` y en `ablacion_backbones/`; el
> resto de los experimentos no se ha vuelto a validar contra MEX con el script corregido.

---

## revisiones/: Respuestas a comentarios de revisores

| Archivo | Contenido |
|---|---|
| [ronda1_analisis.md](revisiones/ronda1_analisis.md) | Respuestas propuestas a la primera ronda de revisión |
| [../results/experimentos_adicionales/analisis.md](results/experimentos_adicionales/analisis.md) | Respuestas y experimentos adicionales de la segunda ronda (vive en `results/` por la cantidad de datos/figuras/scripts de respaldo que genera) |

---

## scripts/: Documentación de scripts individuales

| Archivo | Script | Contenido |
|---|---|---|
| [10_age_range_analysis.md](scripts/10_age_range_analysis.md) | `src/10_age_range_analysis.py` | Salidas, justificación estadística del bin-size, grupos pediátricos, comando para todos los experimentos |

> Cobertura incompleta: solo 1 de los ~11 scripts del pipeline (`src/00`–`src/10`) tiene
> documentación dedicada aquí. El resto está descrito de forma más breve en
> [`planning/pipeline.md`](planning/pipeline.md).
