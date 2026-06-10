# Documentación — Bone Age Predictor

## Mapa de archivos

### planning/ — Flujo del proyecto y próximos pasos

| Archivo | Contenido |
|---|---|
| [pipeline.md](planning/pipeline.md) | Descripción de cada script (00–10), entradas/salidas, tiempos y comandos SLURM |
| [plan_exploracion.md](planning/plan_exploracion.md) | Ablaciones planificadas post-Fase 5: género, LR/épocas, tamaño de imagen, datos clínicos |

---

### design/ — Arquitecturas y decisiones de diseño

| Archivo | Contenido |
|---|---|
| [arquitecturas.md](design/arquitecturas.md) | Diagramas y parámetros de los 4 modos: `backbone`, `simple_cnn`, `backbone_vectors`, `unified_cnn` |
| [propuesta_densenet_vector_fusion.md](design/propuesta_densenet_vector_fusion.md) | Propuesta original de `backbone_vectors` (contexto histórico — ya implementado en exps 35/38/41/43) |

---

### data/ — Dataset

| Archivo | Contenido |
|---|---|
| [dataset_report.md](data/dataset_report.md) | Estadísticas de procesamiento: imágenes raw → cropped → equalized → segmented, splits, distribución de edades |

---

### results/ — Resultados y evaluación

| Archivo | Contenido |
|---|---|
| [resultados.md](results/resultados.md) | Tabla resumen de todos los experimentos con podio y hallazgos principales |
| [resumen_evaluacion.md](results/resumen_evaluacion.md) | Resultados detallados: scripts 07/08/09/10, ranking global, análisis por segmento y rango de edad, conclusiones |
| [experiments_summary.md](results/experiments_summary.md) | Historial completo por experimento: configuración, MAE CV por segmento, estado, línea de tiempo de código |
| [optimizacion.md](results/optimizacion.md) | **Documento vivo** — experimentos de optimización Fase 6 (exps 47+): género, LR/épocas, imagen, datos clínicos |

---

### scripts/ — Documentación de scripts individuales

| Archivo | Script | Contenido |
|---|---|---|
| [10_age_range_analysis.md](scripts/10_age_range_analysis.md) | `src/10_age_range_analysis.py` | Salidas, justificación estadística del bin-size, grupos pediátricos, comando para todos los experimentos |
