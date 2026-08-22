# Comentario 14 — Comparación de arquitecturas no controlada

> Nota: en el borrador de respuestas (`Reviewers.docx.pdf`), la "Respuesta 14" en realidad responde al Comentario 13 (función de pérdida) por error — este comentario (control de la comparación de arquitecturas) está efectivamente sin respuesta todavía. Ver `ANALISIS_COMENTARIOS.md`, Hallazgo Crítico #2.

## Hiperparámetros por backbone — confirmado idéntico

Verificado directamente contra `experiments/{23,26,27,28}/config.py` (23=ResNet50, 26=VGG16, 27=DenseNet121, 28=InceptionV3) y contra el código compartido `src/06_training.py`:

| Parámetro | Valor (igual en los 4 backbones) |
|---|---|
| Pesos iniciales (`WEIGHTS`) | `None` — entrenados desde cero, sin ImageNet |
| Capas descongeladas (`NUM_LAYERS_UNFREEZE`) | 10 |
| Extractores congelados en fusión (`FREEZE_EXTRACTORS`) | `True` |
| Optimizador | Adam |
| Learning rate | 0.001 (fusión); LR/10 en fine-tuning |
| Early stopping | `patience=4` sobre `val_loss`, `restore_best_weights=True` (hardcodeado en `06_training.py`, igual para los 4) |
| Reducción de LR | `ReduceLROnPlateau(factor=0.5, patience=3)` sobre `val_loss` (igual para los 4) |
| Weight decay | No implementado (Adam estándar, sin weight decay) |
| Normalización de imagen | Rescale 1/255 (Tabla 3 del manuscrito) |
| Presupuesto de épocas | 15 (segmento) / 20 (fusión) / 10 (fine-tuning) — igual para los 4 |

**📝 Reemplaza esto** (Sección 2.5, penúltimo párrafo — cita textual del comentario del revisor, no está en el manuscrito pero resume lo que falta):
> *(el manuscrito actual no tiene una tabla de hiperparámetros por backbone — es una omisión, no una afirmación a corregir)*

**Agrega esto** justo después de la Tabla 3 (Data Augmentation Configuration), como nueva tabla o ampliando la Tabla 2:
> "All four backbones were trained under identical optimization conditions: Adam optimizer, learning rate = 0.001 (0.0001 during fine-tuning), no ImageNet pretraining (weights initialized from scratch), the last 10 layers unfrozen during the segment-training phase, feature extractors frozen during the fusion phase, early stopping with patience = 4 epochs on validation loss, and learning-rate reduction with patience = 3 epochs (factor 0.5) — all identical across backbones by construction, since they share the same training code path. No weight decay was used in any configuration."

## Tiempo de entrenamiento real (empírico, no solo conteo de parámetros)

Documentado en `docs/results/ablacion_backbones/2026-07-22_ablacion_backbones.md`, tomado directamente de los logs SLURM de cada corrida:

| Backbone | Parámetros (fusionado) | Tiempo de entrenamiento GPU (real) | Épocas de fusión hasta early stopping |
|---|---:|:---:|:---:|
| F-ResNet50 | 96,451,973 | ~12 h | 9/20 |
| F-VGG16 | 59,387,013 | ~7 h | 7/20 (no mejora en fine-tuning) |
| **F-DenseNet121** | **29,202,565** | **~10 h** | 8/20 |
| F-InceptionV3 | 89,312,261 | ~14 h | 19/20 |

**📝 Reemplaza esto** (Sección 3.2, último párrafo):
> "In terms of model complexity, F-ResNet50 and F-InceptionV3 required the largest parameter counts in their fused states. This finding positions F-DenseNet121 as the architecture offering the optimal balance between predictive accuracy and parameter efficiency."

**Por esto:**
> "In terms of model complexity, F-ResNet50 and F-InceptionV3 required the largest parameter counts in their fused states (96.5M and 89.3M, respectively), compared to F-DenseNet121 (29.2M) and F-VGG16 (59.4M). Beyond parameter count, empirical GPU training time (Table X) shows F-DenseNet121 required substantially less time to train than F-InceptionV3 (~10 h vs. ~14 h) for statistically indistinguishable accuracy (Section 3.3), which — together with its lower parameter count — positions F-DenseNet121 as the architecture offering the best balance between predictive accuracy and computational cost, rather than parameter efficiency alone."

## Latencia de inferencia y memoria (medido en nodo GPU dedicado, batch=1, n=20 repeticiones)

Medido de forma aislada en un nodo GPU (`nvd01`), un backbone a la vez, sin otros jobs compitiendo por recursos. Script: [`scripts/latency_benchmark.py`](scripts/latency_benchmark.py). Log crudo: [`results_json/latency_benchmark.log`](results_json/latency_benchmark.log).

| Backbone | Parámetros | Latencia de inferencia (ms, media ± DE) | Memoria GPU pico |
|---|---:|---:|---:|
| F-ResNet50 | 96,451,973 | 241.3 ± 0.3 | 1,221.9 MB |
| **F-VGG16** | 59,387,013 | **40.7 ± 0.2** ← más rápido | 775.8 MB |
| F-DenseNet121 | 29,202,565 | 526.7 ± 1.5 | **359.1 MB** ← menor memoria |
| F-InceptionV3 | 89,312,261 | 423.3 ± 0.4 | 1,113.6 MB |

**Hallazgo honesto que hay que incluir, no solo el número favorable:** F-DenseNet121 tiene los **menos** parámetros y la **menor** memoria pico, pero es paradójicamente el **más lento** de los cuatro en latencia de inferencia por imagen (526.7 ms) — más lento incluso que ResNet50 (241.3 ms), que tiene 3.3× más parámetros. F-VGG16, pese a ser el peor en precisión, es el más rápido en inferencia (40.7 ms). Esto es consistente con que el conteo de parámetros no predice bien la latencia real: la conectividad densa de DenseNet121 (muchas concatenaciones y operaciones pequeñas) tiene overhead que no se refleja en el conteo de parámetros ni en el tiempo de entrenamiento. Es importante reportar esto tal cual, sin ocultarlo, porque responde exactamente a la crítica del revisor de no basar la afirmación de "balance óptimo" solo en parámetros — y aquí el resultado real es más matizado que favorable en todas las dimensiones.

**📝 Agrega esto** justo después del párrafo ya redactado arriba sobre tiempo de entrenamiento (Sección 3.2):
> "Empirical inference latency and peak GPU memory were also measured in isolation on a dedicated GPU node (batch size = 1, mean of 20 repetitions after 3 warmup runs; Table X). F-DenseNet121 had the lowest parameter count and the lowest peak memory (359 MB) among the four backbones, but, notably, the highest single-image inference latency (526.7 ms) — slower than F-ResNet50 (241.3 ms) despite having 3.3 times fewer parameters, likely reflecting the computational overhead of DenseNet121's dense connectivity pattern. F-VGG16 achieved the lowest inference latency (40.7 ms) despite its poor predictive accuracy (Section 3.4). These results indicate that parameter count alone does not reliably predict inference latency, and that the choice of F-DenseNet121 as the primary backbone in this study is best justified by its training-time efficiency and accuracy (Sections 3.2–3.3) rather than by inference-latency efficiency, which favors F-VGG16 despite its unsuitability on accuracy grounds."

**📝 Si se decide no medirlo**, agregar como limitación explícita en vez de omitirlo:
> "Empirical inference latency and memory utilization were not measured under controlled, isolated conditions in this study and are identified as necessary future work; training time (Table X) is reported as the available empirical efficiency measure."
