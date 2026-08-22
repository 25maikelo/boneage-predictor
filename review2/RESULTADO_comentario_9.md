# Comentario 9 — Validación de segmentación (split real + Dice/IoU por clase)

> Calculado directamente sobre el modelo de segmentación activo (`models/hand-detector/hand-detector_00/`) y el dataset anotado real (`data/hand-detector/`). Script: [`scripts/per_class_dice.py`](scripts/per_class_dice.py). JSON crudo: [`results_json/per_class_dice_results.json`](results_json/per_class_dice_results.json).

## Hallazgo que corrige un dato del manuscrito

El manuscrito (Sección 2.3) dice: *"A total of 200 radiographs were manually annotated at pixel level..."* — recontando directamente `data/hand-detector/images/` y `data/hand-detector/annotations/`, y confirmando contra el log de entrenamiento original del modelo activo (que registra arrays de forma `(303, 224, 224, 5)` train y `(76, 224, 224, 5)` val), el dataset real anotado y usado para entrenar la segmentación tiene **379 imágenes**, no 200. Split reproducible: `train_test_split(test_size=0.2, random_state=42)` → **303 train / 76 val**.

**📝 Reemplaza esto** (Sección 2.3):
> "Segmentation ground-truth masks were generated using the LabelMe annotation tool. A total of 200 radiographs were manually annotated at pixel level by delineating the four anatomical regions of interest (thumb, middle finger, pinky finger, and wrist) according to the TW3 protocol."

**Por esto:**
> "Segmentation ground-truth masks were generated using the LabelMe annotation tool. A total of 379 radiographs were manually annotated at pixel level by delineating the four anatomical regions of interest (thumb, middle finger, pinky finger, and wrist) according to the TW3 protocol. The annotated set was split into 303 training images and 76 validation images (80/20, fixed random seed = 42), with no patient-level overlap between subsets."

## Dice/IoU por clase (split de validación real, n=76)

| Clase | Dice (media ± DE) | IoU (media ± DE) |
|---|:---:|:---:|
| Fondo | 0.9828 ± 0.0067 | 0.9663 ± 0.0128 |
| **Meñique (pinky)** | **0.7785 ± 0.2004** | **0.6712 ± 0.2105** |
| **Medio (middle)** | **0.8905 ± 0.1138** | **0.8147 ± 0.1213** |
| **Pulgar (thumb)** | **0.8636 ± 0.1506** | **0.7799 ± 0.1527** |
| **Muñeca (wrist)** | **0.7695 ± 0.1727** | **0.6514 ± 0.1890** |
| Macro (4 clases anatómicas, sin fondo) | 0.8255 | 0.7293 |
| Pixel accuracy (referencia, secundaria) | 0.9694 | — |

**Nota metodológica:** el Dice agregado de 0.9168 que hoy reporta el manuscrito (Sección 3.1) es notablemente más alto que el promedio macro por clase (0.8255–0.8570). Esto es consistente con que la cifra publicada se calculó como Dice "suave" (sobre probabilidades softmax continuas, la misma métrica usada durante el entrenamiento) en vez de Dice "duro" (sobre la clase final asignada por argmax), que es la forma estándar de reportarlo en segmentación clínica y la que pide el revisor ("class-specific Dice coefficients"). Además hay heterogeneidad real entre clases — meñique y muñeca (Dice ≈0.77–0.78) son notablemente más difíciles de segmentar que el dedo medio (Dice ≈0.89) — algo que el número agregado ocultaba.

**📝 Agrega esto** justo después de (Sección 3.1):
> "...No confidence intervals are reported for segmentation metrics, as the evaluation is performed on a fixed validation set and the primary objective is comparative model selection rather than statistical inference."

**el siguiente párrafo + tabla:**
> "To address per-class segmentation quality, Dice and IoU were additionally computed per anatomical region on the same validation split (n = 76), using the final argmax-assigned class rather than the continuous softmax output (Table X). Segmentation quality was not uniform across regions: the middle finger and thumb achieved the highest overlap (Dice = 0.89 and 0.86, respectively), while the pinky finger and wrist showed lower and more variable agreement (Dice = 0.78 and 0.77, respectively; SD > 0.17 in both cases), likely reflecting their smaller relative area and greater anatomical variability near the image boundary. The macro-averaged Dice across the four anatomical regions (0.83) is meaningfully lower than the single aggregate Dice value obtained under soft (probability-based) scoring, underscoring the value of per-class, hard-label reporting for segmentation-dependent downstream pipelines such as this one."

*(Tabla: usar la tabla de arriba.)*

## ⚠️ El "val" reportado no es un test ciego — y no se puede generar uno nuevo

El comentario pide explícitamente las cantidades usadas para "training, **validation, and testing**" de la segmentación, como tres conjuntos distintos. Al revisar `src/preprocessing/01_train_hand_detector.py` (línea 818-830), el entrenamiento del modelo activo usa:

```python
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
]
...
validation_data=_gen(val_img_gen, val_mask_gen),  # las mismas 76 imágenes
```

Es decir, **las 76 imágenes de "validación" son el mismo conjunto que determinó qué checkpoint quedarse** (`restore_best_weights=True` sobre `val_loss`) y cómo bajó el learning rate durante el entrenamiento. No son un conjunto de test independiente — son, en la práctica, un conjunto de ajuste (tuning), igual que en el problema de fuga del Comentario 12 pero aplicado a la selección de checkpoint del segmentador.

**No se puede resolver entrenando un modelo de segmentación nuevo con un split de 3 vías**, porque `hand-detector_00` es el modelo que efectivamente generó las máscaras con las que se preprocesaron y entrenaron **todos** los experimentos de edad ósea del estudio — reportar el Dice/IoU de un modelo distinto (aunque metodológicamente más limpio) describiría un segmentador que nunca se usó realmente, lo cual sería peor que el problema actual.

**La respuesta honesta es declarar la limitación explícitamente**, no maquillar las 76 imágenes como si fueran un test ciego:

**📝 Reemplaza esto** (Sección 2.3, después del párrafo ya corregido arriba sobre 379/303/76):
> *(no hay texto actual sobre esto — es una omisión, no una afirmación incorrecta a reemplazar)*

**Agrega esto** como aclaración explícita, justo después del párrafo del split 303/76:
> "No independent, held-out test set was used for segmentation: the 76-image validation split was also used to monitor validation loss for early stopping and learning-rate reduction during training of the production segmentation model. Consequently, the Dice and IoU values reported in Section 3.1 reflect performance on the same set used for checkpoint selection, not on data entirely unseen by the model-selection process. This is acknowledged as a methodological limitation; training a separate segmentation model with a proper three-way split was considered but rejected, since the reported metrics must correspond to the exact segmentation model (`hand-detector_00`) that generated the masks used to train and evaluate every bone age experiment in this study — evaluating a different model would misrepresent the pipeline actually used."

## Otras limitaciones del comentario que siguen sin poder responderse con datos

- **Independencia de pacientes entre train/val**: no verificado explícitamente (el dataset de segmentación no tiene un campo de ID de paciente separable del ID de imagen en el CSV/anotaciones actuales; asumible si cada imagen es de un paciente distinto, pero no confirmado formalmente).
- **Procedimiento de revisión de máscaras**: un solo revisor clínico ("Daniel", del grupo de trabajo), sin segunda opinión ni cálculo de concordancia inter-observador — limitación real, a declarar explícitamente en el manuscrito, no resoluble sin una segunda anotación.

**📝 Agrega esto** como nota de limitación (Sección 3.5 o al final de 2.3):
> "Mask review was performed by a single clinical annotator from the study's working group, without a second independent reviewer or a formal measure of inter-observer agreement on the 379 annotated masks. This is acknowledged as a limitation of the segmentation validation."
