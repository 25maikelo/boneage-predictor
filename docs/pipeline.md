# Pipeline — Bone Age Predictor

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 09
                                    ↓
                                    08
```

---

## Preprocesamiento

### 00 · Descarga del dataset
**Script:**
`src/preprocessing/00_download_dataset.py`

Descarga el dataset RSNA Bone Age desde Kaggle usando `kagglehub`.

**Salida:**
`data/images/raw/` — 13,014 imágenes PNG originales

---

### 01 · Entrenamiento del segmentador
**Script:**
`src/preprocessing/01_train_hand_detector.py`

Entrena el modelo UNet + MobileNetV2 para segmentación semántica de la mano en 4 regiones anatómicas. Cada ejecución genera un run en `models/hand-detector/hand-detector_NN/`.

**Entrada:**
`data/hand-detector/` — imágenes anotadas (LabelMe)

**Salida:**
`models/hand-detector/hand-detector_NN/models/modelo_segmentacion.h5`

**Tiempo:**
~29 min (GPU)

> El modelo activo es `hand-detector_00`. Se selecciona automáticamente vía `get_segmentation_model_path()`.

---

### 02 · Recorte y zoom
**Script:**
`src/preprocessing/02_frame_and_zoom.py`

Rotación y recorte de las imágenes de rayos X. Incluye revisión manual de calidad (13 imágenes eliminadas, 190 volteadas — mano izquierda).

**Entrada:**
`data/images/raw/` (12,811 imágenes tras revisión)

**Salida:**
`data/images/cropped/` — 12,811 imágenes

**Tiempo:**
~19 min (CPU)

---

### 03 · Ecualización de histograma
**Script:**
`src/preprocessing/03_histogram_equalization.py`

Ecualización adaptativa CLAHE para mejorar el contraste de las radiografías.

**Entrada:**
`data/images/cropped/`

**Salida:**
`data/images/equalized/` — 12,811 imágenes

**Tiempo:**
~11 min (CPU)

---

### 04 · Segmentación de imágenes
**Script:**
`src/preprocessing/04_segment_images.py`

Aplica el modelo de segmentación para extraer las 4 regiones anatómicas de cada imagen: `pinky`, `middle`, `thumb`, `wrist`. Genera también las máscaras de segmentación.

**Entrada:**
`data/images/equalized/`

**Salida:**
- `data/images/segmented/{pinky,middle,thumb,wrist}/` — 51,244 imágenes (12,811 × 4)
- `data/images/masks/` — 12,811 máscaras PNG

**Modelo:**
`models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5`

**Tiempo:**
~2h 05 min (GPU)

---

## Entrenamiento

### 05 · Análisis del dataset
**Script:**
`src/05_dataset_analysis.py`

**Uso:**
`python src/05_dataset_analysis.py --experiment N`

Filtra el CSV de entrenamiento para conservar solo imágenes con los 4 segmentos y edades con al menos 50 muestras. Genera el dataset balanceado.

**Entrada:**
`data/training/boneage-training-dataset.csv` (12,611 filas)

**Salida:**
`data/training/dataset_analysis/balanced_dataset.csv` (11,783 filas · 36 edades · rango 24–216 meses)

**Tiempo:**
~18 s

---

### 06 · Entrenamiento
**Script:**
`src/06_training.py`

**Uso:**
`python src/06_training.py --experiment N`

Entrena 4 modelos de segmento (uno por región anatómica) con K-Fold CV, luego construye y entrena el modelo de fusión. Soporta dos arquitecturas (ver [arquitecturas.md](arquitecturas.md)):

- `MODEL_TYPE = "simple_cnn"` — CNN desde cero con Flatten
- `MODEL_TYPE = "backbone"` — backbone preentrenado (VGG16, DenseNet121, InceptionV3, ResNet50)

**Fases:**
1. **Segmentos** — `EPOCHS_SEGMENT` épocas × 5 folds por segmento (4 segmentos)
2. **Fusión** — `FUSION_EPOCHS` épocas con extractores congelados
3. **Fine-tuning** — `FINE_TUNING_EPOCHS` épocas con modelo completo

**Entrada:**
`data/images/segmented/`, `data/training/dataset_analysis/balanced_dataset.csv`

**Salida:**
`experiments/N/models/`, `experiments/N/training_history/`

---

## Validación y Análisis

### 07 · Validación estándar
**Script:**
`src/07_validation.py`

**Uso:**
`python src/07_validation.py --experiment N`

Evalúa el modelo fusionado sobre el dataset de validación RSNA. Genera mapas de saliencia, scatter plots y tabla resumen.

**Entrada:**
`data/validation/` (1,425 imágenes), `experiments/N/models/fusion_model`

**Salida:**
`experiments/N/validation/`

---

### 08 · Validación mexicana
**Script:**
`src/08_mex_validation.py`

**Uso:**
`python src/08_mex_validation.py --experiment N`

Evalúa el modelo sobre el dataset de pacientes mexicanos (IMSS). Mide generalización geográfica.

**Entrada:**
`data/mex-validation/` (100 imágenes), `experiments/N/models/fusion_model`

**Salida:**
`experiments/N/mex-validation/`

---

### 09 · Análisis de desempeño
**Script:**
`src/09_performance_analysis.py`

**Uso:**
`python src/09_performance_analysis.py --experiment N`

Genera tabla comparativa de métricas, mapas de saliencia por segmento y análisis de errores.

**Entrada:**
`experiments/N/validation/`, `experiments/N/mex-validation/`

**Salida:**
`experiments/N/evaluation/`

---

## Ejecución en Clúster (SLURM)

```bash
sbatch slurm/06_training.slurm N        # entrenamiento
sbatch slurm/07_validation.slurm N      # validación estándar
sbatch slurm/08_mex_validation.slurm N  # validación mexicana
sbatch slurm/09_performance_analysis.slurm N

bash scripts/quicktest.sh 99            # pipeline completo con datos mínimos
```

**Partición GPU** (`gpu`): nvd01, nvd02 — scripts 04, 06, 07, 08

**Partición CPU** (`q1`): quicktest y scripts ligeros
