# Bone Age Predictor

Predicción de edad ósea mediante aprendizaje profundo sobre radiografías de mano.
El sistema segmenta la mano en 4 regiones anatómicas y entrena modelos independientes que se fusionan para la predicción final.

---

## Estructura del Proyecto

```
boneage-predictor/
├── config/
│   ├── paths.py          ← Todas las rutas (fuente única de verdad)
│   └── experiment.py     ← Cargador de configuraciones de experimento
├── src/
│   ├── preprocessing/
│   │   ├── 00_download_dataset.py   ← Descarga imágenes de Kaggle
│   │   ├── 01_train_hand_detector.py
│   │   ├── 02_frame_and_zoom.py     ← Rotación y recorte
│   │   ├── 03_histogram_equalization.py ← CLAHE
│   │   └── 04_segment_images.py    ← Segmentación en 4 regiones
│   ├── models/
│   │   ├── losses.py       ← Funciones de pérdida personalizadas
│   │   └── fusion_utils.py ← Capas de fusión con atención
│   ├── utils/
│   │   └── timing.py       ← Reporte de tiempos de ejecución
│   ├── dataset_analysis.py
│   ├── training.py
│   ├── validation.py
│   ├── mex_validation.py
│   └── performance_analysis.py
├── slurm/                  ← Scripts de SLURM (0-9, un paso por script)
├── experiments/
│   └── 26/
│       └── config.py       ← Hiperparámetros del experimento
├── images/                 ← (gitignored) Imágenes del dataset
├── models/                 ← (gitignored) Modelos pre-entrenados
├── training-data/          ← (gitignored) CSVs de entrenamiento
├── validation-data/        ← (gitignored)
├── mex-validation-data/    ← (gitignored)
├── logs/                   ← (gitignored) Logs del clúster
├── requirements.txt
└── .gitignore
```

---

## Instalación

```bash
pip install -r requirements.txt
```

Para usar GPU, instala los drivers CUDA correspondientes a tu versión de TensorFlow.

---

## Pipeline Completo

Todos los scripts se ejecutan desde el **root del proyecto** y aceptan `--experiment N`.

### Ejecución Local

```bash
# Paso 0: Descargar dataset RSNA Bone Age desde Kaggle
python src/preprocessing/00_download_dataset.py

# Paso 1: Entrenar detector de mano (U-Net + MobileNetV2)
python src/preprocessing/01_train_hand_detector.py

# Paso 2: Rotación y recorte de imágenes
python src/preprocessing/02_frame_and_zoom.py

# Paso 3: Ecualización de histograma (CLAHE)
python src/preprocessing/03_histogram_equalization.py

# Paso 4: Segmentación en 4 regiones anatómicas
python src/preprocessing/04_segment_images.py

# Paso 5: Análisis y balanceo del dataset
python src/dataset_analysis.py --experiment 26

# Paso 6: Entrenamiento de segmentos + fusión
python src/training.py --experiment 26

# Paso 7: Validación sobre dataset estándar
python src/validation.py --experiment 26

# Paso 8: Validación sobre dataset mexicano
python src/mex_validation.py --experiment 26

# Paso 9: Análisis de desempeño (tabla + saliencias)
python src/performance_analysis.py --experiment 26
```

### Ejecución en Clúster SLURM

Los mismos pasos, con `sbatch`. El argumento opcional es el número de experimento.

```bash
sbatch slurm/00_download.slurm
sbatch slurm/01_hand_detector.slurm
sbatch slurm/02_frame_and_zoom.slurm
sbatch slurm/03_histogram_equalization.slurm
sbatch slurm/04_segment.slurm
sbatch slurm/05_dataset_analysis.slurm 26
sbatch slurm/06_training.slurm 26
sbatch slurm/07_validation.slurm 26
sbatch slurm/08_mex_validation.slurm 26
sbatch slurm/09_performance_analysis.slurm 26
```

---

## Configuración

### Rutas: `config/paths.py`

Fuente única de verdad para todas las rutas del proyecto.
**No edites rutas en los scripts individuales**, edita solo este archivo.

### Hiperparámetros: `experiments/<N>/config.py`

Cada experimento tiene su propio archivo de configuración con hiperparámetros.
Para crear un nuevo experimento:

```bash
cp experiments/26/config.py experiments/27/config.py
# Edita experiments/27/config.py con los hiperparámetros deseados
python src/training.py --experiment 27
```

Variables clave:

| Variable | Descripción |
|---|---|
| `IMAGE_SIZE` | Tamaño de entrada del modelo (px) |
| `BASE_MODEL_CHOICE` | Backbone: `vgg16`, `densenet121`, `inceptionv3`, `resnet50` |
| `LOSS_FUNCTION_NAME` | Pérdida: `attention_loss`, `dynamic_attention_loss`, `custom_mse_loss`, `custom_huber_loss` |
| `EPOCHS_SEGMENT` | Épocas de entrenamiento por segmento |
| `FUSION_EPOCHS` | Épocas del modelo de fusión |
| `USE_GENDER` | Incluir género como feature |
| `SEGMENTS_ORDER` | Orden de segmentos para el modelo de fusión |

---

## Arquitectura

```
Imagen de RX
    ↓ Rotación + CLAHE
    ↓ Segmentación U-Net → [pinky | middle | thumb | wrist]
    ↓
 ┌──────┐ ┌────────┐ ┌───────┐ ┌───────┐
 │Pinky │ │ Middle │ │ Thumb │ │ Wrist │  ← Modelos CNN independientes
 └──┬───┘ └───┬────┘ └───┬───┘ └───┬───┘
    └──────────┴──────────┴─────────┘
                    ↓ Fusión + Género
              Predicción: Edad ósea (meses)
```

---

## Reportes de Tiempo

Todos los scripts imprimen un reporte al finalizar:

```
=======================================================
[TIMING REPORT] training.py
Tiempo total de ejecución: 2:34:07
Finalizado: 2026-03-11 15:42:00
=======================================================
```

---

## Dataset

El proyecto usa el dataset [RSNA Pediatric Bone Age Challenge](https://www.kaggle.com/datasets/kmader/rsna-bone-age) de Kaggle.
Se descarga automáticamente con `src/preprocessing/00_download_dataset.py`.

También incluye validación sobre un dataset de pacientes mexicanos (no público).
