# Bone Age Predictor

Predicción de edad ósea mediante aprendizaje profundo sobre radiografías de mano.
El sistema segmenta la mano en 4 regiones anatómicas y entrena modelos independientes que se fusionan para la predicción final.

---

## Estructura del Proyecto

```
boneage-predictor/
├── config/
│   ├── paths.py           ← Todas las rutas (fuente única de verdad)
│   ├── experiment.py      ← Cargador de configuraciones de experimento
│   └── segmentation.py    ← Hiperparámetros del detector de mano
├── src/
│   ├── preprocessing/
│   │   ├── 00_download_dataset.py      ← Descarga imágenes de Kaggle
│   │   ├── 01_train_hand_detector.py   ← Entrena U-Net segmentador
│   │   ├── 02_frame_and_zoom.py        ← Rotación y recorte
│   │   ├── 03_histogram_equalization.py← CLAHE
│   │   └── 04_segment_images.py        ← Segmentación en 4 regiones
│   ├── models/
│   │   ├── losses.py        ← Funciones de pérdida personalizadas
│   │   └── fusion_utils.py  ← Capas de fusión con atención
│   ├── utils/
│   │   └── timing.py        ← Reporte de tiempos y logging a archivo
│   ├── 05_dataset_analysis.py
│   ├── 06_training.py
│   ├── 07_validation.py
│   ├── 08_mex_validation.py
│   └── 09_performance_analysis.py
├── slurm/                  ← Scripts de SLURM (00–09, un paso por script)
├── experiments/
│   └── NN/
│       └── config.py       ← Hiperparámetros del experimento N
├── data/                   ← (gitignored) Datos del pipeline
│   ├── images/             ← Imágenes en distintas etapas
│   ├── hand-detector/      ← Imágenes etiquetadas para el segmentador
│   ├── training/           ← CSVs de entrenamiento
│   ├── validation/         ← Dataset de validación estándar
│   └── mex-validation/     ← Dataset de validación mexicano
├── models/                 ← (gitignored) Modelos pre-entrenados
├── logs/                   ← (gitignored) Logs de ejecución
├── requirements.txt
└── .gitignore
```

---

## Pipeline: Qué genera cada script y dónde lo guarda

| # | Script | Entrada | Genera | Destino |
|---|--------|---------|--------|---------|
| 00 | `src/preprocessing/00_download_dataset.py` | Kaggle (RSNA Bone Age) | Imágenes PNG del dataset | `data/images/raw/` |
| 01 | `src/preprocessing/01_train_hand_detector.py` | `data/hand-detector/images/` + `annotations/` | Modelo segmentador U-Net (×2), curvas de entrenamiento, tabla de métricas (Loss/Accuracy/IoU/Dice), predicción de prueba con overlay, visualizaciones de muestras | `models/modelo_segmentacion.h5`, `models/hand-detector/models/segmentation_model.h5`, `models/hand-detector/training_history/training_history.png`, `models/hand-detector/evaluation/performance_table.png`, `models/hand-detector/evaluation/test_prediction.png`, `models/hand-detector/evaluation/samples/` |
| 02 | `src/preprocessing/02_frame_and_zoom.py` | `data/images/raw/` | Imágenes rotadas y recortadas | `data/images/cropped/` |
| 03 | `src/preprocessing/03_histogram_equalization.py` | `data/images/cropped/` | Imágenes con CLAHE aplicado | `data/images/equalized/` |
| 04 | `src/preprocessing/04_segment_images.py` | `data/images/equalized/` + `models/modelo_segmentacion.h5` | Segmentos: pinky, middle, thumb, wrist | `data/images/segmented/{pinky,middle,thumb,wrist}/` |
| 05 | `src/05_dataset_analysis.py --experiment N` | `data/training/*.csv` + `data/images/segmented/` | CSV balanceado, estadísticas JSON, histogramas de distribución de edad (original/filtrado/balanceado), gráfico de proporción | `data/training/dataset_analysis/` |
| 06 | `src/06_training.py --experiment N` | `data/images/segmented/` + `experiments/N/config.py` | Modelos de segmentos (con K-Fold CV opcional) + modelo de fusión, curvas de entrenamiento, métricas CV en JSON | `experiments/N/models/`, `experiments/N/training_history/` |
| 07 | `src/07_validation.py --experiment N` | `data/validation/` + `experiments/N/models/` | Histograma de edades, pastel de sexo, tabla resumen MAE/tiempos, saliencias sobre muestras, dispersión real vs predicción | `experiments/N/validation/` |
| 08 | `src/08_mex_validation.py --experiment N` | `data/mex-validation/` + `experiments/N/models/` | Histograma de edades, pastel de sexo, tabla resumen MAE, saliencias sobre muestras, dispersión real vs predicción | `experiments/N/mex-validation/` |
| 09 | `src/09_performance_analysis.py --experiment N` | `experiments/N/models/` + `data/images/segmented/` | Saliencias sobre muestras del dataset, tabla comparativa (Loss/MAE train-val por modelo) | `experiments/N/evaluation/` |

---

## Instalación

> Todos los comandos se ejecutan desde el **root del proyecto** (`boneage-predictor/`).

### 1. Requisitos previos

- Python 3.10–3.12 (recomendado; 3.14 puede tener incompatibilidades con TensorFlow)
- Git

### 2. Crear y activar el entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS / SLURM:**
```bash
python -m venv venv
source venv/bin/activate
```

El prompt cambiará a `(venv)` cuando esté activo. Todos los comandos siguientes deben ejecutarse con el entorno activo.

### 3. Instalar dependencias

**Windows:**
```bash
python.exe -m pip install --upgrade pip
python.exe -m pip install -r requirements.txt
```

**Linux / macOS / SLURM:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. GPU (opcional)

**Local:** TensorFlow 2.18 requiere **CUDA 11.8+** y **cuDNN 8.6+**.

**Clúster HPC (Leo Átrox, CADS):** CUDA 11.4 no es compatible con TF 2.18.
Usar el entorno conda `boneage_gpu` preconfigurado con TF 2.10 + cuDNN 8.1:

```bash
module load anaconda3/2024.02
conda activate boneage_gpu
```

Si necesitas recrear el entorno desde cero:
```bash
conda create --name boneage_gpu python=3.9 -y
conda activate boneage_gpu
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' \
     > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
pip install tensorflow==2.10
pip install pandas scikit-learn scipy "opencv-python<4.10" Pillow \
    scikit-image matplotlib seaborn joblib tqdm kagglehub
```

### 5. Verificar instalación

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Debe imprimir la versión de TF y la lista de GPUs (vacía si solo hay CPU).

---

## Pipeline Completo

Todos los scripts se ejecutan desde el **root del proyecto** con el entorno virtual activo.

### Ejecución Local

```bash
# Paso 0: Descargar dataset RSNA Bone Age desde Kaggle
python src/preprocessing/00_download_dataset.py

# Paso 1: Entrenar detector de mano (U-Net + MobileNetV2)
#         Datos: data/hand-detector/images/ + data/hand-detector/annotations/
python src/preprocessing/01_train_hand_detector.py

# Paso 2: Rotación y recorte de imágenes
python src/preprocessing/02_frame_and_zoom.py

# Paso 3: Ecualización de histograma (CLAHE)
python src/preprocessing/03_histogram_equalization.py

# Paso 4: Segmentación en 4 regiones anatómicas
python src/preprocessing/04_segment_images.py

# Paso 5: Análisis y balanceo del dataset
python src/05_dataset_analysis.py --experiment 26

# Paso 6: Entrenamiento de segmentos + fusión
python src/06_training.py --experiment 26

# Paso 7: Validación sobre dataset estándar
python src/07_validation.py --experiment 26

# Paso 8: Validación sobre dataset mexicano
python src/08_mex_validation.py --experiment 26

# Paso 9: Análisis de desempeño (tabla + saliencias)
python src/09_performance_analysis.py --experiment 26
```

Cada script crea un log en `logs/<script_name>_YYYYMMDD_HHMMSS.log`.

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

### Hiperparámetros del segmentador: `config/segmentation.py`

Controla el entrenamiento del modelo U-Net (script 01).
Edita este archivo para cambiar `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, etc.

### Hiperparámetros por experimento: `experiments/<N>/config.py`

Cada experimento tiene su propio archivo. Para crear uno nuevo:

```bash
cp experiments/26/config.py experiments/27/config.py
# Edita experiments/27/config.py
python src/06_training.py --experiment 27
```

Variables clave:

| Variable | Descripción |
|---|---|
| `IMAGE_SIZE` | Tamaño de entrada del modelo (px) |
| `BASE_MODEL_CHOICE` | Backbone: `vgg16`, `densenet121`, `inceptionv3`, `resnet50` |
| `LOSS_FUNCTION_NAME` | Pérdida: `attention_loss`, `dynamic_attention_loss`, `custom_mse_loss`, `custom_huber_loss` |
| `EPOCHS_SEGMENT` | Épocas de entrenamiento por segmento |
| `FUSION_EPOCHS` | Épocas del modelo de fusión |
| `FINE_TUNING_EPOCHS` | Épocas de fine-tuning del modelo de fusión |
| `USE_GENDER` | Incluir género como feature |
| `SEGMENTS_ORDER` | Orden de segmentos para el modelo de fusión |
| `USE_CROSS_VALIDATION` | Activar K-Fold CV en el entrenamiento de segmentos (`True`/`False`) |
| `N_FOLDS` | Número de folds para cross-validation (por defecto `5`) |

---

## Cross-Validation

El script `06_training.py` soporta **K-Fold Cross-Validation** sobre los modelos de segmento.
Se activa en el `config.py` del experimento:

```python
USE_CROSS_VALIDATION = True
N_FOLDS = 5
```

### Comportamiento

- Cada segmento (pinky, middle, thumb, wrist) se entrena `N_FOLDS` veces con distintas particiones del dataset.
- El fold con menor `val_loss` se guarda como `{segmento}_model.keras` (modelo definitivo).
- Los modelos intermedios se guardan como `{segmento}_fold{k}.keras`.
- El modelo de **fusión** se entrena una sola vez con los mejores modelos de segmento (Opción B estándar en papers).

### Salidas generadas en `experiments/N/training_history/`

| Archivo | Descripción |
|---|---|
| `{seg}_cv_metrics.json` | Métricas y historial completo de cada fold por segmento |
| `{seg}_fold{k}_history.png` | Curvas loss/MAE por época de cada fold |
| `cv_summary_table.png` | Tabla resumen: MAE por fold, media ± std, mejor fold |
| `cv_boxplot.png` | Boxplot de distribución del val MAE entre folds por segmento |
| `cv_heatmap.png` | Heatmap val MAE — Fold × Segmento |
| `{seg}_cv_bars.png` | Barras de val MAE y val Loss por fold |
| `cv_summary.json` | Resumen global con media, std y mejor fold por segmento |

---

## Regeneración de Gráficos en Otro Idioma

Todos los scripts guardan los datos subyacentes de sus gráficos en archivos JSON.
El script `scripts/generate_plots.py` lee esos JSONs y regenera los gráficos en el idioma elegido **sin necesidad de reentrenar ni de tener los modelos cargados**.

```bash
# Español (por defecto)
python scripts/generate_plots.py --experiment 30 --lang es

# Inglés
python scripts/generate_plots.py --experiment 30 --lang en
```

Los gráficos generados se guardan con sufijo `_es.png` o `_en.png` junto a los originales.

### Cobertura por script

| Script origen | Gráficos regenerados |
|---|---|
| `05_dataset_analysis` | Histogramas de distribución de edad, pastel original vs usables |
| `06_training` | Curvas loss/MAE por fold, histogramas de fusión y fine-tuning, todos los gráficos CV |
| `07_validation` | Histograma de edades, pastel de sexo, tabla resumen, scatter real vs predicción |
| `08_mex_validation` | Igual que 07 para el dataset mexicano |
| `09_performance_analysis` | Tabla comparativa de modelos |

### Datos guardados por script

Cada script guarda un archivo JSON con los datos de sus gráficos al ejecutarse:

| Script | Archivo JSON |
|---|---|
| `05` | `data/training/dataset_analysis/plot_data.json` |
| `06` (segmentos) | `experiments/N/training_history/{seg}_cv_metrics.json`, `{seg}_history.json` |
| `06` (fusión) | `experiments/N/training_history/fusion_history.json`, `fusion_ft.json` |
| `07` | `experiments/N/validation/plot_data.json` |
| `08` | `experiments/N/mex-validation/plot_data.json` |
| `09` | `experiments/N/evaluation/comparative_table_data.json` |

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

Todos los scripts imprimen un reporte al finalizar y crean un log en `logs/`:

```
=======================================================
[TIMING REPORT] 06_training.py
Tiempo total de ejecución: 2:34:07
Finalizado: 2026-03-11 15:42:00
=======================================================
```

---

## Sincronización con Drive

Al sincronizar la carpeta del proyecto, las siguientes carpetas **no son necesarias** para ejecutar el pipeline y pueden excluirse para ahorrar espacio:

| Carpeta | Motivo |
|---|---|
| `venv/` | Entorno virtual de Python — se recrea con `pip install -r requirements.txt` |
| `**/__pycache__/` | Caché de bytecode de Python — se regenera automáticamente |
| `.git/` | Historial de Git — no necesario para solo ejecutar el código |

---

## Dataset

El proyecto usa el dataset [RSNA Pediatric Bone Age Challenge](https://www.kaggle.com/datasets/kmader/rsna-bone-age) de Kaggle.
Se descarga automáticamente con `src/preprocessing/00_download_dataset.py`.

El detector de mano se entrena con imágenes etiquetadas manualmente en formato LabelMe,
almacenadas en `data/hand-detector/` (4 clases: pinky, middle, thumb, wrist).

También incluye validación sobre un dataset de pacientes mexicanos (no público).
