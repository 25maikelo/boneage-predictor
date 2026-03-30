#!/bin/bash
# download_data.sh
# Descarga los datos y modelos necesarios desde Google Drive usando rclone.
# Requisitos: rclone configurado con un remote llamado 'gdrive' con acceso a Google Drive.
#
# Uso: bash scripts/download_data.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Directorio del proyecto: $PROJECT_ROOT"

# imágenes raw del dataset RSNA
echo "[1/6] Descargando data/images/raw..."
rclone copy "gdrive:" "$PROJECT_ROOT/data/images/raw" \
    --drive-root-folder-id 18vLXotwEtkOyX_spNp73sSLxvnzfRmUy \
    --exclude "deleted/**" --exclude "flipped/**" \
    --progress

# hand-detector (imágenes + anotaciones LabelMe)
echo "[2/6] Descargando data/hand-detector..."
rclone copy "gdrive:" "$PROJECT_ROOT/data/hand-detector" \
    --drive-root-folder-id 1BO8jUEtKtVXBvSPrRdLf34u3fYXTpu7a \
    --progress

# training (CSV principal + dataset_analysis)
echo "[3/6] Descargando data/training..."
rclone copy "gdrive:" "$PROJECT_ROOT/data/training" \
    --drive-root-folder-id 1F2m10_S__MFCqz5oJKnR2iPv8Rlg6juR \
    --progress

# validation (imágenes + CSV)
echo "[4/6] Descargando data/validation..."
rclone copy "gdrive:" "$PROJECT_ROOT/data/validation" \
    --drive-root-folder-id 1oVrdha-NUaHXjbasdaaSO5sOoWRvhX7N \
    --progress

# mex-validation (imágenes + CSV)
echo "[5/6] Descargando data/mex-validation..."
rclone copy "gdrive:" "$PROJECT_ROOT/data/mex-validation" \
    --drive-root-folder-id 11PACx89AcI3yt5ZHpQQh-vPjiD64zsDD \
    --progress

# modelo de segmentación pre-entrenado
echo "[6/6] Descargando models/modelo_segmentacion.h5..."
mkdir -p "$PROJECT_ROOT/models"
source "$PROJECT_ROOT/venv/bin/activate"
gdown "173V_YvxFWiDBOib_9CCRfTZB7z0XtpiA" -O "$PROJECT_ROOT/models/modelo_segmentacion.h5"

echo "Descarga completada."
