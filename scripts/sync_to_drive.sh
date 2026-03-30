#!/bin/bash
# sync_to_drive.sh
# Sube al Drive los archivos generados por el pipeline que no están en git.
# Requisitos: rclone configurado con remote 'gdrive'.
#
# Uso:
#   bash scripts/sync_to_drive.sh            # sincroniza todo
#   bash scripts/sync_to_drive.sh --exp 30   # incluye resultados del experimento 30

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIVE_ROOT="gdrive:BoneAgePredictor"

# Parsear argumento opcional --exp N
EXP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp) EXP="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "Sincronizando hacia $DRIVE_ROOT ..."

# Datos de entrada (imágenes raw, hand-detector, training, validation, mex-validation)
echo "[1] data/images/raw..."
rclone sync "$PROJECT_ROOT/data/images/raw" "$DRIVE_ROOT/data/images/raw" \
    --exclude "deleted/**" --exclude "flipped/**" --progress

echo "[2] data/hand-detector..."
rclone sync "$PROJECT_ROOT/data/hand-detector" "$DRIVE_ROOT/data/hand-detector" --progress

echo "[3] data/training..."
rclone sync "$PROJECT_ROOT/data/training" "$DRIVE_ROOT/data/training" --progress

echo "[4] data/validation..."
rclone sync "$PROJECT_ROOT/data/validation" "$DRIVE_ROOT/data/validation" --progress

echo "[5] data/mex-validation..."
rclone sync "$PROJECT_ROOT/data/mex-validation" "$DRIVE_ROOT/data/mex-validation" --progress

# Imágenes procesadas por el pipeline
echo "[6] data/images/cropped..."
rclone sync "$PROJECT_ROOT/data/images/cropped" "$DRIVE_ROOT/data/images/cropped" --progress

echo "[7] data/images/equalized..."
rclone sync "$PROJECT_ROOT/data/images/equalized" "$DRIVE_ROOT/data/images/equalized" --progress

echo "[8] data/images/segmented..."
rclone sync "$PROJECT_ROOT/data/images/segmented" "$DRIVE_ROOT/data/images/segmented" --progress

# Modelo de segmentación
echo "[9] models/..."
rclone sync "$PROJECT_ROOT/models" "$DRIVE_ROOT/models" --progress

# Resultados del experimento (si se especificó --exp)
if [[ -n "$EXP" ]]; then
    echo "[10] experiments/$EXP/..."
    rclone sync "$PROJECT_ROOT/experiments/$EXP" "$DRIVE_ROOT/experiments/$EXP" --progress
fi

# Logs
echo "[11] logs/..."
rclone sync "$PROJECT_ROOT/logs" "$DRIVE_ROOT/logs" --progress

echo "Sincronización completada."
