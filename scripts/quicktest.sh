#!/usr/bin/env bash
# ============================================================
# Quick Test — Pipeline completo con datos mínimos (exp. 99)
# Prueba cada script del pipeline de forma rápida (~10-20 min)
# Uso:
#   bash scripts/quicktest.sh            # experimento 99 (por defecto)
#   bash scripts/quicktest.sh 99
# ============================================================

set -euo pipefail

EXP=${1:-99}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/experiments/$EXP/quicktest_log"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/quicktest_$(date +%Y%m%d_%H%M%S).log"

# ── Logger ───────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }
fail() { log "ERROR: $*"; exit 1; }

log "=========================================="
log "  QUICK TEST — experimento $EXP"
log "  Directorio: $PROJECT_ROOT"
log "=========================================="

# ── Verificar entorno ────────────────────────────────────────
python - <<'PYCHECK'
import tensorflow as tf, cv2, pandas, sklearn, scipy
gpus = tf.config.list_physical_devices('GPU')
print(f"TF {tf.__version__} | GPUs: {gpus if gpus else 'ninguna (CPU)'}")
PYCHECK

# ── Paso 05: Análisis de dataset ─────────────────────────────
log "--- Paso 05: dataset analysis ---"
python src/05_dataset_analysis.py --experiment $EXP \
    2>&1 | tee -a "$LOGFILE" || fail "05_dataset_analysis falló"

# ── Paso 06: Entrenamiento ───────────────────────────────────
log "--- Paso 06: training ---"
python src/06_training.py --experiment $EXP \
    2>&1 | tee -a "$LOGFILE" || fail "06_training falló"

# ── Paso 07: Validación estándar ─────────────────────────────
log "--- Paso 07: validation ---"
python src/07_validation.py --experiment $EXP \
    2>&1 | tee -a "$LOGFILE" || fail "07_validation falló"

# ── Paso 08: Validación mexicana ─────────────────────────────
log "--- Paso 08: mex-validation ---"
python src/08_mex_validation.py --experiment $EXP \
    2>&1 | tee -a "$LOGFILE" || fail "08_mex_validation falló"

# ── Paso 09: Análisis de desempeño ───────────────────────────
log "--- Paso 09: performance analysis ---"
python src/09_performance_analysis.py --experiment $EXP \
    2>&1 | tee -a "$LOGFILE" || fail "09_performance_analysis falló"

# ── Generación de gráficos bilingüe ──────────────────────────
log "--- Generando gráficos en inglés ---"
python scripts/generate_plots.py --experiment $EXP --lang en \
    2>&1 | tee -a "$LOGFILE" || log "AVISO: generate_plots falló (no crítico)"

# ── Resumen de resultados ────────────────────────────────────
log "=========================================="
log "  QUICK TEST COMPLETADO"
log "  Experimento: $EXP"
log "  Resultados en: experiments/$EXP/"
log "  Log: $LOGFILE"
log "=========================================="

# Listar archivos generados
log "Archivos generados:"
find "experiments/$EXP" -type f | sort | tee -a "$LOGFILE"
