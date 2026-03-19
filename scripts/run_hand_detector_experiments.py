#!/usr/bin/env python3
"""
Ejecuta todos los experimentos del hand detector en secuencia.
Cada experimento sobreescribe config/segmentation.py, lanza el script
de entrenamiento como subproceso y restaura la configuración original al final.

Uso:
    # Prueba rápida (2 épocas, para verificar que todo se guarda correctamente)
    python run_hand_detector_experiments.py --quick

    # Entrenamiento completo (épocas definidas en cada config)
    python run_hand_detector_experiments.py

    # Solo los experimentos indicados (por índice 1-based)
    python run_hand_detector_experiments.py --only 3 5 7
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "segmentation.py"
SCRIPT_PATH  = PROJECT_ROOT / "src" / "preprocessing" / "01_train_hand_detector.py"

# Usa el Python del venv si existe, si no el que está corriendo este script
_venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

# ─── Matriz de experimentos ───────────────────────────────────────────────────
EXPERIMENTS = [
    # (arquitectura,          encoder_weights, base_model_trainable, canales, data_aug)
    ("unet_mobilenetv2",      "imagenet",      False,                3,       False),  # 00
    ("unet_mobilenetv2",      "imagenet",      False,                3,       True),   # 01
    ("unet_mobilenetv2",      "imagenet",      True,                 3,       False),  # 02
    ("unet_mobilenetv2",      "imagenet",      True,                 3,       True),   # 03
    ("unet_mobilenetv2",      None,            True,                 3,       False),  # 04
    ("unet_mobilenetv2",      None,            True,                 3,       True),   # 05
    ("unet_mobilenetv2",      None,            False,                3,       False),  # 06
    ("unet_mobilenetv2",      None,            False,                3,       True),   # 07
    ("unet_mobilenetv2",      "imagenet",      True,                 1,       False),  # 08
    ("unet_mobilenetv2",      "imagenet",      True,                 1,       True),   # 09
    ("unet",                  None,            None,                 3,       False),  # 10
    ("unet",                  None,            None,                 3,       True),   # 11
    ("unet",                  None,            None,                 1,       False),  # 12
    ("unet",                  None,            None,                 1,       True),   # 13
    ("mobilenetv2_sym",       "imagenet",      False,                3,       False),  # 14
    ("mobilenetv2_sym",       "imagenet",      False,                3,       True),   # 15
    ("mobilenetv2_sym",       "imagenet",      True,                 3,       False),  # 16
    ("mobilenetv2_sym",       "imagenet",      True,                 3,       True),   # 17
    ("mobilenetv2_sym",       None,            True,                 3,       False),  # 18
    ("mobilenetv2_sym",       None,            True,                 3,       True),   # 19
    ("mobilenetv2_sym",       "imagenet",      True,                 1,       False),  # 20
    ("mobilenetv2_sym",       "imagenet",      True,                 1,       True),   # 21
]

# ─── Plantilla de config/segmentation.py ─────────────────────────────────────
def render_config(arch, enc_weights, base_trainable, channels, data_aug, epochs, batch_size):
    enc_weights_str   = f'"{enc_weights}"' if enc_weights else "None"
    base_train_str    = str(base_trainable) if base_trainable is not None else "True"
    uses_backbone     = arch in ("unet_mobilenetv2", "mobilenetv2_sym")
    backbone_comment  = "" if uses_backbone else "  # ignorado para esta arquitectura"

    return dedent(f"""\
        \"\"\"
        Configuración generada automáticamente por run_hand_detector_experiments.py.
        Modifica aquí los parámetros antes de ejecutar:

            python src/preprocessing/01_train_hand_detector.py

        Las rutas de entrada/salida vienen de config/paths.py.
        \"\"\"

        # ─── Imagen ───────────────────────────────────────────────────────────────────
        IMAGE_SIZE  = (224, 224)

        # Número de canales de entrada: 1 → escala de grises  |  3 → RGB
        # Para arquitecturas con backbone MobileNetV2 ("unet_mobilenetv2", "mobilenetv2"),
        # si se elige INPUT_CHANNELS = 1 el canal único se replica 3 veces (tf.repeat)
        # antes del backbone, preservando la compatibilidad con pesos imagenet.
        # Para "unet" y "mobilenetv2_blocks" los canales se pasan directamente.
        INPUT_CHANNELS = {channels}

        NUM_CLASSES = 5           # 0=fondo, 1=pinky, 2=middle, 3=thumb, 4=wrist

        # ─── Entrenamiento ────────────────────────────────────────────────────────────
        BATCH_SIZE = {batch_size}
        EPOCHS     = {epochs}

        # ─── Arquitectura ─────────────────────────────────────────────────────────────
        # Opciones disponibles:
        #
        #   "unet"               → U-Net puro, sin backbone preentrenado.
        #                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE se IGNORAN.
        #
        #   "unet_mobilenetv2"   → U-Net con encoder MobileNetV2 y skip connections.
        #                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE aplican.
        #
        #   "mobilenetv2"        → MobileNetV2 encoder + decoder inverted-residual simétrico.
        #                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE aplican.
        #
        #   "mobilenetv2_blocks" → U-Net construida desde cero con bloques inverted-residual
        #                          estilo MobileNetV2 (DepthwiseSeparableConv).
        #                          ENCODER_WEIGHTS y BASE_MODEL_TRAINABLE se IGNORAN.
        #
        ARCHITECTURE = "{arch}"

        # Pesos del encoder.
        # Solo aplica si ARCHITECTURE es "unet_mobilenetv2" o "mobilenetv2".
        # Para "unet" y "mobilenetv2_blocks" siempre se entrena desde cero.
        ENCODER_WEIGHTS = {enc_weights_str}{backbone_comment}

        # Permite ajuste fino del backbone durante el entrenamiento.
        # Solo aplica si ARCHITECTURE es "unet_mobilenetv2" o "mobilenetv2".
        # Para "unet" y "mobilenetv2_blocks" este parámetro no tiene efecto.
        BASE_MODEL_TRAINABLE = {base_train_str}{backbone_comment}

        # ─── Data Augmentation ────────────────────────────────────────────────────────
        # Activa o desactiva el data augmentation en el conjunto de entrenamiento.
        DATA_AUGMENTATION = {data_aug}

        # Transformaciones geométricas — se aplican igual a imagen Y máscara.
        AUG_ROTATION_RANGE     = 15
        AUG_WIDTH_SHIFT_RANGE  = 0.1
        AUG_HEIGHT_SHIFT_RANGE = 0.1
        AUG_ZOOM_RANGE         = 0.1
        AUG_HORIZONTAL_FLIP    = True
        AUG_VERTICAL_FLIP      = False
        AUG_SHEAR_RANGE        = 0.05

        # Transformaciones fotométricas — solo se aplican a la imagen, NO a la máscara.
        AUG_BRIGHTNESS_RANGE = (0.8, 1.2)
    """)


def run_experiment(idx, arch, enc_weights, base_trainable, channels, data_aug, epochs, batch_size):
    label = (
        f"[{idx:02d}/{len(EXPERIMENTS)}] {arch} | "
        f"weights={enc_weights or 'None'} | "
        f"trainable={base_trainable} | "
        f"ch={channels} | "
        f"aug={data_aug}"
    )
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    config_content = render_config(arch, enc_weights, base_trainable, channels, data_aug, epochs, batch_size)
    CONFIG_PATH.write_text(config_content, encoding="utf-8")

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(SCRIPT_PATH)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"ERROR (código {result.returncode})"
    print(f"\n  >> {status}  |  tiempo: {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Prueba rápida: 2 épocas para verificar que todo fluye correctamente."
    )
    parser.add_argument(
        "--only", nargs="+", type=int, metavar="N",
        help="Ejecutar solo los experimentos indicados (índices 0-based, igual que hand-detector_NN)."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Sobreescribe el número de épocas (tiene prioridad sobre --quick)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size para todos los experimentos (default: 8)."
    )
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else (2 if args.quick else 30)

    original_config = CONFIG_PATH.read_text(encoding="utf-8")

    selected = (
        [EXPERIMENTS[i] for i in args.only]
        if args.only
        else EXPERIMENTS
    )
    indices = args.only if args.only else list(range(len(EXPERIMENTS)))

    print(f"\nExperimentos a ejecutar: {len(selected)}")
    print(f"Épocas por experimento:  {epochs}")
    print(f"Batch size:              {args.batch_size}")

    results = {}
    try:
        for idx, (arch, enc_weights, base_trainable, channels, data_aug) in zip(indices, selected):
            ok = run_experiment(
                idx, arch, enc_weights, base_trainable, channels, data_aug,
                epochs, args.batch_size,
            )
            results[idx] = ("OK" if ok else "ERROR", arch, enc_weights, base_trainable, channels, data_aug)
    finally:
        CONFIG_PATH.write_text(original_config, encoding="utf-8")
        print("\nConfig original restaurada.")

    print(f"\n{'='*70}")
    print("  RESUMEN")
    print(f"{'='*70}")
    for idx, (status, arch, enc_w, trainable, ch, aug) in results.items():
        print(f"  [{idx:02d}] {status:5s}  {arch:20s}  weights={str(enc_w):8s}  trainable={trainable}  ch={ch}  aug={aug}")

    failed = [i for i, r in results.items() if r[0] == "ERROR"]
    if failed:
        print(f"\nExperimentos fallidos: {failed}")
        sys.exit(1)
    else:
        print(f"\nTodos los experimentos completados correctamente.")


if __name__ == "__main__":
    main()
