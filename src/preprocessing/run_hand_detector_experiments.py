#!/usr/bin/env python3
"""
Corre en secuencia los experimentos del hand-detector con las configuraciones:
  - Exp A: imagenet,  trainable=False  (réplica _00)
  - Exp B: None,      trainable=False  (réplica _01)
  - Exp C: None,      trainable=True   (réplica _02)
  - Exp D: imagenet,  trainable=True   (réplica _05)

Uso:
    python src/preprocessing/run_hand_detector_experiments.py
"""
import os
import re
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_SEG   = os.path.join(PROJECT_ROOT, "config", "segmentation.py")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "preprocessing", "01_train_hand_detector.py")

EXPERIMENTS = [
    {"label": "A — imagenet / trainable=False", "encoder_weights": '"imagenet"', "trainable": "False"},
    {"label": "B — None    / trainable=False",  "encoder_weights": '"None"',     "trainable": "False"},
    {"label": "C — None    / trainable=True",   "encoder_weights": '"None"',     "trainable": "True"},
    {"label": "D — imagenet / trainable=True",  "encoder_weights": '"imagenet"', "trainable": "True"},
]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def set_encoder_weights(weights_value):
    """Modifica ENCODER_WEIGHTS en config/segmentation.py."""
    content = read(CONFIG_SEG)
    content = re.sub(
        r'(ENCODER_WEIGHTS\s*=\s*).*',
        rf'\g<1>{weights_value}   # None | "imagenet"',
        content,
    )
    write(CONFIG_SEG, content)


def set_base_model_trainable(trainable_value):
    """Modifica BASE_MODEL_TRAINABLE en 01_train_hand_detector.py."""
    content = read(TRAIN_SCRIPT)
    content = re.sub(
        r'(BASE_MODEL_TRAINABLE\s*=\s*).*',
        rf'\g<1>{trainable_value}',
        content,
    )
    write(TRAIN_SCRIPT, content)


def restore_originals(orig_seg, orig_train):
    write(CONFIG_SEG, orig_seg)
    write(TRAIN_SCRIPT, orig_train)


def main():
    orig_seg   = read(CONFIG_SEG)
    orig_train = read(TRAIN_SCRIPT)

    try:
        for i, exp in enumerate(EXPERIMENTS, 1):
            print(f"\n{'='*60}")
            print(f"  Experimento {i}/{len(EXPERIMENTS)}: {exp['label']}")
            print(f"{'='*60}\n")

            set_encoder_weights(exp["encoder_weights"])
            set_base_model_trainable(exp["trainable"])

            result = subprocess.run(
                [sys.executable, TRAIN_SCRIPT],
                cwd=PROJECT_ROOT,
            )

            if result.returncode != 0:
                print(f"\n[ERROR] Experimento {exp['label']} falló (código {result.returncode}). Abortando.")
                break
    finally:
        restore_originals(orig_seg, orig_train)
        print("\nConfiguración original restaurada.")


if __name__ == "__main__":
    main()
