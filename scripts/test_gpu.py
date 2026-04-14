#!/usr/bin/env python3
"""
Verifica la disponibilidad de GPU e imprime versión de TensorFlow y salida de nvidia-smi.
Uso: python scripts/test_gpu.py
"""
import tensorflow as tf
print("TF:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
import subprocess
print(subprocess.getoutput("nvidia-smi"))
