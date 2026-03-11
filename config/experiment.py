"""
Utilidad para cargar la configuración de un experimento.
Cada experimento vive en experiments/<idx>/config.py y define
exclusivamente hiperparámetros; las rutas vienen de config/paths.py.
"""
import os
import sys
import importlib.util

from config.paths import EXPERIMENTS_DIR


def load_experiment_config(experiment_idx: int):
    """
    Carga y retorna el módulo config.py del experimento indicado.

    Uso:
        cfg = load_experiment_config(26)
        print(cfg.BATCH_SIZE)
    """
    exp_dir = os.path.join(EXPERIMENTS_DIR, str(experiment_idx).zfill(2))
    config_path = os.path.join(exp_dir, "config.py")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No se encontró config.py en: {config_path}\n"
            f"Crea el archivo en experiments/{str(experiment_idx).zfill(2)}/config.py"
        )

    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["experiment_config"] = cfg
    spec.loader.exec_module(cfg)
    return cfg


def get_experiment_output_dir(experiment_idx: int) -> str:
    """Devuelve el directorio de salida del experimento (se crea si no existe)."""
    path = os.path.join(EXPERIMENTS_DIR, str(experiment_idx).zfill(2))
    os.makedirs(path, exist_ok=True)
    for sub in ("models", "training_history", "evaluation", "validation", "mex-validation"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path