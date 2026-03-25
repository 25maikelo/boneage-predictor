#!/usr/bin/env python3
"""
Descarga los pesos preentrenados de RadImageNet (Mei et al. 2022) para
ResNet50 y DenseNet121, necesarios para los experimentos 22–33.

Repositorio: https://github.com/BMEII-AI/RadImageNet

Uso:
    python download_radimagenet_weights.py
    python download_radimagenet_weights.py --force   # re-descarga aunque existan
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pretrained")

# Archivo de Google Drive con los pesos TensorFlow/Keras de RadImageNet.
# Fuente: https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc/view
FILE_ID  = "1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Mapeo: patrones en el nombre del archivo descargado → nombre esperado por el proyecto.
NAME_MAP = [
    (["resnet50", "resnet_50"],         "radimagenet_resnet50.h5"),
    (["densenet121", "densenet_121"],   "radimagenet_densenet121.h5"),
]

TARGET_FILES = [dest for _, dest in NAME_MAP]


def ensure_gdown():
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        print("Instalando gdown...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
            capture_output=True,
        )
        if result.returncode != 0:
            print("ERROR: No se pudo instalar gdown.")
            print(result.stderr.decode())
            return False
        return True


def already_exist():
    return [f for f in TARGET_FILES if os.path.exists(os.path.join(DEST_DIR, f))]


def find_and_rename(search_dir):
    """Busca archivos .h5 y los mueve a DEST_DIR con el nombre canónico."""
    found = glob.glob(os.path.join(search_dir, "**", "*.h5"), recursive=True)
    mapped = []
    for src in found:
        basename = os.path.basename(src).lower()
        for patterns, dest_name in NAME_MAP:
            if any(p in basename for p in patterns):
                dest = os.path.join(DEST_DIR, dest_name)
                shutil.move(src, dest)
                size_mb = os.path.getsize(dest) / (1024 ** 2)
                print(f"  {os.path.basename(src)} -> {dest_name} ({size_mb:.1f} MB)")
                mapped.append(dest_name)
                break
        else:
            print(f"  (sin mapeo para '{os.path.basename(src)}' — muévelo manualmente si es necesario)")
    return mapped


def print_manual_instructions():
    print(
        "\nDescarga manual:\n"
        "  1. Abre: https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc/view\n"
        "  2. Descarga el archivo y extrae si es un .zip\n"
        f"  3. Renombra los .h5 y colócalos en: {DEST_DIR}\n"
        "       radimagenet_resnet50.h5\n"
        "       radimagenet_densenet121.h5"
    )


def main():
    parser = argparse.ArgumentParser(description="Descarga pesos RadImageNet")
    parser.add_argument("--force", action="store_true",
                        help="Re-descarga aunque el archivo ya exista.")
    args = parser.parse_args()

    os.makedirs(DEST_DIR, exist_ok=True)

    existing = already_exist()
    if not args.force:
        for f in existing:
            size_mb = os.path.getsize(os.path.join(DEST_DIR, f)) / (1024 ** 2)
            print(f"[OK] {f} ya existe ({size_mb:.1f} MB) — omitiendo.")
        missing = [f for f in TARGET_FILES if f not in existing]
        if not missing:
            print("\nTodos los pesos ya están disponibles.")
            return
    else:
        missing = TARGET_FILES

    print(f"\nArchivos necesarios: {missing}")

    if not ensure_gdown():
        print_manual_instructions()
        sys.exit(1)

    import gdown

    tmp_dir = os.path.join(DEST_DIR, "_tmp_radimagenet")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, "radimagenet_download")

    try:
        print(f"\nDescargando desde Google Drive...")
        print(f"  https://drive.google.com/file/d/{FILE_ID}/view")
        gdown.download(FILE_URL, tmp_file, quiet=False, fuzzy=True)

        if not os.path.exists(tmp_file):
            raise RuntimeError("La descarga no produjo ningún archivo.")

        # Detectar si es un zip o un .h5 directo
        if zipfile.is_zipfile(tmp_file):
            print("\nArchivo zip detectado — extrayendo...")
            with zipfile.ZipFile(tmp_file, "r") as zf:
                zf.extractall(tmp_dir)
            os.remove(tmp_file)
            find_and_rename(tmp_dir)
        else:
            # Asumir que es un .h5; intentar mapear por nombre o mover con nombre genérico
            basename = os.path.basename(tmp_file).lower()
            mapped = False
            for patterns, dest_name in NAME_MAP:
                if any(p in basename for p in patterns):
                    dest = os.path.join(DEST_DIR, dest_name)
                    shutil.move(tmp_file, dest)
                    size_mb = os.path.getsize(dest) / (1024 ** 2)
                    print(f"  -> {dest_name} ({size_mb:.1f} MB)")
                    mapped = True
                    break
            if not mapped:
                # Nombre sin pista — guardarlo con nombre temporal para inspección
                raw_dest = os.path.join(DEST_DIR, "radimagenet_download.h5")
                shutil.move(tmp_file, raw_dest)
                size_mb = os.path.getsize(raw_dest) / (1024 ** 2)
                print(f"\nArchivo .h5 descargado ({size_mb:.1f} MB) pero su nombre no permite")
                print(f"identificar la arquitectura automáticamente.")
                print(f"Guardado como: {raw_dest}")
                print("Renómbralo manualmente a:")
                print("  radimagenet_resnet50.h5   o   radimagenet_densenet121.h5")

    except Exception as e:
        print(f"\nERROR durante la descarga: {e}")
        print_manual_instructions()
        sys.exit(1)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    final_existing = already_exist()
    still_missing = [f for f in TARGET_FILES if f not in final_existing]

    if still_missing:
        print(f"AVISO: aún faltan: {still_missing}")
        print_manual_instructions()

    if final_existing:
        print("Pesos disponibles en:")
        for f in final_existing:
            size_mb = os.path.getsize(os.path.join(DEST_DIR, f)) / (1024 ** 2)
            print(f"  {os.path.join(DEST_DIR, f)} ({size_mb:.1f} MB)")

    if not still_missing:
        print("\nPuedes ejecutar los experimentos RadImageNet con:")
        print("  python scripts/run_hand_detector_experiments.py --only 22 23 24 25 26 27 28 29 30 31 32 33")


if __name__ == "__main__":
    main()
