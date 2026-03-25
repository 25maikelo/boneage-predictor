"""
Utilidades de medición de tiempo de ejecución y logging.
Cada script principal debe llamar a setup_logging() y report_timing().
"""
import os
import sys
import time
import datetime
from contextlib import contextmanager


# ============================================================
# LOGGING A ARCHIVO
# ============================================================

class _Tee:
    """Escribe simultáneamente en la consola y en un archivo de log."""

    def __init__(self, stream, filepath: str):
        import io
        self._stream = io.TextIOWrapper(
            stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
        ) if hasattr(stream, "buffer") else stream
        self._file = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Delegar atributos como fileno, isatty, etc.
    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging(script_name: str, log_dir: str = None) -> str:
    """
    Redirige stdout y stderr a un archivo de log además de la consola.

    Si log_dir es None guarda en logs/<script_name>_YYYYMMDD_HHMMSS.log.
    Si log_dir se especifica, guarda el log dentro de ese directorio
    (útil para asociar el log al run_dir del experimento).

    Retorna la ruta del archivo de log creado.
    """
    if log_dir is None:
        from config.paths import LOGS_DIR
        log_dir = LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)

    name = os.path.splitext(script_name)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    sys.stdout = _Tee(sys.stdout, log_path)
    sys.stderr = _Tee(sys.stderr, log_path)

    print(f"[LOG] {log_path}")
    return log_path


# ============================================================
# TIMING
# ============================================================

@contextmanager
def timer(name: str = ""):
    """Context manager para medir un bloque de código."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMING] {name}: {_fmt(elapsed)}")


def report_timing(start_time: float, script_name: str = ""):
    """Imprime el reporte de tiempo total al finalizar un script."""
    elapsed = time.time() - start_time
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"[TIMING REPORT] {script_name}")
    print(f"Tiempo total de ejecución: {_fmt(elapsed)}")
    print(f"Finalizado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)


def _fmt(seconds: float) -> str:
    """Formatea segundos como H:MM:SS."""
    return str(datetime.timedelta(seconds=int(seconds)))
