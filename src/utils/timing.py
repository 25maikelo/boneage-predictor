"""
Utilidades de medición de tiempo de ejecución.
Cada script principal debe llamar a report_timing() al finalizar.
"""
import time
import datetime
from contextlib import contextmanager


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
