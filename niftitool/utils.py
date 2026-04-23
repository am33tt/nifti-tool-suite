"""Small cross-cutting helpers: RAM monitor, wall-clock timer."""

import os
import time
from contextlib import contextmanager

from .deps import HAS_PSUTIL, psutil


def process_rss_mb() -> float | None:
    """Resident set size of the current process, in MiB. ``None`` if psutil is
    not installed or an unexpected error occurs."""
    if not HAS_PSUTIL:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def available_ram_mb() -> float | None:
    """System-wide free RAM in MiB. ``None`` if psutil is missing.

    Used as a soft budget: callers that are about to allocate a multi-GB
    float32 array check this and stay on the lazy (memory-mapped) path
    when they would blow past available headroom.
    """
    if not HAS_PSUTIL:
        return None
    try:
        return psutil.virtual_memory().available / (1024 ** 2)
    except Exception:
        return None


@contextmanager
def timed(label: str, logger=None):
    """Context manager that records wall-clock duration. If *logger* is a
    callable it is invoked with ``"<label>: <ms> ms"`` on exit.

    Useful for instrumenting hot paths without pulling in a profiler.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if logger is not None:
            logger(f"{label}: {dt_ms:.1f} ms")
