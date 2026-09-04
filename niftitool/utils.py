"""Small cross-cutting helpers: RAM monitor, wall-clock timer."""

import os
import time
from contextlib import contextmanager

from .deps import HAS_PSUTIL, psutil


class OperationCancelled(Exception):
    """Raised inside progress callbacks when the user hits Stop."""


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

    Used as a soft budget. Callers facing a large allocation stay on the
    memory-mapped path when it would exceed the available headroom.
    """
    if not HAS_PSUTIL:
        return None
    try:
        return psutil.virtual_memory().available / (1024 ** 2)
    except Exception:
        return None


def total_ram_mb() -> float | None:
    """Installed system RAM in MiB. ``None`` if psutil is missing."""
    if not HAS_PSUTIL:
        return None
    try:
        return psutil.virtual_memory().total / (1024 ** 2)
    except Exception:
        return None


@contextmanager
def timed(label: str, logger=None):
    """Measure wall-clock duration of the enclosed block.

    If *logger* is callable it is invoked on exit with ``"<label>: <ms> ms"``.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if logger is not None:
            logger(f"{label}: {dt_ms:.1f} ms")
