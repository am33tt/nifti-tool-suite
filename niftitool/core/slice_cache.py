"""LRU-bounded cache for windowed 2-D slices.

Slicing a volume is a cheap numpy view, but :func:`apply_window` allocates
and touches a new float32 buffer per frame (about 16 MiB for a 2000 by 2000
slice), which dominates triplanar slider interaction.

The cache key is ``(axis, idx, ww, wc, max_px)``. The cache must be
invalidated whenever the volume changes, for example on load, crop or
reorientation.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from ..deps import np
from .windowing import apply_window


class SliceCache:
    """Cache windowed 2-D slices extracted from a 3-D volume.

    Parameters
    ----------
    max_entries : int, default 48
        Maximum number of cached slices. Each entry is one float32 2-D
        array, so 48 entries of a 2000 by 2000 volume is 750 MiB at worst.
    """

    __slots__ = ("_vol", "_entries", "_max", "_lock")

    def __init__(self, max_entries: int = 48) -> None:
        self._vol = None
        self._entries: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._max = max_entries
        # get() runs on the GUI thread and the prefetch worker at the same
        # time, so OrderedDict mutation needs a lock.
        self._lock = threading.Lock()

    # Lifecycle

    def set_volume(self, gray) -> None:
        """Bind a new volume and clear the cache."""
        with self._lock:
            self._vol = gray
            self._entries.clear()

    def invalidate(self) -> None:
        """Drop all cached slices (but keep the bound volume)."""
        with self._lock:
            self._entries.clear()

    # Queries

    def get(self, axis: str, idx: int, ww: float, wc: float,
            max_px: int | None = None):
        """Return a windowed slice along *axis* at index *idx*.

        ``axis`` is one of ``'X'``, ``'Y'``, ``'Z'``. The returned array is
        transposed into the viewer orientation and is float32 in ``[0, 1]``.
        It is the live cache entry, so do not mutate it.

        ``max_px`` decimates the slice so its longest side is at most that
        many pixels. The viewer preserves data coordinates by drawing the
        decimated image with the original extent.
        """
        if self._vol is None:
            raise RuntimeError("SliceCache has no volume bound")

        key = (axis, int(idx), float(ww), float(wc), max_px)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
            g = self._vol

        if axis == 'X':
            raw = g[idx, :, :].T
        elif axis == 'Y':
            raw = g[:, idx, :].T
        else:
            raw = g[:, :, idx].T

        if max_px:
            step = int(np.ceil(max(raw.shape) / max_px))
            if step > 1:
                raw = raw[::step, ::step]

        windowed = apply_window(raw.astype(np.float32, copy=False), ww, wc)
        with self._lock:
            self._entries[key] = windowed
            if len(self._entries) > self._max:
                self._entries.popitem(last=False)  # evict oldest
        return windowed

    # Diagnostics

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SliceCache(entries={len(self._entries)}/{self._max})"
