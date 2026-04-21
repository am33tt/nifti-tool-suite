"""LRU-bounded cache for windowed 2-D slices.

Why this exists
---------------
Every time the user drags a triplanar slider, the viewer needs a
windowed, float32, ``[0, 1]`` 2-D slice to hand to matplotlib. Recomputing
that per frame is wasteful:

* Slicing the volume is a numpy *view* → cheap.
* But :func:`apply_window` allocates a fresh ``float32`` buffer of the
  slice size, does a ``clip`` and a ``subtract`` and a ``multiply``.
  For a 2000 × 2000 slice that's ~16 MiB touched per frame.

The user rarely looks at more than ~5 distinct slice indices per axis
per window setting while inspecting a volume. Caching those lets us
serve repeat frames from RAM and keep the interaction at native monitor
refresh rate.

Key is ``(axis, idx, ww, wc)``. The cache is invalidated whenever the
volume itself changes (new file loaded, crop applied, reorient, …).
"""

from __future__ import annotations

from collections import OrderedDict

from ..deps import np
from .windowing import apply_window


class SliceCache:
    """Cache windowed 2-D slices extracted from a 3-D volume.

    Parameters
    ----------
    max_entries : int, default 48
        Maximum number of cached slices. Each entry is one float32 2-D
        array; 48 on a 2000² volume is ~750 MiB at worst. Tune to taste.
    """

    __slots__ = ("_vol", "_entries", "_max")

    def __init__(self, max_entries: int = 48) -> None:
        self._vol = None
        self._entries: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._max = max_entries

    # ── lifecycle ────────────────────────────────────────────────────────────

    def set_volume(self, gray) -> None:
        """Bind a new volume and clear the cache."""
        self._vol = gray
        self._entries.clear()

    def invalidate(self) -> None:
        """Drop all cached slices (but keep the bound volume)."""
        self._entries.clear()

    # ── queries ──────────────────────────────────────────────────────────────

    def get(self, axis: str, idx: int, ww: float, wc: float):
        """Return a windowed slice along *axis* at index *idx*.

        ``axis`` is one of ``'X'``, ``'Y'``, ``'Z'``. The returned array
        is already transposed into the same orientation the viewer uses
        (``.T``) and in ``[0, 1]`` float32. Do *not* mutate it — it is
        the actual cache entry.
        """
        if self._vol is None:
            raise RuntimeError("SliceCache has no volume bound")

        key = (axis, int(idx), float(ww), float(wc))
        entry = self._entries.get(key)
        if entry is not None:
            # Mark as recently used
            self._entries.move_to_end(key)
            return entry

        g = self._vol
        if axis == 'X':
            raw = g[idx, :, :].T
        elif axis == 'Y':
            raw = g[:, idx, :].T
        else:
            raw = g[:, :, idx].T

        windowed = apply_window(raw.astype(np.float32, copy=False), ww, wc)
        self._entries[key] = windowed
        if len(self._entries) > self._max:
            self._entries.popitem(last=False)  # evict oldest
        return windowed

    # ── diagnostics ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SliceCache(entries={len(self._entries)}/{self._max})"
