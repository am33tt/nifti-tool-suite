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


def _axis_len(shape, axis: str) -> int:
    return shape[{'X': 0, 'Y': 1, 'Z': 2}[axis]]


def _plane_shape(shape, axis: str) -> tuple:
    """Shape of one slice along *axis*, before transposing for display."""
    return {
        'X': (shape[1], shape[2]),
        'Y': (shape[0], shape[2]),
        'Z': (shape[0], shape[1]),
    }[axis]


def _read_plane(volume, axis: str, idx: int, step: int):
    """One slice, decimated by *step* as early as the source allows.

    :class:`~niftitool.core.io.LazyGrayVolume` can decimate before it
    converts RGB channels to a scalar, which is most of the per-frame cost
    on an RGB8 scan. A plain ndarray is sliced and strided directly.
    """
    key = {
        'X': (idx, slice(None), slice(None)),
        'Y': (slice(None), idx, slice(None)),
        'Z': (slice(None), slice(None), idx),
    }[axis]
    reader = getattr(volume, "read_scalar", None)
    if reader is not None:
        return reader(key, step)
    plane = np.asarray(volume[key])
    return plane[::step, ::step] if step > 1 else plane


class SliceCache:
    """Cache windowed 2-D slices extracted from a 3-D volume.

    Parameters
    ----------
    max_entries : int, default 48
        Maximum number of cached slices. Each entry is one float32 2-D
        array, so 48 entries of a 2000 by 2000 volume is 750 MiB at worst.
    """

    __slots__ = ("_vol", "_preview", "_preview_step", "_entries", "_max",
                 "_lock")

    def __init__(self, max_entries: int = 48) -> None:
        self._vol = None
        self._preview = None
        self._preview_step = 1
        self._entries: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._max = max_entries
        # get() runs on the GUI thread and the prefetch worker at the same
        # time, so OrderedDict mutation needs a lock.
        self._lock = threading.Lock()

    # Lifecycle

    def set_volume(self, gray) -> None:
        """Bind a new volume and clear the cache and any preview."""
        with self._lock:
            self._vol = gray
            self._preview = None
            self._preview_step = 1
            self._entries.clear()

    def set_preview(self, array, step: int) -> None:
        """Bind a resident decimated copy for interactive reads.

        A slice along X or Y is strided across the whole file: every plane
        has to be touched to collect one row from each. On a volume larger
        than the page cache that is the single dominant cost of dragging a
        slider, and no amount of caching windowed results helps, because
        each new index pays it again.

        A preview is one strided pass over the file, held in memory, at the
        resolution the panels actually display. Reads from it are memory
        reads: microseconds instead of the tens or hundreds of
        milliseconds a strided file read costs.
        """
        with self._lock:
            self._preview = array
            self._preview_step = max(1, int(step))
            self._entries.clear()

    def has_preview(self) -> bool:
        return self._preview is not None

    @property
    def preview_step(self) -> int:
        return self._preview_step

    def invalidate(self) -> None:
        """Drop all cached slices (but keep the bound volume and preview)."""
        with self._lock:
            self._entries.clear()

    # Queries

    def get(self, axis: str, idx: int, ww: float, wc: float,
            max_px: int | None = None, preview: bool = False):
        """Return a windowed slice along *axis* at index *idx*.

        ``axis`` is one of ``'X'``, ``'Y'``, ``'Z'``. The returned array is
        transposed into the viewer orientation and is float32 in ``[0, 1]``.
        It is the live cache entry, so do not mutate it.

        ``max_px`` decimates the slice so its longest side is at most that
        many pixels. The viewer preserves data coordinates by drawing the
        decimated image with the original extent.

        ``preview`` reads from the resident decimated copy bound by
        :meth:`set_preview`, when there is one. *idx* stays a
        full-resolution index either way.
        """
        if self._vol is None:
            raise RuntimeError("SliceCache has no volume bound")

        key = (axis, int(idx), float(ww), float(wc), max_px, bool(preview))
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
            g = self._preview if (preview and self._preview is not None) \
                else self._vol
            source_step = self._preview_step if (
                preview and self._preview is not None) else 1

        # A preview index is the full-resolution index divided by its step,
        # so callers keep passing full-resolution indices throughout. The
        # division floors: the preview holds only every step-th plane, so
        # an index between two of them shows the one at or below it, up to
        # step - 1 voxels away. Rounding to the nearer plane would halve
        # that at the cost of an off-by-one at the last plane, which is not
        # worth it -- the panel lands on the exact plane anyway once the
        # drag settles and the finer read arrives.
        read_idx = int(idx) // source_step
        plane_shape = _plane_shape(g.shape, axis)
        read_idx = max(0, min(read_idx, _axis_len(g.shape, axis) - 1))

        step = 1
        if max_px:
            step = max(1, int(np.ceil(max(plane_shape) / max_px)))

        raw = _read_plane(g, axis, read_idx, step).T
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
