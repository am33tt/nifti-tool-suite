"""A resident decimated copy of the volume, for interactive viewing.

Why this exists
---------------
The tri-planar viewer draws each panel a few hundred pixels across, so it
already decimates every slice before it reaches the screen. What it could
not avoid was *reading* the slice: an axial slice is contiguous on disk and
costs one sequential read, but a sagittal or coronal slice takes one row
from every plane in the file. Collecting it touches the whole volume. On a
scan larger than the page cache that is the entire cost of dragging a
slider, and it is paid again at every new index, so caching windowed
results does not help.

This module reads the volume once, decimated, into an array that stays in
memory. After that a slice along any axis is a memory read -- microseconds
rather than the tens or hundreds of milliseconds a strided file read costs
-- which is what makes a slider feel continuous. It is the same trick the
3-D view already relies on: that view is smooth because its volume is
resident, not because rendering is cheap.

Reading it cheaply
------------------
The pass is by whole planes along the slowest axis, which are contiguous on
disk, and only every *step*-th plane is read at all. So building a preview
at step 3 reads about a third of the file, sequentially, rather than
seeking through all of it. Peak memory is one plane plus the finished
preview.

Fidelity
--------
A preview is for interaction only. It is decimated, so it must never be the
source of a measurement, a threshold, a histogram or an exported image; the
viewer re-reads at full resolution once a drag settles. Nothing here writes
to disk.
"""

from __future__ import annotations

import math

from ..deps import np
from ..utils import OperationCancelled, available_ram_mb

__all__ = ["choose_step", "estimate_bytes", "build_preview"]

#: Longest side, in voxels, a preview is built to. Matched to the largest
#: panel a window is likely to show; the viewer decimates to its own budget
#: on top of this.
DEFAULT_TARGET_PX = 768

#: Fraction of free RAM a preview may occupy. Deliberately small: the whole
#: point of the tool is to work on volumes larger than memory, so the
#: preview must never be the thing that fills it.
RAM_FRACTION = 0.15

#: A preview smaller than this is not worth the pass over the file -- the
#: volume is already small enough that strided reads are cheap.
MIN_USEFUL_VOXELS = 8 ** 3


def choose_step(shape, target_px: int = DEFAULT_TARGET_PX,
                max_bytes: int | None = None) -> int:
    """Decimation step for *shape*, honouring a memory budget.

    Starts from the step that brings the longest side down to *target_px*
    and increases it until the result fits *max_bytes* (by default a small
    fraction of free RAM). Returns 1 when the volume is already small
    enough that a preview would not earn its keep.
    """
    dims = [int(d) for d in shape[:3]]
    if min(dims) <= 0:
        return 1
    step = max(1, math.ceil(max(dims) / max(1, int(target_px))))

    if max_bytes is None:
        free_mb = available_ram_mb()
        max_bytes = int(RAM_FRACTION * free_mb * 1024 ** 2) if free_mb else None

    if max_bytes:
        while estimate_bytes(dims, step) > max_bytes and step < 64:
            step += 1

    if step == 1:
        return 1
    if _preview_voxels(dims, step) < MIN_USEFUL_VOXELS:
        return 1
    return step


def _preview_voxels(shape, step: int) -> int:
    voxels = 1
    for dim in shape[:3]:
        voxels *= len(range(0, int(dim), step))
    return voxels


def estimate_bytes(shape, step: int) -> int:
    """Bytes a float32 preview of *shape* at *step* would occupy."""
    return _preview_voxels(shape, step) * 4


def build_preview(gray, *, target_px: int = DEFAULT_TARGET_PX,
                  max_bytes: int | None = None, progress=None, cancel=None):
    """Read *gray* once, decimated, into a resident float32 array.

    Parameters
    ----------
    gray
        The scalar volume: a :class:`~niftitool.core.io.LazyGrayVolume` or
        any ndarray-like with ``shape`` and ``__getitem__``.
    target_px, max_bytes
        Passed to :func:`choose_step`.
    progress
        Optional ``fn(fraction)`` called as planes are read.
    cancel
        Optional ``fn() -> bool``; raises
        :class:`~niftitool.utils.OperationCancelled` when it turns true.

    Returns ``(array, step)``, or ``(None, 1)`` when a preview would not
    help -- the volume is already small, or the budget cannot hold a useful
    one. Callers treat that as "keep reading the file directly".
    """
    shape = tuple(int(d) for d in gray.shape[:3])
    step = choose_step(shape, target_px=target_px, max_bytes=max_bytes)
    if step == 1:
        return None, 1

    z_indices = range(0, shape[2], step)
    out_shape = (len(range(0, shape[0], step)),
                 len(range(0, shape[1], step)),
                 len(z_indices))
    out = np.empty(out_shape, dtype=np.float32)

    reader = getattr(gray, "read_scalar", None)
    total = max(1, out_shape[2])
    for out_z, z in enumerate(z_indices):
        if cancel is not None and cancel():
            raise OperationCancelled("preview cancelled")
        key = (slice(None), slice(None), z)
        if reader is not None:
            # Decimates before the RGB-to-scalar conversion, so the
            # arithmetic runs on the voxels that are kept.
            plane = reader(key, step)
        else:
            plane = np.asarray(gray[key])[::step, ::step]
        # A plane can come back one row or column longer than the output
        # when a dimension is not a whole multiple of the step.
        out[:, :, out_z] = plane[:out_shape[0], :out_shape[1]]
        if progress is not None:
            progress((out_z + 1) / total)

    return out, step
