"""Volume histogram + per-slice mean.

Large volumes (say 2000³) would produce a 10¹⁰-element flat array if we
naively called :func:`numpy.histogram`; the :data:`MAX_HIST_VOXELS`
budget in :mod:`niftitool.config` caps the sample by subsampling the
flattened array with a fixed stride.
"""

from __future__ import annotations

from ..config import MAX_HIST_VOXELS
from ..deps import np


def compute_histogram(gray, n_bins: int = 256):
    """Return ``(zidx, means, counts, edges, mn, mx)`` for plotting.

    * ``counts, edges`` — histogram of the (possibly subsampled) voxels.
    * ``zidx, means``   — mean intensity along Z, evaluated on a stride
      that produces at most ~80 points (so the slice-mean curve is legible
      even on 1000-slice volumes).
    """
    flat = gray.ravel()
    if flat.size > MAX_HIST_VOXELS:
        step = flat.size // MAX_HIST_VOXELS
        flat = flat[::step]
    mn, mx = float(flat.min()), float(flat.max())
    counts, edges = np.histogram(flat, bins=n_bins, range=(mn, mx))
    nz   = gray.shape[2]
    zidx = list(range(0, nz, max(1, nz // 80)))
    means = [float(gray[:, :, z].mean()) for z in zidx]
    return zidx, means, counts, edges, mn, mx
