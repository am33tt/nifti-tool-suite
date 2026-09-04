"""Volume histogram and per-slice mean.

Voxels are subsampled with a fixed stride to the :data:`MAX_HIST_VOXELS`
budget in :mod:`niftitool.config` before histogramming.
"""

from __future__ import annotations

from ..config import MAX_HIST_VOXELS
from ..deps import np


def compute_histogram(gray, n_bins: int = 256):
    """Return ``(zidx, means, counts, edges, mn, mx)`` for plotting.

    ``counts, edges`` is the histogram of the subsampled voxels and
    ``zidx, means`` the mean intensity along Z, on a stride giving at most
    80 points.
    """
    if hasattr(gray, "subsample_flat"):
        # Lazy volume: sample without materialising the full array.
        flat = gray.subsample_flat(MAX_HIST_VOXELS)
    else:
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
