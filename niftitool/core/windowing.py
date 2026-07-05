"""Intensity windowing.

A *window* maps a linear intensity range ``[wc - ww/2, wc + ww/2]`` onto
``[0, 1]`` for display.  ``ww`` is the window width, ``wc`` the window
center — the same convention used everywhere in CT software.

The implementation is hot — it runs every time the user drags a slider —
so it takes pre-allocated output buffers where possible to avoid 16+ MB
allocations per frame on large volumes.
"""

from __future__ import annotations

from ..deps import np


def apply_window(gray, ww: float, wc: float, out=None):
    """Window a slice/volume into the range ``[0, 1]``.

    Parameters
    ----------
    gray : ndarray
        Any float array (typically a 2-D slice).
    ww, wc : float
        Window width and center. If ``ww == 0`` the output is all zeros.
    out : ndarray, optional
        Pre-allocated float32 array of the same shape, used as the
        destination to avoid an allocation.

    Returns
    -------
    ndarray
        Float32 array in ``[0, 1]``.
    """
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    if hi == lo:
        if out is None:
            return np.zeros_like(gray, dtype=np.float32)
        out.fill(0.0)
        return out
    scale = 1.0 / (hi - lo)
    # np.clip supports the `out=` keyword and is the main hot step here.
    if out is None:
        out = np.empty(gray.shape, dtype=np.float32)
    np.clip(gray, lo, hi, out=out)
    out -= lo
    out *= scale
    return out


def auto_window(gray) -> tuple[float, float]:
    """Sensible fallback window covering the data range.

    Sampled, never a full scan: on a multi-GB ndarray ``.min()`` /
    ``.max()`` touch every voxel (seconds per call) and this function
    used to sit directly on the slider hot path. A ~1M-voxel sample
    yields the same display window at a fraction of the cost.
    """
    if hasattr(gray, "subsample_flat"):          # LazyGrayVolume
        s = gray.subsample_flat(1_000_000)
        mn, mx = float(s.min()), float(s.max())
    elif getattr(gray, "size", 0) > 4_000_000:   # big ndarray → sample
        flat = gray.reshape(-1)
        s = flat[:: max(1, flat.size // 1_000_000)]
        mn, mx = float(s.min()), float(s.max())
    else:
        mn, mx = float(gray.min()), float(gray.max())
    return (mx - mn), (mn + mx) / 2.0
