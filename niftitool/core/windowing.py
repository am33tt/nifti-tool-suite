"""Intensity windowing.

A window maps the intensity range ``[wc - ww/2, wc + ww/2]`` onto
``[0, 1]`` for display; ``ww`` is the window width and ``wc`` the window
center (standard CT convention). Pre-allocated output buffers are
accepted to avoid a per-frame allocation on large volumes.
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
    if out is None:
        out = np.empty(gray.shape, dtype=np.float32)
    np.clip(gray, lo, hi, out=out)
    out -= lo
    out *= scale
    return out


def _display_sample(gray, budget: int = 1_000_000):
    """Finite values from *gray*, subsampled to roughly *budget* entries.

    Exact statistics touch every voxel, which is prohibitive on the slider
    hot path for multi-GB volumes.
    """
    if hasattr(gray, "subsample_flat"):          # LazyGrayVolume
        sample = np.asarray(gray.subsample_flat(budget), dtype=np.float32)
    else:
        flat = np.asarray(gray).reshape(-1)
        step = max(1, flat.size // budget)
        sample = np.asarray(flat[::step], dtype=np.float32)
    if sample.size and not np.isfinite(sample).all():
        sample = sample[np.isfinite(sample)]
    return sample


def _otsu_split(sample, lo: float, hi: float, bins: int = 256):
    """Otsu threshold over ``[lo, hi]``, with both class means."""
    hist, edges = np.histogram(sample, bins=bins, range=(float(lo), float(hi)))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(lo), 0.0, 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_lo = np.cumsum(hist)
    weight_hi = total - weight_lo
    cum_mean = np.cumsum(hist * centers)
    mean_lo = cum_mean / np.maximum(weight_lo, 1.0)
    mean_hi = (cum_mean[-1] - cum_mean) / np.maximum(weight_hi, 1.0)
    between = weight_lo * weight_hi * (mean_lo - mean_hi) ** 2
    k = int(np.argmax(between))
    return float(centers[k]), float(mean_lo[k]), float(mean_hi[k])


def auto_window(gray, k_lo: float = 3.0, k_hi: float = 1.0) -> tuple[float, float]:
    """Display window (ww, wc) covering the specimen, not the whole range.

    A min-max window is unusable on CT data: a few hot outliers, or a
    background the reconstruction pinned to zero, stretch the ramp until the
    material occupies a few percent of it and every slice reads as flat
    grey. Beam-hardening-corrected volumes have both.

    The window is therefore built from the material alone. Otsu splits off
    the darkest population (air, background) and the remainder is bracketed
    around its median by its own upper spread: wide below, so pores and
    edges keep their gradient, and narrow above, where CT intensities are
    tightly packed.
    """
    sample = _display_sample(gray)
    if sample.size == 0:
        return 1.0, 0.0

    p_lo, p_hi = (float(v) for v in np.percentile(sample, (1.0, 99.0)))
    if p_hi <= p_lo:
        # A specimen small enough that both percentiles land in the
        # background, so histogram the full range instead.
        p_lo, p_hi = float(sample.min()), float(sample.max())

    material = sample
    if p_hi > p_lo:
        threshold, mean_lo, mean_hi = _otsu_split(sample, p_lo, p_hi)
        upper = sample[sample > threshold]
        # Split only when the two populations are separated. Judging by
        # class separation rather than by the size of the upper population
        # also covers a small specimen in a wide field of view.
        if (mean_hi - mean_lo) > 0.15 * (p_hi - p_lo) and upper.size >= 64:
            material = upper

    median, top = (float(v) for v in np.percentile(material, (50.0, 99.5)))
    spread = top - median
    if spread <= 0:
        mn, mx = float(sample.min()), float(sample.max())
        if mx > mn:
            return mx - mn, (mx + mn) / 2.0
        return 1.0, mn

    lo = median - k_lo * spread
    hi = median + k_hi * spread
    return hi - lo, (hi + lo) / 2.0
