"""Automatic grey-level thresholding.

:func:`otsu_threshold` is the shared void/solid rule: the porosity
analysis, the background removal, the beam-hardening specimen mask and
the binarisation all call it, so a threshold quoted by one of them means
the same thing in the others.
"""

from __future__ import annotations

from ..deps import np


def otsu_threshold(values, nbins: int = 512) -> float:
    """Otsu threshold on a flat sample of intensities.

    Maximises between-class variance over a histogram, which suits bimodal
    CT data with an air peak and a material peak. Accepts raw intensities or
    HU and needs no calibration.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return float(centers[nbins // 2])
    p /= total
    omega = np.cumsum(p)                    # class-0 probability
    mu = np.cumsum(p * centers)             # class-0 cumulative mean
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    t = float(centers[int(np.argmax(sigma_b))])

    # Refine to the midpoint between the class means (isodata,
    # Ridler-Calvard). With well-separated peaks the Otsu optimum is flat
    # across the empty valley and argmax lands at its edge, while the
    # midpoint sits mid-valley and also captures partial-volume voxels
    # around small pores.
    for _ in range(20):
        lo = v[v < t]
        hi = v[v >= t]
        if lo.size == 0 or hi.size == 0:
            break
        t_new = 0.5 * (float(lo.mean()) + float(hi.mean()))
        if abs(t_new - t) <= 1e-6 * max(1.0, abs(t)):
            t = t_new
            break
        t = t_new
    return float(t)
