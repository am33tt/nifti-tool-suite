"""Threshold-based phase segmentation and per-phase statistics.

Three-phase labelling for concrete-like volumes:

* **Void / air**   — HU below ``void_thresh``
* **Matrix**       — HU between ``void_thresh`` and ``solid_thresh``
* **Aggregate**    — HU above ``solid_thresh`` (when ``solid_thresh`` is given)

If ``solid_thresh`` is ``None`` the volume is simply binarised into
void vs solid.

The statistics returned by :func:`phase_statistics` match the columns of
the CSV exporter in :mod:`niftitool.core.cpp_export` so the UI and the
file output never disagree.
"""

from __future__ import annotations

from ..deps import np


PHASE_NAMES = {0: "Void/Air", 1: "Matrix/Paste", 2: "Aggregate/Solid"}


def otsu_threshold(values, nbins: int = 512) -> float:
    """Classic Otsu threshold on a flat sample of intensities.

    Maximises between-class variance over a histogram — a good automatic
    void/solid split for bimodal CT data (air peak vs material peak).
    Works on raw intensities or HU alike; no calibration required.
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

    # Refine to the midpoint between the class means (isodata /
    # Ridler–Calvard).  With well-separated peaks the raw Otsu optimum is
    # flat across the whole empty valley and argmax lands at its very
    # edge — e.g. "threshold 25" on data whose material peak sits in the
    # thousands.  The midpoint lands mid-valley, which also captures
    # partial-volume voxels around small pores.
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


def segment_phases(hu_vol, void_thresh, solid_thresh=None):
    """Return a ``uint8`` label volume using the two thresholds above."""
    labels = np.zeros(hu_vol.shape, dtype=np.uint8)
    labels[hu_vol >= void_thresh] = 1
    if solid_thresh is not None:
        labels[hu_vol >= solid_thresh] = 2
    return labels


def phase_statistics(hu_vol, E_vol, labels):
    """Return ``(stats_dict, porosity)``.

    ``stats_dict`` has one entry per *occupied* phase, keyed by the
    phase's human-readable name.  Each entry includes voxel count,
    volume fraction, HU mean/std, and E min/mean/max/std in MPa.

    Porosity is extracted as the volume fraction of the void phase, or
    0.0 if no void voxels exist.
    """
    total = hu_vol.size
    stats: dict = {}
    for pid, name in PHASE_NAMES.items():
        mask = labels == pid
        count = int(mask.sum())
        if count == 0:
            continue
        E_vals  = E_vol[mask]
        hu_vals = hu_vol[mask]
        stats[name] = {
            "voxels":     count,
            "vol_frac":   count / total,
            "HU_mean":    float(hu_vals.mean()),
            "HU_std":     float(hu_vals.std()),
            "E_mean_MPa": float(E_vals.mean()),
            "E_min_MPa":  float(E_vals.min()),
            "E_max_MPa":  float(E_vals.max()),
            "E_std_MPa":  float(E_vals.std()),
        }
    porosity = stats.get("Void/Air", {}).get("vol_frac", 0.0)
    return stats, porosity
