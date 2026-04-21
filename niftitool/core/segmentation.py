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
