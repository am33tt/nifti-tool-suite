"""Automatic pore threshold and threshold sensitivity.

The grey level separating pore from solid decides the porosity: the
figure can double or halve within the width of the noise. This module
samples the specimen interior only, fits the solid peak, offers three
thresholds, and reports how much the answer moves between them.

sigma   T = mode - k * sigma of the solid peak. A voxel is a pore only
        if it is darker than solid by more than k noise widths. Always
        defined, including when the pores are too few to form a peak.
otsu    Maximum between class variance. Good for two populations of
        comparable weight, biased towards mid range when one is tiny.
valley  Histogram minimum between a dark pore peak and the solid peak.
        The most physical choice, but only defined when both peaks are
        resolved; returns None otherwise.

auto uses the valley when one is found and the sigma threshold
otherwise. Nothing assumes a particular specimen shape: the interior is
the filled largest connected component of each slice, eroded to drop the
partial volume skin. Background zeroed volumes are detected separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..deps import np
from . import accel


#: k values reported in the porosity sensitivity sweep.
DEFAULT_K_SWEEP = (2.0, 2.5, 3.0, 3.5, 4.0)
#: Minimum connected sizes reported in the count sweep.
DEFAULT_MIN_VOXEL_SWEEP = (1, 2, 3, 4, 6, 8, 12, 16)


@dataclass
class ThresholdResult:
    """Chosen threshold plus every candidate and the evidence behind it."""

    T: float
    method_used: str
    solid_mode: float
    solid_sigma: float
    k_sigma: float
    t_sigma: float
    t_otsu: float
    t_valley: float | None
    sample_percentiles: dict = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "chosen_threshold": float(self.T),
            "method_used": self.method_used,
            "solid_peak_mode": float(self.solid_mode),
            "solid_peak_sigma": float(self.solid_sigma),
            "k_sigma": float(self.k_sigma),
            "threshold_sigma": float(self.t_sigma),
            "threshold_otsu": float(self.t_otsu),
            "threshold_valley": (None if self.t_valley is None
                                 else float(self.t_valley)),
            "interior_percentiles": self.sample_percentiles,
            "n_interior_samples": int(self.n_samples),
        }


# --------------------------------------------------------------------------
#  interior sampling
# --------------------------------------------------------------------------

def otsu_on_slice(slice_2d, step: int = 4) -> float:
    """Otsu grey level separating bright specimen from dark background."""
    sub = np.asarray(slice_2d, dtype=np.float32)[::step, ::step]
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        return 0.0
    lo, hi = float(sub.min()), float(sub.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(sub, bins=256, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return lo
    p /= total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return float(centres[int(np.argmax(sigma_b))])


def specimen_mask_2d(slice_2d, already_masked: bool = False, erode: int = 1):
    """Boolean interior mask for one slice, for any cross section shape.

    With already_masked the background is exactly zero and the specimen is
    the non-zero region. Otherwise the specimen is found with Otsu, its
    pores filled back in, the largest blob kept, and the result eroded by
    erode voxels to drop the partial volume skin at the surface.
    """
    sl = np.asarray(slice_2d, dtype=np.float32)
    if already_masked:
        m = sl != 0
    else:
        if float(sl.max() - sl.min()) < 1e-9:
            return np.zeros(sl.shape, dtype=bool)
        m = accel.binary_fill_holes(sl >= otsu_on_slice(sl))
        lab, n = accel.label(m)
        if n > 1:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            m = lab == int(counts.argmax())
    if erode and m.any():
        m = accel.binary_erosion(m, iterations=int(erode))
    return np.asarray(m, dtype=bool)


def detect_already_masked(volume, n_scan: int = 15) -> float:
    """Fraction of exactly zero voxels over a few evenly spread slices.

    A background zeroed volume has a large exactly zero region; a raw
    reconstruction does not.
    """
    sz = int(volume.shape[2])
    zs = np.unique(np.linspace(0, sz - 1, min(n_scan, sz)).astype(int))
    fr = [float((np.asarray(volume[:, :, int(z)]) == 0).mean()) for z in zs]
    return float(np.mean(fr)) if fr else 0.0


def resolve_masking(volume, setting="auto", zero_fraction: float = 0.02):
    """Return ``(already_masked, explanation)`` for ``'auto' | True | False``."""
    if setting in (True, False):
        return bool(setting), "forced by the user"
    zf = detect_already_masked(volume)
    return zf > zero_fraction, f"auto-detected, exact-zero fraction {zf:.3f}"


def find_valid_zrange(volume, already_masked: bool = False, *, n_scan: int = 60,
                      area_fraction: float = 0.6, grey_fraction: float = 0.85,
                      erode: int = 1, progress=None):
    """Slice range over which the specimen is present and complete.

    The end slices of a scan are often partial or empty and bias both the
    threshold and the porosity. A slice is kept when its specimen area and
    median interior grey value match the middle of the stack.
    """
    sz = int(volume.shape[2])
    zs = np.unique(np.linspace(0, sz - 1, min(n_scan, sz)).astype(int))
    areas, greys = [], []
    for i, z in enumerate(zs):
        sl = np.asarray(volume[:, :, int(z)], dtype=np.float32)
        if float(sl.max() - sl.min()) < 1e-9:
            areas.append(0.0)
            greys.append(0.0)
            continue
        m = specimen_mask_2d(sl, already_masked, erode)
        areas.append(float(m.mean()))
        greys.append(float(np.median(sl[m])) if m.any() else 0.0)
        if progress is not None:
            progress(i + 1, len(zs))
    areas = np.asarray(areas)
    greys = np.asarray(greys)
    if areas.max() <= 0:
        return 0, sz - 1, zs, areas
    central = (zs > sz * 0.3) & (zs < sz * 0.7)
    ref_a = np.median(areas[central]) if central.any() else areas.max()
    ref_g = np.median(greys[central]) if central.any() else greys.max()
    good = np.flatnonzero((areas >= area_fraction * ref_a)
                          & (greys >= grey_fraction * ref_g))
    if good.size == 0:
        good = np.flatnonzero(areas >= area_fraction * ref_a)
    if good.size == 0:
        return 0, sz - 1, zs, areas
    z_lo, z_hi = int(zs[good.min()]), int(zs[good.max()])
    step = int(np.ceil(sz / max(1, len(zs))))
    z_lo = min(z_lo + step, sz - 1)
    z_hi = max(z_hi - step, 0)
    if z_hi <= z_lo:
        z_lo, z_hi = int(zs[good.min()]), int(zs[good.max()])
    return z_lo, z_hi, zs, areas


def sample_interior(volume, zrange=None, already_masked: bool = False, *,
                    n_slices: int = 40, keep_every: int = 8, erode: int = 1,
                    progress=None):
    """Grey values of interior voxels over evenly spread slices.

    Read at full resolution, since block averaging would blur away the dark
    tail the threshold sits in, but keeping only every keep_every-th voxel.
    """
    sz = int(volume.shape[2])
    z_lo, z_hi = (0, sz - 1) if zrange is None else zrange
    z_hi = max(z_hi, z_lo)
    zs = np.unique(np.linspace(z_lo, z_hi,
                               min(n_slices, z_hi - z_lo + 1)).astype(int))
    parts = []
    for i, z in enumerate(zs):
        sl = np.asarray(volume[:, :, int(z)], dtype=np.float32)
        if float(sl.max() - sl.min()) < 1e-9:
            continue
        m = specimen_mask_2d(sl, already_masked, erode)
        if int(m.sum()) < 50:
            continue
        v = sl[m]
        if already_masked:
            v = v[v != 0]
        if v.size:
            parts.append(v[::max(1, int(keep_every))])
        if progress is not None:
            progress(i + 1, len(zs))
    if not parts:
        raise RuntimeError(
            "No specimen voxels were found while sampling for the threshold. "
            "Check that the volume actually contains a specimen, or set the "
            "background mode manually."
        )
    return np.concatenate(parts)


# --------------------------------------------------------------------------
#  threshold candidates
# --------------------------------------------------------------------------

#: Quantiles of the bright flank used for the robust width estimate, and the
#: normal-distribution constant that converts their spacing into a sigma.
_FLANK_Q_LO, _FLANK_Q_HI = 50.0, 80.0
_FLANK_SIGMA_FACTOR = 0.60706        # Phi^-1(0.90) - Phi^-1(0.75)
_Q75_Z = 0.67449                     # Phi^-1(0.75)


def fit_solid_peak(sample, refine: int = 3):
    """Robust mode and sigma of the solid material peak.

    The peak is located on a smoothed histogram, searching the bright half
    only so a tall dark spike cannot drag it off the material.

    The width comes from quantiles of the bright flank, not from the
    histogram half maximum. At a few tens of thousands of samples the peak
    bin is noisy and its maximum biased upward, which moves the half
    maximum crossing inward and underestimates sigma by around 15 %, and
    that shows up as several times too much porosity at a three sigma
    threshold. The quantile spacing has no such bias, is insensitive to
    sample size, and ignores both the dark pore tail and any bright
    aggregate tail.
    """
    s = np.asarray(sample, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0, 1e-6
    med = float(np.median(s))
    p25, p75 = np.percentile(s, [25, 75])
    iqr = float(p75 - p25)
    if iqr <= 0:
        return med, max(float(np.std(s)), 1e-6)

    # Locate the peak. The bin count follows the sample size; too many bins
    # on a small sample is what makes a width estimate unreliable.
    nbins = int(np.clip(np.sqrt(s.size) * 2.0, 48, 400))
    lo, hi = med - 6.0 * iqr, med + 6.0 * iqr
    hist, edges = np.histogram(s, bins=nbins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = max(3, nbins // 40) | 1                    # odd smoothing window
    smooth = np.convolve(hist.astype(float), np.ones(width) / width,
                         mode="same")
    bright = centres >= med
    peak_i = int(np.argmax(np.where(bright, smooth, -1.0)))
    mode = float(centres[peak_i])

    # --- width from the bright flank ------------------------------------
    sigma = None
    for _ in range(max(1, int(refine))):
        flank = s[s >= mode]
        if flank.size < 32:
            break
        q_lo, q_hi = np.percentile(flank, [_FLANK_Q_LO, _FLANK_Q_HI])
        candidate = float(q_hi - q_lo) / _FLANK_SIGMA_FACTOR
        if not np.isfinite(candidate) or candidate <= 0:
            break
        sigma = candidate
        # Re-centre on the value the flank implies, so a peak bin displaced
        # by histogram noise does not propagate into the width.
        mode = float(q_lo) - _Q75_Z * sigma

    if not sigma or not np.isfinite(sigma) or sigma <= 0:
        sigma = iqr / 1.349                    # robust normal-equivalent
    return float(mode), max(float(sigma), 1e-6)


def otsu_threshold(sample, max_values: int = 2_000_000) -> float:
    """Otsu threshold on a (sub-sampled) flat array of interior values."""
    s = np.asarray(sample, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    if s.size > max_values:
        s = s[:: s.size // max_values]
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(s, bins=512, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64)
    p /= p.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return float(centres[int(np.argmax(sigma_b))])


def valley_threshold(sample, mode: float, sigma: float,
                     floor_pct: float = 0.5) -> float | None:
    """Histogram minimum between a real pore peak and the solid peak.

    Returns None when there is no separated dark population or no dip
    between the two, rather than a made up value.
    """
    s = np.asarray(sample, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    lo = float(np.percentile(s, floor_pct))
    hi = float(mode)
    if hi - lo < 3.0 * sigma:
        return None
    hist, edges = np.histogram(s, bins=256, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(hist.astype(float), np.ones(7) / 7.0, mode="same")

    solid_peak = float(smooth[-10:].max())
    left = smooth[: int(0.6 * len(smooth))]
    if left.size == 0 or solid_peak <= 0:
        return None
    pore_i = int(np.argmax(left))
    pore_peak = float(left[pore_i])
    if pore_peak < 0.02 * solid_peak:
        return None                       # no real dark population
    v_i = pore_i + int(np.argmin(smooth[pore_i:]))
    if float(smooth[v_i]) > 0.6 * min(pore_peak, solid_peak):
        return None                       # not a real dip
    return float(centres[v_i])


def auto_threshold(sample, k_sigma: float = 3.0, method: str = "auto",
                   manual: float | None = None) -> ThresholdResult:
    """Compute every candidate threshold and select one."""
    s = np.asarray(sample, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    mode, sigma = fit_solid_peak(s)
    t_sigma = float(mode - float(k_sigma) * sigma)
    t_otsu = otsu_threshold(s)
    t_valley = valley_threshold(s, mode, sigma)

    if manual is not None:
        T, used = float(manual), "manual"
    elif method == "sigma":
        T, used = t_sigma, "sigma"
    elif method == "otsu":
        T, used = t_otsu, "otsu"
    elif method == "valley":
        if t_valley is None:
            T, used = t_sigma, "sigma (no valley found)"
        else:
            T, used = t_valley, "valley"
    else:
        if t_valley is not None:
            T, used = t_valley, "valley"
        else:
            T, used = t_sigma, "sigma"

    pcts = [0.5, 1, 25, 50, 75, 99, 99.5]
    percentiles = {}
    if s.size:
        vals = np.percentile(s, pcts)
        percentiles = {str(p): float(v) for p, v in zip(pcts, vals)}

    return ThresholdResult(
        T=float(T), method_used=used,
        solid_mode=float(mode), solid_sigma=float(sigma),
        k_sigma=float(k_sigma),
        t_sigma=t_sigma, t_otsu=t_otsu, t_valley=t_valley,
        sample_percentiles=percentiles, n_samples=int(s.size),
    )


# --------------------------------------------------------------------------
#  sensitivity
# --------------------------------------------------------------------------

def porosity_sensitivity(sample, result: ThresholdResult,
                         k_sweep=DEFAULT_K_SWEEP) -> dict:
    """Pore fraction of the interior sample at several thresholds.

    A porosity that swings from 0.4 % to 4 % between k = 2 and k = 4 is not
    a measurement, and the sweep makes that visible.
    """
    s = np.asarray(sample, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    out = {}
    if s.size == 0:
        return out
    for k in k_sweep:
        t = result.solid_mode - float(k) * result.solid_sigma
        out[f"k={k:g}"] = round(100.0 * float((s < t).mean()), 4)
    out["otsu"] = round(100.0 * float((s < result.t_otsu).mean()), 4)
    if result.t_valley is not None:
        out["valley"] = round(100.0 * float((s < result.t_valley).mean()), 4)
    out["chosen"] = round(100.0 * float((s < result.T).mean()), 4)
    return out


def count_sensitivity(sizes, min_voxel_sweep=DEFAULT_MIN_VOXEL_SWEEP) -> dict:
    """Connected pores surviving each minimum size filter.

    A single dark voxel is not a pore: noise, partial volume and ring
    artefacts all produce them. The count only means something together
    with the minimum size it was measured at.
    """
    s = np.asarray(sizes, dtype=np.int64).ravel()
    s = s[s > 0]
    return {str(int(m)): int((s >= int(m)).sum()) for m in min_voxel_sweep}


def summary_lines(result: ThresholdResult, sensitivity: dict | None = None,
                  spacing_mm=None) -> list[str]:
    """Human-readable block for the log pane and the PDF report."""
    lines = [
        f"Interior samples    : {result.n_samples:,} voxels (full resolution)",
        f"Solid peak          : mode {result.solid_mode:.1f}, "
        f"sigma {result.solid_sigma:.1f}",
        f"Candidates          : {result.k_sigma:g}-sigma = {result.t_sigma:.1f}"
        f"   Otsu = {result.t_otsu:.1f}"
        + (f"   valley = {result.t_valley:.1f}"
           if result.t_valley is not None else "   valley = none found"),
        f"Chosen threshold    : {result.T:.1f}   (method: {result.method_used})",
    ]
    if result.t_valley is None:
        lines.append(
            "Note                : no separated pore peak exists in this "
            "histogram, so the noise-referenced sigma threshold was used."
        )
    if sensitivity:
        pairs = ", ".join(f"{k} → {v:.3f} %" for k, v in sensitivity.items()
                          if k != "chosen")
        lines.append(f"Sensitivity         : {pairs}")
    if spacing_mm is not None:
        sp = tuple(float(s) for s in spacing_mm[:3])
        d_min = 2.0 * max(sp)
        lines.append(
            f"Resolution limit    : voxel {sp[0]:.4f} x {sp[1]:.4f} x "
            f"{sp[2]:.4f} mm; pores below about {d_min:.4f} mm across are "
            f"not resolved and are not measured."
        )
    return lines
