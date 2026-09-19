"""Binarisation into the indicator field the finite cell method uses.

Output is uint8: 1 material, 0 pore/background -- the indicator alpha(x)
sampled at the FCM quadrature points. Use alpha = eps (1e-6 to 1e-10) in
the void, not exactly zero (a cell entirely outside the material gives a
singular stiffness matrix); the substitution belongs in the FCM code, not
here. Disconnected material is an unconstrained rigid body; see
:func:`component_report`.

Thresholding runs on a strided sample; binarisation streams slab by slab.
The threshold is recomputed inside the specimen envelope by default, since
surrounding air would otherwise drag a global threshold off the boundary.

Methods: ``otsu`` (default), ``isodata``, ``valley``, ``sigma``,
``triangle``, ``manual`` -- see each function's docstring.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..deps import np, nib, ndimage
from . import accel, fcm_field


HEADER_BYTES = 352
SAMPLE_BUDGET = 4_000_000

METHOD_NAMES = ("otsu", "isodata", "valley", "sigma", "triangle", "manual")
#: k values reported by :func:`porosity_sweep`.
DEFAULT_K_SWEEP = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0)


@dataclass
class BinarizeResult:
    """Everything the binarisation produced, for the log, report and JSON."""

    method: str
    threshold: float
    threshold_before_refinement: float
    specimen_fraction_of_box: float | None
    separability_warning: str | None
    material_mode: float
    material_sigma: float
    shape: tuple
    voxel_size_mm: tuple
    material_voxels: int
    total_voxels: int
    material_fraction_of_box: float
    smooth_sigma: float = 0.0
    downsample: int = 1
    output_path: str | None = None
    material_components: int = 0
    largest_component_fraction: float | None = None
    removed_voxels: int = 0
    voids_filled_voxels: int = 0
    porosity_inside_specimen: float | None = None
    specimen_voxels: int = 0
    sample: object = field(default=None, repr=False)
    preview_mask: object = field(default=None, repr=False)
    preview_slice: object = field(default=None, repr=False)
    preview_z: int = 0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "threshold": float(self.threshold),
            "threshold_before_refinement":
                float(self.threshold_before_refinement),
            "specimen_fraction_of_box": self.specimen_fraction_of_box,
            "separability_warning": self.separability_warning,
            "material_mode": float(self.material_mode),
            "material_sigma": float(self.material_sigma),
            "smooth_sigma": float(self.smooth_sigma),
            "downsample": int(self.downsample),
            "shape": [int(v) for v in self.shape],
            "voxel_size_mm": [float(v) for v in self.voxel_size_mm],
            "material_voxels": int(self.material_voxels),
            "total_voxels": int(self.total_voxels),
            "material_fraction_of_box": float(self.material_fraction_of_box),
            "material_components": int(self.material_components),
            "largest_component_fraction": self.largest_component_fraction,
            "removed_island_voxels": int(self.removed_voxels),
            "voids_filled_voxels": int(self.voids_filled_voxels),
            "porosity_inside_specimen": self.porosity_inside_specimen,
            "specimen_voxels": int(self.specimen_voxels),
            "output": self.output_path,
        }


# ---------------------------------------------------------------------------
#  threshold rules
# ---------------------------------------------------------------------------

def _finite(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


def otsu_threshold(values, nbins: int = 512) -> float:
    """Otsu threshold, refined towards the middle of the valley.

    With well separated peaks the between class variance is flat across the
    empty valley and its argmax lands at one edge. The isodata iteration
    moves the cut to the midpoint between the two class means.
    """
    v = _finite(values)
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return float(centres[nbins // 2])
    p /= total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return isodata_threshold(v, start=float(centres[int(np.argmax(sigma_b))]))


def isodata_threshold(values, start: float | None = None,
                      iterations: int = 50) -> float:
    """Ridler-Calvard: converge on the midpoint of the two class means."""
    v = _finite(values)
    if v.size == 0:
        return 0.0
    t = float(v.mean()) if start is None else float(start)
    for _ in range(iterations):
        lo, hi = v[v < t], v[v >= t]
        if lo.size == 0 or hi.size == 0:
            break
        t_new = 0.5 * (float(lo.mean()) + float(hi.mean()))
        if abs(t_new - t) <= 1e-6 * max(1.0, abs(t)):
            return t_new
        t = t_new
    return float(t)


def _smoothed_histogram(values, nbins: int, smooth: float):
    hist, edges = np.histogram(values, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    counts = hist.astype(np.float64)
    if smooth <= 0:
        return counts, centres
    if ndimage is not None:
        counts = accel.gaussian_filter1d(counts, smooth, mode="nearest")
    else:
        width = max(3, int(smooth * 2) | 1)
        counts = np.convolve(counts, np.ones(width) / width, mode="same")
    return counts, centres


def _material_window(counts, rel_height: float = 0.02):
    """Indices bracketing the lower flank of the material peak.

    The material peak is the highest mode above the noise floor; the
    window starts at the next mode below it, keeping a third population
    (typically surrounding air) from capturing valley/triangle. rel_height
    is kept low since a small pore peak can be ~1-2% of the material peak.
    """
    if counts.size < 3:
        return 0, int(np.argmax(counts)) if counts.size else 0
    floor = rel_height * float(counts.max())
    peaks = [i for i in range(1, counts.size - 1)
             if counts[i] >= counts[i - 1] and counts[i] > counts[i + 1]
             and counts[i] >= floor]
    if not peaks:
        return 0, int(np.argmax(counts))
    material = peaks[-1]
    lower = [p for p in peaks if p < material]
    return (lower[-1] if lower else 0), material


def valley_threshold(values, nbins: int = 512, smooth: float = 4.0) -> float:
    """Minimum of the smoothed histogram on the material peak lower flank.

    Smoothing first, otherwise noise creates spurious local minima. The cut
    sits at the centre of the minimal plateau rather than its first bin, so
    a wide empty valley does not bias the threshold to one side.
    """
    v = _finite(values)
    if v.size == 0:
        return 0.0
    counts, centres = _smoothed_histogram(v, nbins, smooth)
    left, material = _material_window(counts)
    if material <= left:
        return otsu_threshold(v, nbins=nbins)
    window = counts[left:material + 1]
    lowest = window.min()
    flat = np.flatnonzero(
        window <= lowest + 1e-9 * max(1.0, float(window.max())))
    return float(centres[left + int(flat[flat.size // 2])])


def sigma_threshold(values, k: float = 3.0, nbins: int = 512) -> float:
    """Material peak minus k robust standard deviations.

    Uses the mode of the upper population and the spread above it, so a
    small pore population cannot drag the estimate.
    """
    v = _finite(values)
    if v.size == 0:
        return 0.0
    split = otsu_threshold(v, nbins=nbins)
    material = v[v >= split]
    if material.size < 64:
        material = v
    hist, edges = np.histogram(material, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    mode = float(centres[int(np.argmax(hist))])
    upper = material[material >= mode]
    # Spread of the upper half only: the lower half is contaminated by the
    # partial-volume voxels of the pore walls.
    sigma = float(np.sqrt(np.mean((upper - mode) ** 2))) if upper.size else 0.0
    if sigma <= 0:
        return split
    return mode - float(k) * sigma


def triangle_threshold(values, nbins: int = 512) -> float:
    """Zack triangle rule on the material peak lower flank.

    The chord runs from the material peak to the foot of the flank and the
    cut is the histogram point furthest from it. Restricting the chord to
    the flank matters when a large air population dominates the histogram.
    """
    v = _finite(values)
    if v.size == 0:
        return 0.0
    counts, centres = _smoothed_histogram(v, nbins, smooth=2.0)
    left, material = _material_window(counts)
    if material <= left:
        return otsu_threshold(v, nbins=nbins)
    x = np.arange(left, material + 1, dtype=np.float64)
    y = counts[left:material + 1]
    x0, y0 = float(left), float(counts[left])
    x1, y1 = float(material), float(counts[material])
    norm = math.hypot(x1 - x0, y1 - y0)
    if norm == 0:
        return float(centres[material])
    distance = np.abs((y1 - y0) * x - (x1 - x0) * y
                      + x1 * y0 - y1 * x0) / norm
    return float(centres[left + int(np.argmax(distance))])


def compute_threshold(values, method: str = "otsu", *, nbins: int = 512,
                      sigma_k: float = 3.0, manual: float | None = None) -> float:
    """Dispatch to the named threshold rule."""
    method = (method or "otsu").strip().lower()
    if method == "manual":
        if manual is None:
            raise ValueError("The manual method needs a threshold value.")
        return float(manual)
    if method == "otsu":
        return otsu_threshold(values, nbins=nbins)
    if method == "isodata":
        return isodata_threshold(values)
    if method == "valley":
        return valley_threshold(values, nbins=nbins)
    if method == "sigma":
        return sigma_threshold(values, k=sigma_k, nbins=nbins)
    if method == "triangle":
        return triangle_threshold(values, nbins=nbins)
    raise ValueError(f"Unknown threshold method: {method!r}")


# ---------------------------------------------------------------------------
#  sampling, envelope and diagnostics
# ---------------------------------------------------------------------------

def sample_volume(volume, budget: int = SAMPLE_BUDGET, hist_crop: float = 1.0):
    """Strided sample of the volume, for histogram based thresholding.

    hist_crop below 1 keeps a centred fraction of each axis, which excludes
    surrounding air when the specimen sits in a wide field of view.
    """
    shape = tuple(int(s) for s in volume.shape[:3])
    slices = []
    for n in shape:
        if hist_crop >= 1.0:
            slices.append(slice(0, n))
        else:
            keep = max(1, int(n * hist_crop))
            start = (n - keep) // 2
            slices.append(slice(start, start + keep))
    kept = [s.stop - s.start for s in slices]
    stride = max(1, int(round((float(np.prod(kept)) / budget) ** (1.0 / 3.0))))
    sub = tuple(slice(s.start, s.stop, stride) for s in slices)
    return np.asarray(volume[sub], dtype=np.float32), stride


def material_peak(values, nbins: int = 512):
    """Mode and sigma of the material population.

    Pass values from inside the specimen. The mode is the tallest histogram
    peak, so on a whole volume it would land on the air population instead.

    Sigma comes from the upper half only, since the lower half carries the
    partial volume voxels of the pore walls.
    """
    v = _finite(values)
    if v.size == 0:
        return 0.0, 0.0
    median = float(np.median(v))
    upper = v[v >= median]
    hist, edges = np.histogram(v, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    mode = float(centres[int(np.argmax(hist))])
    sigma = float(np.sqrt(np.mean((upper - median) ** 2))) if upper.size else 0.0
    return mode, sigma


def separability_note(values, threshold):
    """Warn when the threshold lands inside the material peak.

    *values* must come from inside the specimen, as for :func:`material_peak`.
    Otsu-family methods assume comparable-weight populations; at low
    porosity the optimum can degenerate into splitting the material noise,
    seen as a cut within ~2 sigma of the material mode.
    """
    mode, sigma = material_peak(values)
    if sigma <= 0:
        return None, mode, sigma
    distance = (mode - threshold) / sigma
    if distance < 2.0:
        return (f"the threshold is only {distance:.1f} sigma below the "
                f"material mode ({mode:.0f}, sigma {sigma:.0f}): this "
                f"histogram is not bimodal and the cut is splitting the "
                f"material itself. Run the threshold sweep, then use the "
                f"sigma or manual method."), mode, sigma
    return None, mode, sigma


def porosity_sweep(values, ks=DEFAULT_K_SWEEP):
    """Void fraction against threshold = mode - k * sigma.

    A real pore population shows as a plateau (fraction stops falling
    steeply past the noise tail); picking k there measures pores, picking
    it on the steep part measures noise.
    """
    v = _finite(values)
    mode, sigma = material_peak(v)
    rows = []
    if sigma <= 0:
        return mode, sigma, rows
    previous = None
    for k in ks:
        t = mode - float(k) * sigma
        fraction = float((v < t).mean())
        rows.append({
            "k": float(k), "threshold": float(t),
            "void_fraction": fraction,
            "drop_from_previous": (None if previous is None
                                   else previous - fraction),
        })
        previous = fraction
    return mode, sigma, rows


def specimen_envelope(sample, threshold):
    """Boolean envelope of the specimen: material mask, closed and filled."""
    if ndimage is None:
        return None
    mask = np.asarray(sample) >= threshold
    if not mask.any():
        return None
    # Close before filling, so a pore breaking the surface cannot let the
    # fill leak out into the surrounding air.
    closed = accel.binary_closing(mask, iterations=2, border_value=0)
    return accel.binary_fill_holes(closed)


# ---------------------------------------------------------------------------
#  streaming binarisation
# ---------------------------------------------------------------------------

def _downsampled_affine(affine, factor: int):
    """Affine for a block-averaged volume: scaled axes, shifted origin."""
    out = np.array(affine, dtype=np.float64, copy=True)
    out[:3, :3] = np.asarray(affine)[:3, :3] * float(factor)
    offset = (factor - 1) / 2.0
    out[:3, 3] = np.asarray(affine)[:3, 3] + \
        np.asarray(affine)[:3, :3] @ np.array([offset] * 3)
    return out


def _block_reduce(mask, factor: int):
    """Majority-vote downsample. See :func:`fcm_field.block_reduce_binary`.

    Kept as a thin alias so there is exactly one implementation of the
    reduction in the project: this module and ``scripts/binarize_for_fcm``
    each had their own copy, which is how two code paths in one paper end
    up disagreeing.
    """
    return fcm_field.block_reduce_binary(mask, factor)


def _write_header(path, img, out_shape, zooms, affine, *,
                  dtype=np.uint8, descrip=b'FCM indicator: 1=material 0=void') -> int:
    """Write a NIfTI-1 header carrying the source orientation. *dtype* and
    *descrip* let the same writer emit both the uint8 indicator and the
    float32 fraction field, which must share affine/zooms/grid."""
    import io as _io
    import struct

    header = nib.Nifti1Header()
    header.set_data_shape(tuple(int(s) for s in out_shape))
    header.set_data_dtype(dtype)
    try:
        header.set_qform(affine, code=int(img.header['qform_code']) or 1)
    except Exception:
        header.set_qform(affine, code=1)
    header.set_sform(affine, code=int(img.header['sform_code']) or 1)
    header.set_zooms(tuple(float(z) for z in zooms))
    header['cal_min'] = 0.0
    header['cal_max'] = 1.0
    header['descrip'] = descrip

    buf = _io.BytesIO()
    header.write_to(buf)
    data = bytearray(buf.getvalue()[:HEADER_BYTES])
    if len(data) < HEADER_BYTES:
        data.extend(b"\x00" * (HEADER_BYTES - len(data)))
    # nibabel does not reliably carry hand set values through write_to, and
    # a reader that honours them literally would then start at byte zero or
    # apply a NaN scale factor.
    struct.pack_into("<f", data, 108, float(HEADER_BYTES))   # vox_offset
    struct.pack_into("<f", data, 112, 1.0)                   # scl_slope
    struct.pack_into("<f", data, 116, 0.0)                   # scl_inter
    data[344:348] = b"n+1\x00"
    with open(path, "wb") as fh:
        fh.write(bytes(data))
    return HEADER_BYTES


def binarize_stream(img, volume, threshold: float, out_path, *,
                    smooth: float = 0.0, downsample: int = 1,
                    slab: int = 64, progress=None, cancelled=None,
                    fraction_path=None):
    """Threshold the volume slab by slab, writing a uint8 NIfTI (peak memory
    is one slab). When *fraction_path* is given and downsampled > 1, also
    writes the exact material volume fraction per coarse voxel on the same
    grid (see :mod:`niftitool.core.fcm_field`); at downsample 1 there is no
    sub-voxel information and no fraction file is written.
    """
    out_path = Path(out_path)
    nx, ny, nz = (int(s) for s in volume.shape[:3])
    factor = max(1, int(downsample))
    halo = int(math.ceil(3.0 * smooth)) if (smooth > 0 and ndimage) else 0
    if smooth > 0 and ndimage is None:
        smooth = 0.0

    out_shape = ((nx // factor, ny // factor, nz // factor) if factor > 1
                 else (nx, ny, nz))
    if min(out_shape) == 0:
        raise ValueError(
            f"A downsample factor of {factor} leaves no voxels in a "
            f"{nx}x{ny}x{nz} volume.")
    affine = (_downsampled_affine(img.affine, factor) if factor > 1
              else np.asarray(img.affine))
    zooms = tuple(float(z) * factor for z in img.header.get_zooms()[:3])

    hdr_bytes = _write_header(out_path, img, out_shape, zooms, affine)
    slice_bytes = out_shape[0] * out_shape[1]
    step = max(factor, (slab // factor) * factor)
    material = 0
    preview_mask = None
    preview_slice = None
    preview_z = out_shape[2] // 2
    t0 = time.time()

    write_fraction = fraction_path is not None and factor > 1
    frac_fh = None
    frac_hdr_bytes = 0
    if write_fraction:
        frac_hdr_bytes = _write_header(
            fraction_path, img, out_shape, zooms, affine,
            dtype=np.float32,
            descrip=b'FCM material volume fraction, 0..1',
        )

    # ExitStack, not a bare open(): a cancelled run raises out of the loop
    # and the fraction file must still be closed.
    with open(out_path, "r+b") as fh, contextlib.ExitStack() as stack:
        fh.truncate(hdr_bytes + int(np.prod(out_shape)))
        if write_fraction:
            frac_fh = stack.enter_context(open(fraction_path, "r+b"))
            frac_fh.truncate(frac_hdr_bytes + int(np.prod(out_shape)) * 4)
        for z0 in range(0, out_shape[2] * factor, step):
            if cancelled is not None and cancelled():
                from ..utils import OperationCancelled
                raise OperationCancelled()
            z1 = min(z0 + step, out_shape[2] * factor)
            lo = max(0, z0 - halo)
            hi = min(nz, z1 + halo)
            block = np.asarray(volume[:, :, lo:hi], dtype=np.float32)
            if smooth > 0:
                block = accel.gaussian_filter(block, smooth, mode="nearest")
            block = block[:, :, z0 - lo:z1 - lo]
            mask = (block >= threshold).astype(np.uint8)
            fraction = None
            if factor > 1:
                # Compute the fraction first; the binary mask is a threshold
                # of it, so this cannot make the two disagree.
                fraction = fcm_field.block_volume_fraction(mask, factor)
                mask = (fraction >= 0.5).astype(np.uint8)
            material += int(mask.sum())

            out_z0 = z0 // factor
            if out_z0 <= preview_z < out_z0 + mask.shape[2]:
                k = preview_z - out_z0
                preview_mask = mask[:, :, k].copy()
                preview_slice = block[:, :, min(k * factor,
                                                block.shape[2] - 1)].copy()

            # NIfTI stores i fastest, k slowest, so a z slab is contiguous.
            fh.seek(hdr_bytes + out_z0 * slice_bytes)
            fh.write(mask.tobytes(order="F"))
            if frac_fh is not None and fraction is not None:
                frac_fh.seek(frac_hdr_bytes + out_z0 * slice_bytes * 4)
                frac_fh.write(
                    np.ascontiguousarray(fraction, dtype="<f4")
                    .tobytes(order="F"))
            if progress is not None:
                progress(min(z1, out_shape[2] * factor), out_shape[2] * factor)

    return {
        "shape": out_shape, "affine": affine, "zooms": zooms,
        "material_voxels": int(material),
        "fraction_path": str(fraction_path) if write_fraction else None,
        "fraction_skipped": (
            "downsample is 1, so a coarse voxel has no sub-voxel fraction"
            if fraction_path is not None and factor == 1 else None),
        "preview_mask": preview_mask, "preview_slice": preview_slice,
        "preview_z": int(preview_z), "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
#  post-processing
# ---------------------------------------------------------------------------

def component_report(mask, largest_only: bool):
    """Count material islands, optionally keeping only the largest.

    Disconnected material is unconstrained in an FCM analysis, so it is
    reported even when left in place.
    """
    if ndimage is None:
        return mask, {"material_components": 0, "removed_voxels": 0}
    structure = ndimage.generate_binary_structure(3, 1)   # face connectivity
    labels, count = accel.label(mask, structure=structure)
    if count <= 1:
        return mask, {"material_components": int(count), "removed_voxels": 0}
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    info = {
        "material_components": int(count),
        "largest_component_fraction": float(sizes[biggest] / sizes.sum()),
        "removed_voxels": 0,
    }
    if largest_only:
        info["removed_voxels"] = int(sizes.sum() - sizes[biggest])
        mask = (labels == biggest).astype(np.uint8)
    return mask, info


def remove_small_voids(mask, min_size: int):
    """Fill pores smaller than *min_size* voxels (noise speckle)."""
    if ndimage is None or min_size <= 1:
        return mask, 0
    labels, count = accel.label(mask == 0)
    if count == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    small = np.flatnonzero((sizes > 0) & (sizes < int(min_size)))
    if small.size == 0:
        return mask, 0
    fill = np.isin(labels, small)
    mask = mask.copy()
    mask[fill] = 1
    return mask, int(fill.sum())


def porosity_inside_specimen(mask, max_voxels: int = 40_000_000):
    """Porosity within the specimen envelope, not of the whole box.

    The envelope is the material mask with its holes filled, so air around
    the specimen is excluded from the denominator.
    """
    if ndimage is None:
        return None
    stride = 1
    while mask[::stride, ::stride, ::stride].size > max_voxels:
        stride += 1
    small = mask[::stride, ::stride, ::stride]
    envelope = accel.binary_fill_holes(small.astype(bool))
    envelope_voxels = int(envelope.sum())
    if envelope_voxels == 0:
        return None
    pore_voxels = int(envelope_voxels - int(small.sum()))
    return {
        "stride": stride,
        "specimen_voxels": envelope_voxels,
        "pore_voxels": pore_voxels,
        "porosity": pore_voxels / envelope_voxels,
    }


def summary_lines(result: BinarizeResult) -> list[str]:
    """Human-readable block for the log pane and the report."""
    lines = [
        f"Method              : {result.method}",
        f"Threshold           : {result.threshold:.2f}"
        + (f"   (before refinement: "
           f"{result.threshold_before_refinement:.2f})"
           if result.threshold != result.threshold_before_refinement else ""),
        f"Material peak       : mode {result.material_mode:.1f}, "
        f"sigma {result.material_sigma:.1f}",
    ]
    if result.specimen_fraction_of_box is not None:
        lines.append(
            f"Specimen fills      : "
            f"{result.specimen_fraction_of_box * 100:.1f} % of the box "
            f"(threshold recomputed inside it, so surrounding air cannot "
            f"drag it)")
    lines.append(
        f"Material fraction   : "
        f"{result.material_fraction_of_box * 100:.2f} % of the box "
        f"({result.material_voxels:,} of {result.total_voxels:,} voxels)")
    if result.downsample > 1:
        lines.append(
            f"Downsampled         : x{result.downsample} by volume fraction "
            f"(a block is material when at least half of it is)")
    if result.voids_filled_voxels:
        lines.append(
            f"Small pores filled  : {result.voids_filled_voxels:,} voxels")
    if result.material_components:
        line = f"Material bodies     : {result.material_components}"
        if result.removed_voxels:
            line += (f"   (kept the largest; removed "
                     f"{result.removed_voxels:,} island voxels, which would "
                     f"otherwise be unconstrained rigid bodies)")
        elif result.material_components > 1:
            line += ("   (disconnected material is an unconstrained rigid "
                     "body in the analysis)")
        lines.append(line)
    if result.porosity_inside_specimen is not None:
        lines.append(
            f"Porosity            : "
            f"{result.porosity_inside_specimen * 100:.2f} % inside the "
            f"specimen envelope")
    lines.append(
        "FCM note            : the file stores 0/1. Use alpha = eps "
        "(1e-6 to 1e-10) in the void inside the solver; a cell entirely "
        "outside the material gives a singular stiffness matrix at "
        "alpha = 0.")
    if result.separability_warning:
        lines.append(f"WARNING             : {result.separability_warning}")
    return lines
