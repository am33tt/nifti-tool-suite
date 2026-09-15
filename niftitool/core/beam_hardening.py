"""Beam hardening (cupping) correction.

A polychromatic beam hardens as it penetrates, so the reconstruction
reports higher grey values near the specimen surface than in its core.
The mean solid grey value is measured against depth below the surface,
fitted with a low order polynomial I_fit(d), and every voxel rescaled by
I_ref / I_fit(d) with I_ref the deep interior value.

Depth comes from a Euclidean distance transform of the specimen mask at
the true voxel spacing, so any specimen shape and any anisotropic grid
is handled. Distance to the image border, the usual shortcut, is valid
only for a specimen cropped flush to the field of view; setting
border_is_surface reproduces that case.

The distance field is built on a strided grid and the corrected volume
is streamed slice by slice, so volumes larger than RAM are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..deps import np, nib, ndimage
from . import accel


#: Default polynomial degree of the cupping fit.
DEFAULT_POLY_DEGREE = 4
#: Default depth bin width in mm for the measured profile.
DEFAULT_BIN_MM = 0.25
#: Outer / inner fraction of the depth span used for the cupping index.
DEFAULT_EDGE_FRACTION = 0.15
#: Coarse-grid budget for the distance transform (bool + int32 + 2x float32).
BYTES_PER_COARSE_VOXEL = 14.0
#: Hard cap on the coarse grid regardless of free RAM.
DEFAULT_MAX_COARSE_VOXELS = 48_000_000
#: The correction factor is never allowed outside this range.
FACTOR_CLIP = (0.2, 5.0)


@dataclass
class CuppingFit:
    """Everything needed to reproduce and apply a cupping correction."""

    poly_coeffs: list           # highest power first, as returned by polyfit
    poly_degree: int
    i_ref: float                # deep-interior reference grey value
    fit_depth_mm: tuple         # (d_min, d_max) validity range of the fit
    depth_mm: np.ndarray = field(repr=False, default=None)      # bin centres
    profile: np.ndarray = field(repr=False, default=None)       # measured means
    counts: np.ndarray = field(repr=False, default=None)        # voxels per bin
    fitted: np.ndarray = field(repr=False, default=None)        # polynomial values
    corrected: np.ndarray = field(repr=False, default=None)     # profile * factor
    cupping_index_before_pct: float = float("nan")
    cupping_index_after_pct: float = float("nan")
    specimen_threshold: float = float("nan")
    solid_threshold: float = float("nan")
    spacing_mm: tuple = (1.0, 1.0, 1.0)
    stride: int = 1
    border_is_surface: bool = True
    n_solid_samples: int = 0

    def to_dict(self) -> dict:
        """JSON-serialisable summary (also usable as a reusable fit file)."""
        return {
            "poly_coeffs": [float(c) for c in self.poly_coeffs],
            "poly_degree": int(self.poly_degree),
            "i_ref": float(self.i_ref),
            "fit_depth_mm": [float(self.fit_depth_mm[0]),
                             float(self.fit_depth_mm[1])],
            "cupping_index_before_pct": float(self.cupping_index_before_pct),
            "cupping_index_after_pct": float(self.cupping_index_after_pct),
            "specimen_threshold": float(self.specimen_threshold),
            "solid_threshold": float(self.solid_threshold),
            "voxel_spacing_mm": [float(s) for s in self.spacing_mm],
            "analysis_stride": int(self.stride),
            "border_is_surface": bool(self.border_is_surface),
            "n_solid_samples": int(self.n_solid_samples),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CuppingFit":
        return cls(
            poly_coeffs=list(d["poly_coeffs"]),
            poly_degree=int(d.get("poly_degree", len(d["poly_coeffs"]) - 1)),
            i_ref=float(d["i_ref"]),
            fit_depth_mm=tuple(float(v) for v in d["fit_depth_mm"]),
            cupping_index_before_pct=float(
                d.get("cupping_index_before_pct", float("nan"))),
            cupping_index_after_pct=float(
                d.get("cupping_index_after_pct", float("nan"))),
            specimen_threshold=float(d.get("specimen_threshold", float("nan"))),
            solid_threshold=float(d.get("solid_threshold", float("nan"))),
            spacing_mm=tuple(d.get("voxel_spacing_mm", (1.0, 1.0, 1.0))),
            stride=int(d.get("analysis_stride", 1)),
            border_is_surface=bool(d.get("border_is_surface", True)),
        )


# --------------------------------------------------------------------------
#  coarse grid
# --------------------------------------------------------------------------

def choose_stride(shape, available_mb=None,
                  max_coarse_voxels: int = DEFAULT_MAX_COARSE_VOXELS,
                  budget_frac: float = 0.35) -> int:
    """Smallest stride whose coarse grid fits both RAM and the hard cap."""
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if available_mb is None:
        available_mb = 4096.0
    budget = max(256.0, float(available_mb) * budget_frac) * 1024 ** 2
    stride = 1
    while True:
        coarse = max(1, total // stride ** 3)
        if (coarse * BYTES_PER_COARSE_VOXEL <= budget
                and coarse <= max_coarse_voxels):
            return stride
        stride += 1
        if stride > 64:
            return stride


def coarse_volume(volume, stride: int, progress=None):
    """Strided float32 copy, read one Z slice at a time.

    Accepts an ndarray or a LazyGrayVolume, so the full volume is never
    materialised.
    """
    sx, sy, sz = volume.shape[:3]
    zids = list(range(0, sz, stride))
    out = np.empty(
        (len(range(0, sx, stride)), len(range(0, sy, stride)), len(zids)),
        dtype=np.float32,
    )
    n = len(zids)
    for i, z in enumerate(zids):
        out[:, :, i] = np.asarray(volume[:, :, z])[::stride, ::stride]
        if progress is not None and (i % 20 == 0 or i == n - 1):
            progress(i + 1, n)
    return out


# --------------------------------------------------------------------------
#  specimen mask + true depth
# --------------------------------------------------------------------------

def specimen_mask(coarse, threshold: float, *, fill_holes: bool = True,
                  largest_only: bool = True):
    """Boolean specimen mask on the coarse grid.

    Internal pores are filled back in, since a pore is not an outer
    surface, and only the largest connected component is kept.
    """
    mask = np.asarray(coarse >= threshold, dtype=bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(3, 1)
    if largest_only:
        lab, n = accel.label(mask, structure=structure)
        if n > 1:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            mask = lab == int(counts.argmax())
        del lab
    if fill_holes:
        mask = accel.binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


def depth_field_mm(mask, spacing_mm, *, border_is_surface: bool = True):
    """Shortest distance from each specimen voxel to the surface, in mm.

    Zero outside the specimen. Anisotropic voxels go through the sampling
    argument of the distance transform, so this is a physical distance and
    not a voxel count.
    """
    spacing = tuple(float(s) for s in spacing_mm[:3])
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)

    if border_is_surface:
        # One layer of background around the volume, so material touching
        # the array edge sits at zero depth.
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        dist = accel.distance_transform_edt(padded, sampling=spacing)
        dist = dist[1:-1, 1:-1, 1:-1]
    else:
        # The array edge is an arbitrary cut, not a surface: extend the
        # material outward so no depth is measured from it.
        padded = np.pad(mask, 1, mode="edge")
        dist = accel.distance_transform_edt(padded, sampling=spacing)
        dist = dist[1:-1, 1:-1, 1:-1]

    # The transform returns the distance to the nearest background *centre*;
    # the interface itself lies half a voxel closer.
    dist = np.asarray(dist, dtype=np.float32)
    dist -= 0.5 * float(min(spacing))
    np.clip(dist, 0.0, None, out=dist)
    dist[~mask] = 0.0
    return dist


def cupping_index(depth_mm, profile, edge_fraction: float = DEFAULT_EDGE_FRACTION):
    """Percentage by which the surface layer is brighter than the core.

    100 * (I_edge - I_core) / I_edge, over the shallowest and deepest
    edge_fraction of the depth span. Positive is cupping, negative capping.
    """
    d = np.asarray(depth_mm, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    span = float(d.max() - d.min())
    if span <= 0:
        return 0.0
    edge_sel = d <= d.min() + edge_fraction * span
    core_sel = d >= d.max() - edge_fraction * span
    if not edge_sel.any() or not core_sel.any():
        return float("nan")
    edge = float(p[edge_sel].mean())
    core = float(p[core_sel].mean())
    if edge == 0.0:
        return float("nan")
    return 100.0 * (edge - core) / edge


# --------------------------------------------------------------------------
#  fit
# --------------------------------------------------------------------------

def fit_cupping(
    volume,
    spacing_mm,
    *,
    specimen_threshold: float,
    solid_threshold: float | None = None,
    stride: int = 1,
    poly_degree: int = DEFAULT_POLY_DEGREE,
    bin_mm: float = DEFAULT_BIN_MM,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
    border_is_surface: bool = True,
    min_bin_count: int = 200,
    progress=None,
    coarse=None,
) -> CuppingFit:
    """Measure and fit the depth profile of the mean solid grey value.

    specimen_threshold separates specimen from surrounding air,
    solid_threshold separates solid from pores inside it. Pores are left
    out of the profile, otherwise a depth dependent pore fraction reads as
    a grey value gradient. A bin counts only once it holds min_bin_count
    solid voxels, so the sparse deep bins cannot dominate the fit.
    """
    spacing = tuple(float(s) for s in spacing_mm[:3])
    if solid_threshold is None:
        solid_threshold = specimen_threshold

    if coarse is None:
        coarse = coarse_volume(volume, stride, progress=progress)
    coarse_spacing = tuple(s * stride for s in spacing)

    mask = specimen_mask(coarse, specimen_threshold)
    if not mask.any():
        raise ValueError(
            f"No specimen found at or above threshold {specimen_threshold:g}. "
            f"The volume's sampled range is "
            f"{float(coarse.min()):.6g} to {float(coarse.max()):.6g}."
        )

    depth = depth_field_mm(mask, coarse_spacing,
                           border_is_surface=border_is_surface)

    def _bin_profile(selection):
        """Mean grey value per depth bin, or ``None`` if too few bins."""
        d_vals = depth[selection].astype(np.float64)
        g_vals = coarse[selection].astype(np.float64)
        if d_vals.size == 0:
            return None
        d_max = float(d_vals.max())
        if d_max <= 0:
            return None
        edges = np.arange(0.0, d_max + bin_mm, bin_mm)
        if edges.size < 2:
            return None
        idx = np.clip(np.digitize(d_vals, edges) - 1, 0, edges.size - 2)
        nbins = edges.size - 1
        counts = np.bincount(idx, minlength=nbins).astype(np.float64)
        sums = np.bincount(idx, weights=g_vals, minlength=nbins)
        ok = counts >= max(1, int(min_bin_count))
        if ok.sum() < poly_degree + 2:
            # Relax rather than fail: a small region legitimately has few
            # voxels per bin.
            ok = counts > 0
        if ok.sum() < poly_degree + 2:
            return None
        centres = 0.5 * (edges[:-1] + edges[1:])
        return (centres[ok], sums[ok] / counts[ok], counts[ok],
                int(d_vals.size))

    solid = mask & (coarse >= solid_threshold)
    if solid.sum() < 50:
        solid = mask
        solid_threshold = float(specimen_threshold)

    binned = _bin_profile(solid)
    if binned is None and float(solid_threshold) > float(specimen_threshold):
        # The solid cut took the deep interior with it, which happens when
        # it lands inside the material distribution. On a cupped scan that
        # distribution is wide because of the gradient being measured. Fall
        # back to the whole specimen: pore contamination biases the profile
        # far less than losing half its depth range.
        solid = mask
        solid_threshold = float(specimen_threshold)
        binned = _bin_profile(solid)

    if binned is None:
        raise ValueError(
            f"Not enough usable depth bins for a degree-{poly_degree} fit. "
            f"The specimen may be too thin on the analysis grid. Use a "
            f"finer stride, a coarser bin width, or a lower polynomial "
            f"degree."
        )

    d_mm, profile, n_per_bin, n_samples = binned

    # Weight each bin by sqrt(count): the mean of a bin has a standard error
    # proportional to 1/sqrt(n), so this is the correct least-squares weight
    # and stops the thinly populated deep bins from steering the polynomial.
    weights = np.sqrt(n_per_bin)
    coeffs = np.polyfit(d_mm, profile, int(poly_degree), w=weights)

    deep = d_mm >= 0.7 * d_mm.max()
    if not deep.any():
        deep = d_mm >= np.median(d_mm)
    i_ref = float(np.average(profile[deep], weights=n_per_bin[deep]))

    fitted = np.polyval(coeffs, d_mm)
    safe = np.where(np.abs(fitted) < 1e-9, np.nan, fitted)
    corrected = profile * np.clip(i_ref / safe, *FACTOR_CLIP)

    return CuppingFit(
        poly_coeffs=[float(c) for c in coeffs],
        poly_degree=int(poly_degree),
        i_ref=i_ref,
        fit_depth_mm=(float(d_mm.min()), float(d_mm.max())),
        depth_mm=d_mm,
        profile=profile,
        counts=n_per_bin,
        fitted=fitted,
        corrected=corrected,
        cupping_index_before_pct=cupping_index(d_mm, profile, edge_fraction),
        cupping_index_after_pct=cupping_index(d_mm, corrected, edge_fraction),
        specimen_threshold=float(specimen_threshold),
        solid_threshold=float(solid_threshold),
        spacing_mm=spacing,
        stride=int(stride),
        border_is_surface=bool(border_is_surface),
        n_solid_samples=n_samples,
    )


def correction_factor(depth_mm, fit: CuppingFit):
    """Per voxel factor I_ref / I_fit(depth).

    Depth is clamped to the fitted range so the polynomial is never
    extrapolated, and the factor itself is clipped to FACTOR_CLIP.
    """
    lo, hi = fit.fit_depth_mm
    d = np.clip(np.asarray(depth_mm, dtype=np.float32), lo, hi)
    fitted = np.polyval(np.asarray(fit.poly_coeffs, dtype=np.float64), d)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = float(fit.i_ref) / fitted
    factor = np.where(np.isfinite(factor), factor, 1.0)
    return np.clip(factor, *FACTOR_CLIP).astype(np.float32)


# --------------------------------------------------------------------------
#  streaming application
# --------------------------------------------------------------------------

#: Byte offsets of the NIfTI-1 header fields written explicitly below.
_OFF_VOX_OFFSET = 108
_OFF_SCL_SLOPE = 112
_OFF_SCL_INTER = 116
_OFF_MAGIC = 344
_NIFTI1_HEADER_BYTES = 352


def _write_header(path, img, shape, np_dtype) -> int:
    """Write a NIfTI-1 header preserving the source affine.

    The source header is copied and only shape, dtype and scaling changed.
    Rebuilding from positive spacing alone would mirror flipped volumes.

    vox_offset, scl_slope and scl_inter are stamped into the byte buffer
    directly: nibabel does not reliably carry hand set values through
    write_to, and a reader that honours them literally would then start at
    byte zero or apply a NaN scale factor.
    """
    import io as _io
    import struct

    hdr = nib.Nifti1Header.from_header(img.header)
    hdr.set_data_shape(tuple(int(s) for s in shape))
    hdr.set_data_dtype(np_dtype)
    try:
        hdr.set_zooms(tuple(float(z) for z in img.header.get_zooms()[:3]))
    except Exception:
        pass

    affine = np.asarray(img.affine, dtype=np.float64)
    s_code = int(img.header["sform_code"]) or 1
    q_code = int(img.header["qform_code"]) or 1
    hdr.set_sform(affine, code=s_code)
    try:
        hdr.set_qform(affine, code=q_code)
    except Exception:
        hdr.set_qform(affine, code=1)

    buf = _io.BytesIO()
    hdr.write_to(buf)
    data = bytearray(buf.getvalue()[:_NIFTI1_HEADER_BYTES])
    if len(data) < _NIFTI1_HEADER_BYTES:
        data.extend(b"\x00" * (_NIFTI1_HEADER_BYTES - len(data)))

    # Single-file NIfTI: voxels start right after the 348-byte header plus
    # the 4-byte extension flag.  The values written below are already the
    # scaled ones read through ``dataobj``, so the scale is the identity.
    struct.pack_into("<f", data, _OFF_VOX_OFFSET, float(_NIFTI1_HEADER_BYTES))
    struct.pack_into("<f", data, _OFF_SCL_SLOPE, 1.0)
    struct.pack_into("<f", data, _OFF_SCL_INTER, 0.0)
    data[_OFF_MAGIC:_OFF_MAGIC + 4] = b"n+1\x00"

    with open(path, "wb") as fh:
        fh.write(bytes(data))
    return _NIFTI1_HEADER_BYTES


def apply_correction(
    img,
    volume,
    out_path,
    fit: CuppingFit,
    *,
    coarse_mask=None,
    stride: int = 1,
    remove_background: bool = False,
    background_value: float = 0.0,
    out_dtype: str = "float32",
    progress=None,
    cancelled=None,
):
    """Write the corrected volume, streaming slice by slice.

    The coarse depth field is interpolated up to full resolution one slice
    at a time; depth is smooth, so trilinear interpolation is accurate to
    well below a bin width. Peak memory is a few slices.

    Returns summary information plus a before/after demo slice.
    """
    out_path = Path(out_path)
    sx, sy, sz = volume.shape[:3]
    spacing = tuple(float(s) for s in fit.spacing_mm[:3])
    coarse_spacing = tuple(s * stride for s in spacing)

    if coarse_mask is None:
        raise ValueError("apply_correction needs the coarse specimen mask.")

    depth_coarse = depth_field_mm(coarse_mask, coarse_spacing,
                                  border_is_surface=fit.border_is_surface)
    factor_coarse = correction_factor(depth_coarse, fit)
    # Outside the specimen there is nothing to correct.
    factor_coarse = np.where(coarse_mask, factor_coarse, 1.0).astype(np.float32)
    mask_coarse = np.asarray(coarse_mask, dtype=np.float32)

    np_dtype = np.int16 if out_dtype == "int16" else np.float32
    little = "<i2" if out_dtype == "int16" else "<f4"
    itemsize = np.dtype(np_dtype).itemsize

    hdr_bytes = _write_header(out_path, img, (sx, sy, sz), np_dtype)
    with open(out_path, "r+b") as fh:
        fh.truncate(hdr_bytes + sx * sy * sz * itemsize)

    # Full-resolution index grid mapped onto the coarse grid. The x and y
    # rows are the same for every slice, so the (3, sx, sy) coordinate
    # array is built once and only its z row is rewritten per slice. That
    # saves two full-size allocations and copies per slice, and -- when the
    # call runs on the GPU -- lets the buffer be reused rather than rebuilt
    # before each upload.
    gx = (np.arange(sx, dtype=np.float32) / float(stride))
    gy = (np.arange(sy, dtype=np.float32) / float(stride))
    coords = np.empty((3, sx, sy), dtype=np.float32)
    coords[0], coords[1] = np.meshgrid(gx, gy, indexing="ij")

    slice_bytes = sx * sy * itemsize
    demo_z = sz // 2
    demo_before = demo_after = None
    n_specimen = 0

    with open(out_path, "r+b") as fh:
        for z in range(sz):
            if cancelled is not None and cancelled():
                from ..utils import OperationCancelled
                raise OperationCancelled()
            raw = np.asarray(volume[:, :, z], dtype=np.float32)
            coords[2] = z / float(stride)

            factor = accel.map_coordinates(
                factor_coarse, coords, order=1, mode="nearest",
            ).astype(np.float32)
            out_sl = raw * factor

            if remove_background:
                m = accel.map_coordinates(
                    mask_coarse, coords, order=1, mode="nearest",
                ) >= 0.5
                out_sl = np.where(m, out_sl, np.float32(background_value))
                n_specimen += int(m.sum())

            if z == demo_z:
                demo_before = raw.copy()
                demo_after = out_sl.copy()

            if np_dtype is np.int16:
                buf = np.clip(np.rint(out_sl), -32768, 32767).astype(little)
            else:
                buf = out_sl.astype(little)
            fh.seek(hdr_bytes + z * slice_bytes)
            fh.write(buf.tobytes(order="F"))

            if progress is not None and (z % 10 == 0 or z == sz - 1):
                progress(z + 1, sz)

    return {
        "output_path": str(out_path),
        "shape": (int(sx), int(sy), int(sz)),
        "dtype": out_dtype,
        "bytes": out_path.stat().st_size,
        "background_removed": bool(remove_background),
        "specimen_voxels": int(n_specimen) if remove_background else None,
        "demo_z": int(demo_z),
        "demo_before": demo_before,
        "demo_after": demo_after,
        "fit": fit.to_dict(),
    }


def summary_lines(fit: CuppingFit) -> list[str]:
    """Human-readable block for the log pane and the PDF report."""
    lines = [
        f"Analysis grid       : stride x{fit.stride}  "
        f"(voxel {fit.spacing_mm[0]:.4f} x {fit.spacing_mm[1]:.4f} x "
        f"{fit.spacing_mm[2]:.4f} mm)",
        f"Depth reference     : "
        f"{'array border counts as surface' if fit.border_is_surface else 'only real material/air interfaces'}",
        f"Specimen threshold  : {fit.specimen_threshold:.4g}   "
        f"solid threshold: {fit.solid_threshold:.4g}",
        f"Solid samples       : {fit.n_solid_samples:,} voxels on the "
        f"analysis grid",
        f"Depth range fitted  : {fit.fit_depth_mm[0]:.4f} - "
        f"{fit.fit_depth_mm[1]:.4f} mm  (degree {fit.poly_degree})",
        f"Interior reference  : I_ref = {fit.i_ref:.2f}",
        f"Cupping index       : {fit.cupping_index_before_pct:+.2f} %  →  "
        f"{fit.cupping_index_after_pct:+.2f} %  after correction",
    ]
    if abs(fit.cupping_index_before_pct) < 1.0:
        lines.append(
            "Note                : the measured cupping is below 1 %, so this "
            "scan shows little beam hardening and the correction will barely "
            "change it."
        )
    return lines
