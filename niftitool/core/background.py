"""Automatic background detection and removal.

The specimen is the largest connected component of the solid phase -- every
voxel at or above the void/solid threshold -- with its internal cavities
returned to it, so only exterior air is removed. Voxels outside the
specimen are set to a constant, and the volume is optionally cropped to the
specimen's bounding box plus a margin.

Memory
------
The volume is never held in RAM. It is read in slabs of
:data:`~niftitool.core.volume_stream.SLAB_BUDGET_BYTES`, the two masks the
algorithm needs are bit-packed at 0.125 bytes per voxel, and the result is
written slab by slab as it is produced. The working set is therefore

    0.25 bytes/voxel  +  a slab  +  the union-find bookkeeping

independent of the file size: a 3000^3 scan (27 x 10^9 voxels, 54 GB as
uint16) needs about 7 GiB and runs on a 16 GB machine. See
:func:`estimate_memory_bytes`.

Fidelity
--------
Voxels are copied from the source file in their stored dtype and byte
order, and the header's ``scl_slope``/``scl_inter`` are carried over
unchanged, so surviving voxels are bit-identical to the input. Nothing is
interpolated, resampled or re-quantised. Cropping shifts the affine's
origin so world coordinates are preserved, and the fill value is converted
into stored units before it is written, so it means the same grey level it
does on screen.
"""

from __future__ import annotations

from ..deps import np
from .label_stream import PackedMask, fill_cavities, largest_component
from .volume_stream import RawSliceReader, StreamingNiftiWriter

#: How the voxels outside the specimen are filled.
#:
#: ``zero``
#:     Exact zero. The convention the downstream tools expect: the porosity
#:     and threshold code recognises a background-masked volume by its large
#:     exactly-zero region, and simulation exports treat zero as void.
#: ``air-median``
#:     The median grey level of the removed background, which keeps a
#:     realistic air peak in the histogram.
#: ``value``
#:     A grey level given by the caller.
FILL_MODES = ("zero", "air-median", "value")

#: How the outer boundary of the specimen is decided.
#:
#: ``component``
#:     The largest connected body of solid material. Right whenever the
#:     specimen is the only thing in the scan that the threshold sees.
#: ``envelope``
#:     A radial boundary fitted direction by direction, from the void gap
#:     between the specimen and whatever surrounds it. Use it when the
#:     specimen was scanned inside a sleeve, mould or wrapping: those touch
#:     the specimen somewhere over the height of the scan, and one contact
#:     is enough to make the two a single connected body. See
#:     :mod:`niftitool.core.envelope`.
BOUNDARY_MODES = ("component", "envelope")

#: Voxels sampled to estimate the background median.
_MEDIAN_SAMPLE_VOXELS = 4_000_000


def estimate_memory_bytes(shape, itemsize: int = 2) -> float:
    """Peak RAM the removal needs for a volume of *shape*, in bytes.

    Two bit-packed masks at 0.125 bytes per voxel, plus the working copies
    of one slab: the stored voxels, an equally large copy of them and the
    boolean mask of the same slab. The union-find is left out: it is
    proportional to the number of connected components, a few thousand on a
    real scan.
    """
    from .volume_stream import SLAB_BUDGET_BYTES

    n_voxels = 1
    for dim in shape[:3]:
        n_voxels *= int(dim)
    slab_voxels = max(1, SLAB_BUDGET_BYTES // max(1, itemsize))
    return 0.25 * n_voxels + slab_voxels * (2 * itemsize + 1)


def _crop_window(box, shape, margin: int):
    """Bounding box *box* grown by *margin* voxels and clipped to *shape*."""
    margin = max(0, int(margin))
    return tuple(
        (max(0, lo - margin), min(int(dim), hi + margin))
        for (lo, hi), dim in zip(box, shape)
    )


def _sample_positions(shape):
    """Stride giving at most :data:`_MEDIAN_SAMPLE_VOXELS` sampled voxels."""
    n_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
    if n_voxels <= _MEDIAN_SAMPLE_VOXELS:
        return 1
    return int(np.ceil((n_voxels / _MEDIAN_SAMPLE_VOXELS) ** (1.0 / 3.0)))


def _threshold_volume(reader, threshold, *, sample_stride, progress, cancel):
    """First pass: the solid mask, and a strided grey-level sample.

    The sample is kept so the background median can be measured later
    against the *specimen* mask rather than the raw threshold, without
    reading the file again.
    """
    from ..utils import OperationCancelled

    nx, ny, nz = reader.shape
    solid = PackedMask(reader.shape)
    samples = {}
    for z0, raw in reader.iter_slabs():
        if cancel is not None and cancel():
            raise OperationCancelled("cancelled")
        above = reader.solid_mask(raw, threshold)
        for k in range(above.shape[2]):
            z = z0 + k
            solid.set(z, above[:, :, k])
            if sample_stride and z % sample_stride == 0:
                step = slice(None, None, sample_stride)
                samples[z] = reader.to_gray(raw[step, step, k:k + 1])[:, :, 0]
        depth = above.shape[2]
        del raw, above
        if progress is not None:
            progress("thresholding", min(1.0, (z0 + depth) / nz))
    return solid, samples


def _background_median(samples, specimen, stride) -> float:
    """Median grey level of the voxels the mask removes."""
    values = []
    for z, sample in samples.items():
        outside = ~specimen.get(z)[::stride, ::stride]
        values.append(sample[outside])
    if not values:
        return 0.0
    pooled = np.concatenate(values)
    return float(np.median(pooled)) if pooled.size else 0.0


def _write_masked(reader, specimen, out_path, *, window, affine, fill_raw,
                  progress, cancel):
    """Second pass: copy the volume out, replacing background with *fill_raw*."""
    from ..utils import OperationCancelled

    (x0, x1), (y0, y1), (z0, z1) = window
    shape = (x1 - x0, y1 - y0, z1 - z0) + reader.extra_shape
    total = max(1, (z1 - z0) * max(1, int(np.prod(reader.extra_shape or (1,)))))
    done = 0

    with StreamingNiftiWriter(
        out_path,
        source_header=reader.img.header,
        affine=affine,
        shape=shape,
        dtype=reader.dtype,
        slope=reader.slope,
        inter=reader.inter,
    ) as writer:
        for index in np.ndindex(*reader.extra_shape) if reader.extra_shape else [()]:
            for za, raw in reader.iter_slabs(
                index=index, x=(x0, x1), y=(y0, y1), z=(z0, z1),
            ):
                if cancel is not None and cancel():
                    raise OperationCancelled("cancelled")
                block = np.array(raw, copy=True)
                keep = np.stack(
                    [specimen.get(z)[x0:x1, y0:y1]
                     for z in range(za, za + block.shape[2])],
                    axis=2,
                )
                block[~keep] = fill_raw
                writer.write_block(block)
                done += block.shape[2]
                if progress is not None:
                    progress("writing", min(1.0, done / total))
    return shape


def remove_background(
    img,
    threshold: float,
    out_path,
    *,
    margin: int = 10,
    crop: bool = True,
    fill_mode: str = "zero",
    fill_value: float | None = None,
    boundary: str = "component",
    envelope_options: dict | None = None,
    progress=None,
    cancel=None,
) -> dict:
    """Strip everything outside the specimen and write the result to disk.

    Parameters
    ----------
    img
        The loaded :class:`nibabel.Nifti1Image`.
    threshold
        Void/solid grey level, in the same units the viewer displays -- the
        value from the Threshold panel.
    out_path
        Destination ``.nii`` or ``.nii.gz``. It is written as the volume is
        computed, so no copy of the result is ever held in memory.
    margin
        Voxels of padding kept around the specimen's bounding box.
    crop
        Also crop the volume to that padded bounding box.
    fill_mode
        One of :data:`FILL_MODES`.
    fill_value
        The grey level to write when *fill_mode* is ``"value"``.
    boundary
        One of :data:`BOUNDARY_MODES`. ``"envelope"`` fits the specimen
        surface radially first and discards everything beyond it, which is
        what removes a sleeve or mould that touches the specimen.
    envelope_options
        Keyword arguments for :func:`niftitool.core.envelope.fit_envelope`.
    progress
        Optional ``fn(stage: str, fraction: float)`` callback.
    cancel
        Optional ``fn() -> bool``; raises
        :class:`~niftitool.utils.OperationCancelled` when it turns true.

    Returns a dictionary describing what was done: the threshold used, the
    number of solid components found, the specimen and cavity voxel counts,
    the fill value in both grey and stored units, the crop window and the
    shape written.
    """
    if fill_mode not in FILL_MODES:
        raise ValueError(f"fill_mode must be one of {FILL_MODES}, got {fill_mode!r}")
    if boundary not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary must be one of {BOUNDARY_MODES}, got {boundary!r}")

    reader = RawSliceReader(img)
    stride = _sample_positions(reader.shape) if fill_mode == "air-median" else 0

    solid, samples = _threshold_volume(
        reader, threshold, sample_stride=stride, progress=progress, cancel=cancel,
    )
    if solid.count == 0:
        raise ValueError(
            f"No solid material found at or above threshold {threshold:g}."
        )

    envelope_info = None
    if boundary == "envelope":
        from .envelope import apply_envelope, fit_envelope

        before = solid.count
        fitted = fit_envelope(solid, progress=progress, cancel=cancel,
                              **(envelope_options or {}))
        cut = apply_envelope(solid, fitted, progress=progress, cancel=cancel)
        diameters = fitted.diameters()
        envelope_info = dict(
            fitted.summary(),
            cut_solid_voxels=int(cut),
            cut_pct_of_solid=100.0 * cut / max(1, before),
            diameter_px={
                "min": float(np.nanmin(diameters)),
                "median": float(np.nanmedian(diameters)),
                "max": float(np.nanmax(diameters)),
            },
        )
        if solid.count == 0:
            raise ValueError(
                "The fitted envelope removed every solid voxel; the surface "
                "fit did not find the specimen."
            )

    specimen, n_components = largest_component(
        solid, progress=progress, cancel=cancel,
    )
    del solid
    cavity_voxels = fill_cavities(specimen, progress=progress, cancel=cancel)
    specimen_voxels = specimen.count

    if fill_mode == "zero":
        fill_gray = 0.0
    elif fill_mode == "value":
        if fill_value is None:
            raise ValueError("fill_mode='value' needs a fill_value.")
        fill_gray = float(fill_value)
    else:
        fill_gray = _background_median(samples, specimen, stride)
    del samples
    fill_raw = reader.raw_value(fill_gray)

    box = specimen.bounding_box()
    window = (
        _crop_window(box, reader.shape, margin) if crop
        else tuple((0, int(dim)) for dim in reader.shape)
    )

    affine = img.affine.copy()
    origin = np.array([lo for lo, _ in window], dtype=float)
    affine[:3, 3] += affine[:3, :3] @ origin

    out_shape = _write_masked(
        reader, specimen, out_path, window=window, affine=affine,
        fill_raw=fill_raw, progress=progress, cancel=cancel,
    )

    return {
        "path": str(out_path),
        "threshold": float(threshold),
        "boundary": boundary,
        "envelope": envelope_info,
        "solid_components": int(n_components),
        "specimen_voxels": int(specimen_voxels),
        "cavity_voxels": int(cavity_voxels),
        "background_pct": 100.0 * (1.0 - specimen_voxels / reader.n_voxels),
        "fill_mode": fill_mode,
        "fill_value": float(fill_gray),
        "fill_stored": reader.scaled_value(fill_raw) if reader.is_rgb else float(fill_raw),
        "bbox": tuple(window) if crop else None,
        "out_shape": tuple(int(s) for s in out_shape),
        "dtype": str(reader.dtype),
        "peak_mask_bytes": int(specimen.nbytes),
    }
