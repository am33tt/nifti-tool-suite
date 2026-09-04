"""Automatic background detection and removal.

The specimen is the largest connected component of the solid phase, that
is everything at or above the void/solid threshold. ``binary_fill_holes``
then returns internal pores to the mask, so only exterior air is removed.
Voxels outside the specimen are set to a constant fill value and the volume
is optionally cropped to the specimen bounding box plus a margin.
"""

from __future__ import annotations

from ..deps import np, nib, ndimage


def extract_specimen_mask(solid_mask, connectivity: int = 1):
    """Largest connected solid component with its internal holes filled.

    Returns ``(mask, n_components)``; mask is ``None`` when the volume has
    no solid voxel.
    """
    structure = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(solid_mask, structure=structure)
    if n == 0:
        return None, 0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    spec = lab == int(counts.argmax())
    del lab
    spec = ndimage.binary_fill_holes(spec)
    return spec, int(n)


def remove_background(
    img,
    thresh: float,
    *,
    margin: int = 10,
    crop: bool = True,
    fill=None,
    progress=None,
):
    """Strip everything outside the specimen from *img*.

    Parameters
    ----------
    img
        The loaded :class:`nibabel.Nifti1Image`.
    thresh
        Void/solid intensity threshold (the app's auto/Otsu value).
    margin
        Voxels of padding kept around the specimen bounding box.
    crop
        Also crop the volume to the specimen bounding box + margin.
    fill
        Value written outside the specimen. Defaults to the background
        median, which keeps a realistic air peak in the histogram.
    progress
        Optional ``fn(str)`` stage callback.

    Returns ``(new_img, info_dict)``.
    """
    def _stage(msg):
        if progress is not None:
            progress(msg)

    _stage("reading volume")
    raw = np.asanyarray(img.dataobj)
    if raw.ndim > 3:
        raw = raw[..., 0]

    _stage("finding specimen")
    spec, n_comp = extract_specimen_mask(raw >= thresh)
    if spec is None:
        raise ValueError(
            f"No solid material found at or above threshold {thresh:g}."
        )
    spec_voxels = int(spec.sum())

    _stage("removing background")
    if fill is None:
        # Median of the background, sampled on a coarse grid for speed.
        bg_sample = raw[::4, ::4, ::4][~spec[::4, ::4, ::4]]
        fill = float(np.median(bg_sample)) if bg_sample.size else 0.0
    if np.issubdtype(raw.dtype, np.integer):
        info_dt = np.iinfo(raw.dtype)
        fill_cast = int(np.clip(round(fill), info_dt.min, info_dt.max))
    else:
        fill_cast = raw.dtype.type(fill)

    out = raw.copy()
    out[~spec] = fill_cast

    affine = img.affine.copy()
    bbox = None
    if crop:
        _stage("cropping to specimen")
        obj = ndimage.find_objects(spec.astype(np.uint8))[0]
        m = int(max(0, margin))
        sl = tuple(
            slice(max(0, s.start - m), min(dim, s.stop + m))
            for s, dim in zip(obj, out.shape)
        )
        bbox = tuple((s.start, s.stop) for s in sl)
        out = np.ascontiguousarray(out[sl])
        # Shift the affine origin so world coordinates stay correct.
        offset = np.array([s.start for s in sl], dtype=float)
        affine[:3, 3] += affine[:3, :3] @ offset

    _stage("building NIfTI")
    hdr = img.header.copy()
    hdr.set_data_shape(out.shape)
    new_img = nib.Nifti1Image(out, affine, hdr)

    info = {
        "threshold": float(thresh),
        "solid_components": n_comp,
        "specimen_voxels": spec_voxels,
        "background_pct": 100.0 * (1.0 - spec_voxels / raw.size),
        "fill_value": float(fill),
        "bbox": bbox,
        "out_shape": tuple(int(s) for s in out.shape),
    }
    return new_img, info
