"""Sub-voxel geometry fields for the finite cell method.

Two companion fields to the binary mask, both measured on the *export
grid* (post-downsample), not modelled from it:

:func:`block_volume_fraction`
    Exact material fraction per coarse voxel, counted from the fine mask.

:func:`signed_distance_field`
    Distance to the material surface in mm, positive inside, with a
    half-voxel correction so the zero level sits between voxel centres.

Neither changes the boundary-value problem; grading alpha from the grey
value would, and is not implemented here.

The distance transform is not streamable, so these fields are sized to the
export grid only. :func:`estimate_sdf_bytes` / :func:`signed_distance_field`
refuse a grid too large for the available RAM rather than let it swap.
"""

from __future__ import annotations

import numpy as np

from . import accel

__all__ = [
    "block_volume_fraction", "block_reduce_binary",
    "signed_distance_field", "write_sdf_nifti", "alpha_from_sdf",
    "estimate_sdf_bytes", "is_isotropic", "trimmed_shape",
]

#: Peak bytes per voxel of :func:`signed_distance_field` on an isotropic
#: grid: SciPy returns a float64 distance (8), the running float32 result
#: (4) and the float32 second transform (4). The boolean input is
#: negligible beside those.
SDF_BYTES_PER_VOXEL = 16

#: Peak bytes per voxel on an anisotropic grid, where the transform must
#: also return the nearest-voxel indices (three int32 planes) and the
#: displacement components are built from them one axis at a time.
SDF_BYTES_PER_VOXEL_ANISOTROPIC = 40


def trimmed_shape(shape, factor: int) -> tuple[int, int, int]:
    """*shape* cropped to a whole number of *factor*-sized blocks."""
    return tuple(int(s) - int(s) % int(factor) for s in shape[:3])


def block_volume_fraction(mask, factor: int):
    """Material volume fraction of each *factor*-cubed block, as float32.

    Exact: fine-mask count / factor**3, so total material volume is
    preserved (unlike the majority-vote reduction below, which loses
    sub-block features). Voxels past the last whole block are dropped,
    matching :func:`block_reduce_binary`.
    """
    factor = max(1, int(factor))
    mask = np.asarray(mask)
    if factor == 1:
        return np.asarray(mask, dtype=np.float32)

    nx, ny, nz = trimmed_shape(mask.shape, factor)
    if min(nx, ny, nz) == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)

    # int counts / factor**3 stays exact in float32 below 2**24.
    blocks = mask[:nx, :ny, :nz].astype(np.float32).reshape(
        nx // factor, factor, ny // factor, factor, nz // factor, factor)
    counts = blocks.sum(axis=(1, 3, 5))
    return (counts / float(factor ** 3)).astype(np.float32)


def block_reduce_binary(mask, factor: int):
    """Majority-vote downsample: a block is material if at least half is.

    Kept for the binary export and for backwards compatibility. Prefer
    :func:`block_volume_fraction` where the consumer can take a continuous
    field -- this one is a lossy view of it.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return np.asarray(mask, dtype=np.uint8)
    fraction = block_volume_fraction(mask, factor)
    if fraction.size == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    return (fraction >= 0.5).astype(np.uint8)


def estimate_sdf_bytes(shape, spacing=None) -> int:
    """Peak working memory of :func:`signed_distance_field` for *shape*.

    Pass *spacing* to account for the anisotropic path, which costs more
    because it needs the nearest-voxel indices as well as the distances.
    """
    voxels = 1
    for dim in shape[:3]:
        voxels *= int(dim)
    per_voxel = (SDF_BYTES_PER_VOXEL if spacing is None or is_isotropic(spacing)
                 else SDF_BYTES_PER_VOXEL_ANISOTROPIC)
    return voxels * per_voxel


#: Spacing ratios within this of 1.0 are treated as isotropic, and the
#: half-voxel correction becomes exact for them.
ISOTROPY_TOLERANCE = 1e-6


def is_isotropic(spacing) -> bool:
    """Whether *spacing* is equal along every axis to within tolerance."""
    spacing = [float(s) for s in spacing]
    return (max(spacing) - min(spacing)) <= ISOTROPY_TOLERANCE * max(spacing)


def _half_step(idx, spacing, shape):
    """Half a voxel step along the direction to each voxel's nearest
    opposite-phase voxel -- not along the smallest spacing, which on an
    anisotropic grid would put a surface systematically off by up to the
    spacing difference. *idx* is the nearest-opposite-voxel index array
    from the transform."""
    spacing = [float(s) for s in spacing]
    components = []
    norm_sq = np.zeros(shape, dtype=np.float32)
    for axis in range(3):
        extent = [1, 1, 1]
        extent[axis] = shape[axis]
        coordinate = np.arange(shape[axis], dtype=np.float32).reshape(extent)
        component = (idx[axis].astype(np.float32) - coordinate) * spacing[axis]
        norm_sq += component * component
        components.append(component)

    norm = np.sqrt(norm_sq, out=norm_sq)
    # A voxel of the opposite phase is its own nearest one, distance zero;
    # its correction is irrelevant but the division must stay finite.
    np.maximum(norm, np.float32(1e-12), out=norm)

    half = np.zeros(shape, dtype=np.float32)
    for axis, component in enumerate(components):
        half += np.abs(component / norm) * np.float32(spacing[axis])
    return 0.5 * half


def _edt(binary, spacing, *, pad_value: bool):
    """Distance from each True voxel to the material surface: distance to
    the nearest False voxel centre, minus a half-voxel correction so the
    zero level sits midway between the two phases. *pad_value* sets
    whether the array edge counts as a surface."""
    padded = np.pad(binary, 1, mode="constant", constant_values=bool(pad_value))

    if is_isotropic(spacing):
        # Every direction has the same step, so the correction is a constant
        # and the nearest-voxel indices are not needed.
        distance = accel.distance_transform_edt(padded, sampling=spacing)
        distance = np.asarray(distance, dtype=np.float32) - np.float32(
            0.5 * float(spacing[0]))
    else:
        distance, idx = accel.distance_transform_edt(
            padded, sampling=spacing, return_indices=True)
        distance = np.asarray(distance, dtype=np.float32)
        distance -= _half_step(idx, spacing, padded.shape)
        del idx

    return np.ascontiguousarray(distance[1:-1, 1:-1, 1:-1])


def signed_distance_field(mask, spacing_mm, *, border_is_surface: bool = False,
                          max_bytes: int | None = None):
    """Signed distance to the material surface, in mm, positive inside.

    Parameters
    ----------
    mask
        Boolean or 0/1 material mask on the export grid.
    spacing_mm
        Voxel size per axis, in mm.
    border_is_surface
        False (default): the array edge is a cut, not a surface (right for
        a specimen cropped from a larger scan). True: material touching
        the edge is bounded there.
    max_bytes
        Refuse grids exceeding this working set. None uses half of free
        RAM (see :func:`estimate_sdf_bytes`).

    Zero level sits midway between the last material and first void voxel
    centre (see :func:`_edt`); exact for isotropic spacing, accurate to
    within the spacing anisotropy otherwise.
    """
    mask = np.asarray(mask, dtype=bool)
    spacing = tuple(float(s) for s in np.asarray(spacing_mm).ravel()[:3])
    if len(spacing) != 3 or not all(s > 0 for s in spacing):
        raise ValueError(f"spacing_mm must be three positive values, got {spacing_mm!r}")

    needed = estimate_sdf_bytes(mask.shape, spacing)
    if max_bytes is None:
        from ..utils import available_ram_mb
        free_mb = available_ram_mb()
        max_bytes = int(0.5 * free_mb * 1024 ** 2) if free_mb else None
    if max_bytes is not None and needed > max_bytes:
        factor = int(np.ceil((needed / max_bytes) ** (1.0 / 3.0)))
        raise MemoryError(
            f"A signed distance field over {mask.shape} needs about "
            f"{needed / 1024 ** 3:.1f} GiB, more than the "
            f"{max_bytes / 1024 ** 3:.1f} GiB budget. The distance "
            f"transform cannot be streamed. Export at a downsample factor "
            f"of {factor} or more, or raise max_bytes."
        )

    # No surface in a degenerate volume: report a distance outside any
    # band a caller might use, not zero (which reads as "interface everywhere").
    extent = float(np.hypot.reduce([n * s for n, s in zip(mask.shape, spacing)]))
    if not mask.any():
        return np.full(mask.shape, -extent, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, extent, dtype=np.float32)

    # Inside: distance to the nearest void. Padding with True (material)
    # when the border is not a surface stops the array edge acting as one.
    d_in = _edt(mask, spacing, pad_value=not border_is_surface)
    # Outside: distance to the nearest material, with the complementary pad.
    d_out = _edt(~mask, spacing, pad_value=border_is_surface)

    phi = np.where(mask, d_in, -d_out)
    return np.asarray(phi, dtype=np.float32)


def write_sdf_nifti(mask_path, out_path, *, border_is_surface: bool = False,
                    max_bytes: int | None = None) -> dict:
    """Write the signed distance field of an exported binary mask.

    Reads the mask back from the export grid (not the scan grid), computes
    the field there, and writes float32 on the same shape/affine/zooms so
    the solver reads both in one frame. Returns a summary including the
    fraction of voxels within one voxel of the zero level.
    """
    import nibabel as nib

    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj) > 0
    spacing = tuple(float(z) for z in mask_img.header.get_zooms()[:3])

    phi = signed_distance_field(
        mask, spacing, border_is_surface=border_is_surface,
        max_bytes=max_bytes)

    out = nib.Nifti1Image(phi, mask_img.affine, dtype=np.float32)
    out.header.set_zooms(spacing)
    out.header['descrip'] = b'FCM signed distance, mm, + inside'
    out.header['cal_min'] = float(phi.min())
    out.header['cal_max'] = float(phi.max())
    nib.save(out, str(out_path))

    band = float(np.count_nonzero(np.abs(phi) <= max(spacing)))
    return {
        "path": str(out_path),
        "shape": tuple(int(s) for s in phi.shape),
        "spacing_mm": spacing,
        "isotropic": is_isotropic(spacing),
        "border_is_surface": bool(border_is_surface),
        "phi_min_mm": float(phi.min()),
        "phi_max_mm": float(phi.max()),
        "cut_voxel_fraction": band / max(1, phi.size),
    }


def alpha_from_sdf(phi, *, width_mm: float = 0.0, alpha_min: float = 1e-8):
    """FCM indicator from a signed distance field.

    ``width_mm = 0`` (default): sharp indicator ``H(phi)``, geometry
    unchanged. ``width_mm > 0``: linear ramp of that width centred on the
    surface -- a graded-stiffness modelling choice, not a numerical one.

    ``alpha_min`` is the fictitious-domain floor (must match the solver's
    ``alpha_fcm``) so the stiffness matrix stays non-singular in the void.
    """
    phi = np.asarray(phi, dtype=np.float32)
    if width_mm <= 0:
        alpha = (phi >= 0.0).astype(np.float32)
    else:
        alpha = np.clip(0.5 + phi / float(width_mm), 0.0, 1.0).astype(np.float32)
    return np.maximum(alpha, np.float32(alpha_min))
