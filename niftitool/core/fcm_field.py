"""Sub-voxel geometry fields for the finite cell method.

The finite cell method does not need a mesh that conforms to the specimen.
It needs an indicator ``alpha(x)`` it can integrate against, and its
accuracy comes from how well the quadrature resolves the *boundary* of
that indicator. A hard ``uint8`` mask gives the solver a staircase whose
error is fixed by the voxel size, no matter how finely the quadrature is
refined, so the method's main advantage is spent before the solve begins.

This module produces the two fields that give the boundary back, each with
a different standing:

:func:`block_volume_fraction`
    The exact material fraction of a coarse voxel, when the export grid is
    a whole-number downsample of the scan grid. This is a *measurement*:
    the fraction is counted from the fine mask, nothing is modelled. It is
    what the majority-vote downsample throws away.

:func:`signed_distance_field`
    The distance from every voxel centre to the material surface, in
    millimetres, positive inside. The surface is unchanged -- this is the
    same geometry the binary mask describes, written so that a quadrature
    rule can find the interface between voxel centres instead of snapping
    it to a voxel face. Refining the quadrature against it converges to
    the binary geometry rather than to something else.

Neither field changes the boundary-value problem. A third option -- taking
``alpha`` from the grey level itself, so a boundary voxel becomes partly
solid -- *does* change it, into a graded-stiffness problem, and is not
implemented here. Do not confuse the two when reporting results.

Memory
------
Unlike the rest of :mod:`niftitool.core`, the distance transform is not
streamable: an exact Euclidean transform needs the whole array, because the
nearest surface voxel may lie anywhere. That is acceptable because these
fields belong on the **export grid** -- the grid actually handed to the
solver, after any downsample -- which is orders of magnitude smaller than
the scan. :func:`estimate_sdf_bytes` gives the requirement, and
:func:`signed_distance_field` refuses a grid that will not fit rather than
letting the machine swap.
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

    Exact by construction: the value is the count of material voxels in the
    block divided by ``factor ** 3``, so the total material volume is
    preserved. ``fraction.sum() * factor ** 3 == mask.sum()`` holds to the
    last bit for any mask that fills a whole number of blocks, which is the
    property the majority-vote reduction below does not have -- that one
    deletes features thinner than half a block and closes pores smaller
    than half a block, biasing porosity low and the small end of the
    pore-size distribution away.

    Voxels past the last whole block are dropped, as they are in the binary
    reduction, so both stay on the same grid.
    """
    factor = max(1, int(factor))
    mask = np.asarray(mask)
    if factor == 1:
        return np.asarray(mask, dtype=np.float32)

    nx, ny, nz = trimmed_shape(mask.shape, factor)
    if min(nx, ny, nz) == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)

    # Summing the counts in float32 then dividing keeps the result exact:
    # both the count and factor**3 are small integers, and every ratio of
    # two such integers below 2**24 is representable.
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
    """Half a voxel step towards each voxel's nearest opposite-phase voxel.

    On an anisotropic grid the interface does not lie half of the *smallest*
    spacing away: it lies half a step along whichever direction the nearest
    opposite voxel actually sits in. For a surface normal to z at 0.30 mm
    per voxel, that is 0.15 mm, not 0.05 mm, and using the smallest spacing
    puts every such surface 0.10 mm out -- a systematic offset over the
    whole face, not noise.

    *idx* is the nearest-opposite-voxel index array from the transform. The
    displacement is converted to millimetres, normalised, and projected
    back onto the spacing to give the step length in that direction, which
    reduces exactly to ``0.5 * spacing[axis]`` for an axis-aligned surface.
    Components are accumulated one axis at a time to keep the peak working
    set down.
    """
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
    """Distance from each ``True`` voxel to the material surface.

    Returns the distance to the nearest ``False`` voxel *centre* minus half
    a voxel step, so the zero level sits midway between the last voxel of
    one phase and the first of the other -- the only defensible position
    for a two-phase boundary with no sub-voxel information.

    The array is padded by one voxel with *pad_value* before the transform
    and cropped after, which is how the caller chooses whether the edge of
    the array counts as a surface.
    """
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
        Voxel size along each axis, in millimetres. Anisotropic spacing is
        handled through the transform's ``sampling`` argument, so the
        result is a physical distance and not a voxel count.
    border_is_surface
        Whether material touching the edge of the array is bounded there.
        ``False`` (the default) is right for a specimen cropped out of a
        larger scan: the array edge is an arbitrary cut, the material
        continues past it, and treating it as a free surface would carve a
        spurious boundary layer into the model. Set ``True`` only when the
        volume really does contain the whole specimen with air around it.
    max_bytes
        Refuse grids whose peak working set exceeds this. ``None`` uses the
        free RAM reported by psutil, halved.

    Notes
    -----
    The half-voxel convention matters and is easy to get wrong. A discrete
    transform measures the distance to the nearest voxel *centre* of the
    opposite phase, so the last material voxel reports 1 and the first void
    voxel reports 1, while the interface lies midway between them. Half a
    voxel is therefore subtracted from each side:

        phi = (d_inside - h/2)  where material,  -(d_outside - h/2) elsewhere

    which puts the zero level exactly on the midpoint, as it should be for
    a two-phase boundary with no other information. ``h`` is the smallest
    spacing; for anisotropic voxels the exact correction depends on the
    direction to the nearest opposite voxel, so this is accurate to within
    half the spacing *anisotropy*, not half a voxel. For the near-isotropic
    grids these scans produce the difference is negligible; for strongly
    anisotropic data, resample first.

    The result is not better than the mask it is built from: a voxelised
    sphere carries its own discretisation error, and this field inherits
    it. What it removes is the *additional* error of forcing the interface
    onto a voxel face.
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

    # Degenerate volumes have no surface to measure from. Report a distance
    # that is unambiguously outside every band a caller might use, rather
    # than zeros, which would read as "interface everywhere".
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

    Reads the indicator file back from the **export grid** -- not the scan
    grid -- computes the field there, and writes it as float32 on exactly
    the same shape, affine and zooms, so the solver reads both in one
    frame. Doing it as a second pass rather than inside the streaming
    threshold is deliberate: the transform is not streamable, and the
    export grid is the smallest grid that still describes what the solver
    will see.

    Returns a summary including the surface area implied by the zero level
    and the fraction of voxels within one voxel of it, which is the share
    of the domain a cut-cell quadrature actually has to work on.
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

    With ``width_mm = 0`` this is the sharp indicator ``H(phi)``: the
    geometry is unchanged and all the accuracy comes from the quadrature
    finding the zero level. A positive *width_mm* replaces the step with a
    linear ramp of that total width centred on the surface, which is
    sometimes wanted to keep the integrand differentiable -- note that this
    grades the stiffness across the ramp and is a modelling choice, not a
    numerical one.

    *alpha_min* is the fictitious-domain penalty: the void phase is given a
    small non-zero value so the stiffness matrix stays non-singular. It
    must match the solver's ``alpha_fcm``, and it is applied as a floor so
    a nearly-void point can never drop below it and wreck the conditioning.
    """
    phi = np.asarray(phi, dtype=np.float32)
    if width_mm <= 0:
        alpha = (phi >= 0.0).astype(np.float32)
    else:
        alpha = np.clip(0.5 + phi / float(width_mm), 0.0, 1.0).astype(np.float32)
    return np.maximum(alpha, np.float32(alpha_min))
