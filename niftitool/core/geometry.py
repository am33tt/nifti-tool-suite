"""Geometric operations on NIfTI volumes.

Three distinct operations live here, all of which update both the voxel
array **and** the affine/header so that downstream tools (Paraview,
Slicer, a simulation mesh generator, …) see a coherent volume.

1. :func:`run_reorientation` — a *label-swap only* operation. The voxel
   order is permuted/flipped so the anatomical axis codes match the
   requested target (e.g. ``"RAS"``), but every physical point keeps
   exactly the same world coordinate.  No resampling, no interpolation,
   no information loss.

2. :func:`run_angle_rotation` — an *arbitrary-angle* resample using
   bicubic spline interpolation (``order=3``). The affine is rotated so
   the centre of the volume is invariant in world space.  This does
   resample and therefore slightly blurs the volume; use sparingly.

3. :func:`run_cropper` — an integer index slice plus affine origin
   update.  Losslessly trims the volume to a region of interest.

Every function returns a fresh :class:`nibabel.Nifti1Image` with a
properly updated qform/sform.
"""

from __future__ import annotations

from ..deps import np, nib, nio, ndimage
from .io import raw_array


# ── Reorientation ────────────────────────────────────────────────────────────

def run_reorientation(img, target_orientation: str):
    """Reorient *img* to match the three-letter anatomical code
    *target_orientation* (e.g. ``"RAS"``, ``"LPS"``).

    Implemented with :func:`nibabel.orientations.ornt_transform` — this
    is a pure axis swap/flip and never resamples the volume.
    """
    target_orientation = target_orientation.upper()
    cur  = nio.io_orientation(img.affine)
    tgt  = nio.axcodes2ornt(tuple(target_orientation))
    xfm  = nio.ornt_transform(cur, tgt)
    reor = img.as_reoriented(xfm)
    aff  = reor.affine
    hdr  = reor.header.copy()
    q_code = int(img.header.get_qform(coded=True)[1])
    s_code = int(img.header.get_sform(coded=True)[1])
    hdr.set_qform(aff, code=q_code)
    hdr.set_sform(aff, code=s_code)
    return nib.Nifti1Image(np.asanyarray(reor.dataobj), aff, hdr)


# ── Arbitrary-angle rotation ─────────────────────────────────────────────────

def _rot_matrix(axis: str, angle_deg: float):
    """3×3 rotation matrix around one of the three cardinal axes."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def run_angle_rotation(img, axis: str, angle_deg: float):
    """Rotate *img* by *angle_deg* around the requested cardinal axis.

    Uses :func:`scipy.ndimage.rotate` with bicubic interpolation and
    ``reshape=False`` so the output grid dimensions stay put. Structured
    RGB dtypes are rotated channel-by-channel and reassembled.

    The affine is updated so that the **world-space centre** of the
    volume is invariant across the rotation — i.e. the rotation acts
    about the centre, not the corner.
    """
    data = raw_array(img)
    axis_plane = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}
    kw = dict(
        angle=angle_deg, axes=axis_plane[axis],
        reshape=False, order=3, mode='constant', cval=0.0,
    )

    if data.dtype.names:
        # Structured RGB — rotate each channel, rebuild structured array.
        channels = {}
        for ch in data.dtype.names:
            rot = ndimage.rotate(data[ch].astype(np.float32), **kw)
            channels[ch] = np.clip(rot, 0, 255).astype(data.dtype[ch])
        rot_data = np.zeros(channels[data.dtype.names[0]].shape, dtype=data.dtype)
        for ch in data.dtype.names:
            rot_data[ch] = channels[ch]
    else:
        rot_data = ndimage.rotate(data, **kw)

    # Update the affine so the rotation is about the volume centre.
    center  = np.array(data.shape[:3]) / 2.0
    R       = _rot_matrix(axis, angle_deg)
    m_      = img.affine[:3, :3]
    cw      = m_ @ center + img.affine[:3, 3]
    new_m   = R @ m_
    new_t   = cw - new_m @ center
    new_aff = img.affine.copy()
    new_aff[:3, :3] = new_m
    new_aff[:3, 3]  = new_t

    hdr = img.header.copy()
    hdr.set_data_shape(rot_data.shape)
    q_code = int(img.header.get_qform(coded=True)[1])
    s_code = int(img.header.get_sform(coded=True)[1])
    hdr.set_qform(new_aff, code=q_code)
    hdr.set_sform(new_aff, code=s_code)
    return nib.Nifti1Image(rot_data, new_aff, hdr)


# ── Cropping ─────────────────────────────────────────────────────────────────

def run_cropper(img, x_range, y_range, z_range):
    """Crop *img* to the closed-open voxel index ranges provided.

    The affine's translation component is shifted so the cropped volume
    keeps its real-world position — the voxel that was at index
    ``(x0, y0, z0)`` before is still at the same millimetre coordinate
    afterwards.
    """
    shape = img.shape
    for (s, e), dim, ax in zip([x_range, y_range, z_range], shape, 'XYZ'):
        if s < 0 or e > dim or s >= e:
            raise ValueError(f"Invalid {ax} range [{s}:{e}] for size {dim}")
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    cropped = np.asanyarray(img.dataobj[x0:x1, y0:y1, z0:z1])
    off     = np.array([x0, y0, z0])
    new_t   = img.affine[:3, :3] @ off + img.affine[:3, 3]
    new_aff = img.affine.copy()
    new_aff[:3, 3] = new_t
    hdr = img.header.copy()
    hdr.set_data_shape(cropped.shape)
    return nib.Nifti1Image(cropped, new_aff, hdr)
