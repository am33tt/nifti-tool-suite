"""Geometric operations on NIfTI volumes.

Each operation updates the voxel array and the affine/header together so
downstream tools see a coherent volume.

1. :func:`run_reorientation` permutes and flips axes so the anatomical
   codes match a target such as ``"RAS"``. No resampling, so every physical
   point keeps its world coordinate.
2. :func:`run_angle_rotation` resamples at an arbitrary angle with bicubic
   spline interpolation (``order=3``), which blurs the volume slightly.
3. :func:`run_cropper` applies an integer index slice and updates the
   affine origin. Lossless.

Every function returns a fresh :class:`nibabel.Nifti1Image` with updated
qform/sform.
"""

from __future__ import annotations

from ..deps import np, nib, nio, ndimage
from .io import raw_array


# Reorientation

def run_reorientation(img, target_orientation: str):
    """Reorient *img* to match the three-letter anatomical code
    *target_orientation* (e.g. ``"RAS"``, ``"LPS"``).

    Pure axis swap and flip via
    :func:`nibabel.orientations.ornt_transform`; the volume is not resampled.
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


# Arbitrary-angle rotation

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

    Uses :func:`scipy.ndimage.rotate` with ``reshape=False``, so the grid
    dimensions are unchanged. Structured RGB dtypes are rotated channel by
    channel and reassembled. The affine is updated so the rotation acts
    about the world-space volume centre, not the corner.
    """
    data = raw_array(img)
    axis_plane = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}
    kw = dict(
        angle=angle_deg, axes=axis_plane[axis],
        reshape=False, order=3, mode='constant', cval=0.0,
    )

    if data.dtype.names:
        # Structured RGB: rotate each channel, then reassemble.
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


# Cropping

def run_cropper(img, x_range, y_range, z_range):
    """Crop *img* to the closed-open voxel index ranges provided.

    The affine translation is shifted so the cropped volume keeps its
    world position: the voxel at index ``(x0, y0, z0)`` stays at the same
    millimetre coordinate.
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
