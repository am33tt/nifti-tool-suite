"""File I/O primitives for NIfTI volumes.

Responsibilities
----------------
* :func:`load_nifti` — wraps :func:`nibabel.load`.
* :func:`raw_array`  — materialises the voxel array from a nibabel proxy.
* :func:`to_gray`    — collapses any dtype (including RGB-structured and
  4-D) down to a float32 scalar volume suitable for windowing and HU
  calibration.
* :class:`LazyGrayVolume` — wraps a nibabel ``ArrayProxy`` so the file is
  not eagerly materialised. Slicing returns a float32 2-D/3-D array
  on demand, which is what makes loading near-instant for large files
  (Slicer-style: just memory-map the header and read pixels as needed).
* :func:`run_gunzip` — stand-alone ``.nii.gz`` → ``.nii`` decompressor.

All heavy work is deferred to numpy / nibabel so the functions are thin.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from ..deps import np, nib


def load_nifti(path: str | Path):
    """Open a NIfTI-1 or NIfTI-2 file. Returns a :class:`nibabel.Nifti1Image`."""
    return nib.load(str(path))


def raw_array(img):
    """Return the voxel array as a numpy view (no copy if possible)."""
    return np.asanyarray(img.dataobj)


def to_gray(data):
    """Reduce any voxel array to a float32 scalar volume.

    Handles three common layouts:

    1. **Structured RGB** (dtype has fields ``'R','G','B'``) — averaged into
       luminance using equal weights (no photometric luma; CT is not perceptual).
    2. **4-D stacks** of shape ``(X, Y, Z, C)`` — first three channels averaged.
    3. **Plain scalar** volumes — cast to ``float32``.
    """
    if data.dtype.names and 'R' in data.dtype.names:
        return (
            (
                data['R'].astype(np.float32)
                + data['G'].astype(np.float32)
                + data['B'].astype(np.float32)
            )
            / 3.0
        )
    if data.ndim == 4:
        return data[..., :3].mean(axis=-1).astype(np.float32)
    return data.astype(np.float32)


class LazyGrayVolume:
    """Float32 scalar 3-D view of a nibabel ``ArrayProxy``.

    The proxy is memory-mapped (for ``.nii``) or lazily decoded (for
    ``.nii.gz``) by nibabel.  We do **not** materialise the whole volume on
    load — slicing pulls only the bytes that are asked for, which is what
    keeps the open-file step near-instant for multi-hundred-MB volumes.

    Quacks like an ndarray for the operations the viewer hits on the hot
    path: ``shape``, ``dtype``, ``ndim``, ``size`` and ``__getitem__``.
    For full-array reductions (``min``/``mean``/``percentile``/...), call
    :meth:`to_array` once and operate on the result.
    """

    __slots__ = ("_proxy", "_is_rgb", "_is_4d", "shape", "dtype", "ndim", "_full")

    def __init__(self, dataobj) -> None:
        self._proxy = dataobj
        self._full = None  # cache for to_array()

        # Detect RGB structured dtype vs plain scalar vs 4-D channel stack.
        proxy_dtype = getattr(dataobj, 'dtype', None)
        self._is_rgb = bool(
            proxy_dtype is not None
            and proxy_dtype.names is not None
            and 'R' in proxy_dtype.names
        )
        self._is_4d = (not self._is_rgb) and dataobj.ndim == 4

        if self._is_4d:
            self.shape = tuple(dataobj.shape[:3])
        else:
            self.shape = tuple(dataobj.shape[:3])
        self.dtype = np.float32
        self.ndim = 3

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def _convert(self, raw, channel_axis=False):
        """Cast a raw proxy slice/volume into a float32 scalar array.

        ``channel_axis`` says whether ``raw`` still has the trailing
        per-channel axis (true for 4-D inputs that we always read with a
        full last-axis slice).
        """
        if self._is_rgb:
            return (
                raw['R'].astype(np.float32)
                + raw['G'].astype(np.float32)
                + raw['B'].astype(np.float32)
            ) / 3.0
        if channel_axis:
            return raw[..., :3].mean(axis=-1).astype(np.float32)
        return raw.astype(np.float32, copy=False)

    def __getitem__(self, key):
        # 4-D proxies need an extra trailing slice so we read all channels
        # and average them down to a scalar slice.
        if self._is_4d:
            if isinstance(key, tuple):
                key = key + (slice(None),)
            else:
                key = (key, slice(None), slice(None), slice(None))
        raw = self._proxy[key]
        return self._convert(raw, channel_axis=self._is_4d)

    def to_array(self):
        """Materialise the full float32 scalar volume (cached after first call).

        Use this only when you genuinely need the entire array — full
        statistics, histogram over every voxel, calibration, E-Map, etc.
        """
        if self._full is None:
            raw = np.asanyarray(self._proxy)
            self._full = self._convert(raw, channel_axis=self._is_4d)
        return self._full

    # ── ndarray-like reductions (cheap, subsample-based) ────────────────────
    #
    # The triplanar viewer's auto-window helper calls ``.min()`` / ``.max()``
    # on the volume.  Materialising the full array there would defeat the
    # whole point of the lazy load, so we serve those reductions from a
    # representative subsample.  Slicer does the same thing.

    def min(self):
        s = self.subsample_flat(1_000_000)
        return float(s.min()) if s.size else 0.0

    def max(self):
        s = self.subsample_flat(1_000_000)
        return float(s.max()) if s.size else 0.0

    def mean(self):
        s = self.subsample_flat(1_000_000)
        return float(s.mean()) if s.size else 0.0

    def std(self):
        s = self.subsample_flat(1_000_000)
        return float(s.std()) if s.size else 0.0

    def ravel(self):
        return self.to_array().ravel()

    def __array__(self, dtype=None):
        a = self.to_array()
        return a.astype(dtype, copy=False) if dtype is not None else a

    def subsample_flat(self, max_voxels: int):
        """Cheap flat float32 sample for the live stats panel.

        Reads a single contiguous middle Z-slab and strides through it,
        avoiding a full materialisation.  Returned array is 1-D float32.
        """
        sx, sy, sz = self.shape
        # Slab thickness — 1 slice for huge volumes is enough for stats.
        slab = max(1, min(sz, max(1, max_voxels // max(1, sx * sy))))
        z0 = max(0, (sz - slab) // 2)
        z1 = z0 + slab
        block = self[:, :, z0:z1].ravel()
        if block.size > max_voxels:
            step = block.size // max_voxels
            block = block[::step]
        return block


def run_gunzip(gz_path: str | Path) -> Path:
    """Decompress ``*.nii.gz`` → ``*.nii`` into the same directory.

    Raises :class:`ValueError` if the input does not end in ``.gz``.
    Returns the path of the written file.
    """
    gz_path = Path(gz_path)
    if gz_path.suffix != '.gz':
        raise ValueError("File does not end in .gz")
    out_path = gz_path.with_suffix('')
    with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path
