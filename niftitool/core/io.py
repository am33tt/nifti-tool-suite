"""File I/O primitives for NIfTI volumes.

* :func:`load_nifti` wraps :func:`nibabel.load`.
* :func:`raw_array` materialises the voxel array from a nibabel proxy.
* :func:`to_gray` collapses any dtype (structured RGB, 4-D, scalar) to a
  float32 scalar volume for windowing and HU calibration.
* :class:`LazyGrayVolume` wraps a nibabel ``ArrayProxy`` so the file is not
  materialised eagerly; slicing returns float32 on demand.
* :func:`run_gunzip` decompresses ``.nii.gz`` to ``.nii``.
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

    Handles three layouts:

    1. Structured RGB (dtype with fields ``'R','G','B'``), averaged with
       equal weights; photometric luma does not apply to CT.
    2. 4-D stacks of shape ``(X, Y, Z, C)``, first three channels averaged.
    3. Scalar volumes, cast to ``float32``.
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

    The proxy is memory-mapped (``.nii``) or decoded lazily (``.nii.gz``) by
    nibabel, so slicing reads only the requested bytes.

    Implements the ndarray surface the viewer uses on the hot path:
    ``shape``, ``dtype``, ``ndim``, ``size`` and ``__getitem__``. For exact
    full-array reductions call :meth:`to_array` once.
    """

    __slots__ = ("_proxy", "_is_rgb", "_is_4d", "shape", "dtype", "ndim", "_full")

    def __init__(self, dataobj) -> None:
        self._proxy = dataobj
        self._full = None  # cache for to_array()

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
        per-channel axis, which is the case for 4-D inputs.
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
        # 4-D proxies need a trailing full slice so every channel is read
        # and averaged into a scalar slice.
        if self._is_4d:
            if isinstance(key, tuple):
                key = key + (slice(None),)
            else:
                key = (key, slice(None), slice(None), slice(None))
        raw = self._proxy[key]
        return self._convert(raw, channel_axis=self._is_4d)

    def to_array(self):
        """Materialise the full float32 scalar volume, cached after the first
        call.

        Only for operations needing every voxel, such as exact statistics or
        a full-volume histogram.
        """
        if self._full is None:
            raw = np.asanyarray(self._proxy)
            self._full = self._convert(raw, channel_axis=self._is_4d)
        return self._full

    # ndarray-like reductions served from a subsample, and therefore
    # approximate. The auto-window helper calls .min() and .max() on the hot
    # path, where materialising the full array would defeat the lazy load.

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
        """Flat 1-D float32 sample representative of the whole volume.

        Reads Z slices spread evenly across the stack rather than only the
        middle, which in stacked-specimen scans can be an air gap and would
        bias any threshold derived from the sample.
        """
        sx, sy, sz = self.shape
        n_slices = int(min(sz, 9))
        zs = sorted({int(round(z)) for z in np.linspace(0, sz - 1, n_slices)})
        per_slice = max(1, max_voxels // max(1, len(zs)))
        parts = []
        for z in zs:
            sl = self[:, :, z].ravel()
            if sl.size > per_slice:
                sl = sl[:: max(1, sl.size // per_slice)]
            parts.append(sl)
        return np.concatenate(parts)


def preview_volume(volume, max_voxels: int = 150_000_000, progress=None):
    """Strided float32 copy of *volume* small enough to plot or report.

    Accepts ndarrays and :class:`LazyGrayVolume`, reading one Z slice at a
    time so the full-resolution volume is never materialised. Returns
    ``(preview, stride)``; ``stride == 1`` means the input was already small
    enough and is returned unchanged.
    """
    sx, sy, sz = volume.shape[:3]
    total = sx * sy * sz
    stride = 1
    while total / stride**3 > max_voxels:
        stride += 1
    if stride == 1 and isinstance(volume, np.ndarray):
        return volume, 1

    zids = range(0, sz, stride)
    out = np.empty(
        (len(range(0, sx, stride)), len(range(0, sy, stride)), len(zids)),
        dtype=np.float32,
    )
    for i, z in enumerate(zids):
        sl = np.asarray(volume[:, :, z])
        out[:, :, i] = sl[::stride, ::stride]
        if progress is not None and i % 50 == 0:
            progress(i + 1, len(zids))
    return out, stride


def run_gunzip(gz_path: str | Path) -> Path:
    """Decompress ``*.nii.gz`` to ``*.nii`` in the same directory.

    Raises :class:`ValueError` if the input does not end in ``.gz``. Returns
    the path of the written file.
    """
    gz_path = Path(gz_path)
    if gz_path.suffix != '.gz':
        raise ValueError("File does not end in .gz")
    out_path = gz_path.with_suffix('')
    with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path
