"""Slab-wise reading and writing of NIfTI volumes.

Nothing in this module ever holds a whole volume in memory, so the cost of
an operation is set by :data:`SLAB_BUDGET_BYTES` rather than by the size of
the file.

* :class:`RawSliceReader` yields the voxels **exactly as they are stored on
  disk**: the on-disk dtype, byte order and ``scl_slope``/``scl_inter`` are
  left untouched, and the grey levels a threshold is compared against are
  produced on a copy of one slab at a time.
* :class:`StreamingNiftiWriter` writes a NIfTI-1 file slab by slab. The
  header it emits is the one nibabel would write for the same shape, dtype,
  affine and source header, and the voxels follow NIfTI storage order, so
  the result is an ordinary NIfTI-1 file with no trace of how it was made.

Together they let a filter that touches every voxel run on a volume larger
than the machine's RAM without resampling, rescaling or re-quantising it.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from ..deps import nib, np

#: Voxels read into RAM at once. Slabs amortise the per-read overhead --
#: which dominates on compressed files -- without scaling with the volume.
SLAB_BUDGET_BYTES = 64 * 1024 ** 2

#: gzip level used for ``.nii.gz`` output. Matches nibabel's default, so a
#: file written here is the same size as one written by ``nib.save``.
GZIP_COMPRESSLEVEL = 1

#: Byte offset of the voxel data in a single-file NIfTI-1: the 348-byte
#: header plus the 4-byte extension flag.
NIFTI1_DATA_OFFSET = 352


def is_gzipped(path) -> bool:
    """True when *path* names a gzip-compressed NIfTI (``.nii.gz``)."""
    return str(path).lower().endswith('.gz')


class RawSliceReader:
    """Forward, slab-wise reader for the unscaled voxels of a NIfTI image.

    Parameters
    ----------
    img
        A loaded :class:`nibabel.Nifti1Image` (or NIfTI-2 image).

    Attributes
    ----------
    shape
        The three spatial dimensions.
    extra_shape
        Any further dimensions, e.g. ``(n_volumes,)`` for a 4-D series.
    dtype
        The on-disk voxel dtype, byte order included.
    slope, inter
        The header's intensity scaling, ``(1.0, 0.0)`` when unset.

    Notes
    -----
    Slabs are read through a private :class:`nibabel.arrayproxy.ArrayProxy`
    built with unit scaling, which is what keeps the returned values raw.
    Uncompressed files are memory-mapped; compressed files are decompressed
    on the fly, so slabs must be requested in increasing ``z`` -- going
    backwards forces gzip to restart from the beginning of the file.
    """

    def __init__(self, img) -> None:
        self.img = img
        full_shape = tuple(int(s) for s in img.shape)
        self.shape = full_shape[:3]
        self.extra_shape = full_shape[3:]

        dataobj = img.dataobj
        if hasattr(dataobj, 'offset'):
            # A proxy over a file. Its own attributes, not the header's, are
            # what nibabel reads the file with; a loaded header reports a
            # vox_offset of 0 and an unset slope, which the proxy resolves.
            self.dtype = np.dtype(dataobj.dtype)
            self.slope = self._finite(dataobj.slope, 1.0)
            self.inter = self._finite(dataobj.inter, 0.0)
            self._proxy = self._unscaled_proxy(img, dataobj)
        else:
            # An image already held in memory, e.g. the result of a crop.
            self._proxy = np.asanyarray(dataobj)
            self.dtype = self._proxy.dtype
            self.slope, self.inter = 1.0, 0.0

        names = self.dtype.names
        self.is_rgb = bool(names is not None and 'R' in names)

    @staticmethod
    def _finite(value, default: float) -> float:
        """*value* as a float, falling back when it is unset or not finite."""
        if value is None:
            return default
        value = float(value)
        return value if np.isfinite(value) else default

    @staticmethod
    def _unscaled_proxy(img, dataobj):
        """A proxy over the same voxels with the intensity scaling disabled."""
        from nibabel.arrayproxy import ArrayProxy

        spec = (
            tuple(int(s) for s in img.shape),
            np.dtype(dataobj.dtype),
            int(dataobj.offset),
            1.0,
            0.0,
        )
        holder = img.file_map['image']
        source = holder.filename if holder.filename is not None else holder.fileobj
        # keep_file_open matters for .nii.gz: without it every read reopens
        # the file and decompresses it from the start again.
        return ArrayProxy(source, spec, mmap=True, keep_file_open=True)

    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def slab_depth(self) -> int:
        """Number of z-slices read at once, at least one."""
        nx, ny, _ = self.shape
        per_slice = nx * ny * self.dtype.itemsize
        for n in self.extra_shape:
            per_slice *= int(n)
        return max(1, int(SLAB_BUDGET_BYTES // max(1, per_slice)))

    def read(self, z0: int, z1: int, *, index=(), x=None, y=None):
        """Raw voxels of ``[x, y, z0:z1] + index``, in the on-disk dtype.

        *x* and *y* are optional ``(start, stop)`` pairs restricting the
        in-plane extent; *index* selects one entry of each extra dimension.
        """
        nx, ny, _ = self.shape
        x0, x1 = x if x is not None else (0, nx)
        y0, y1 = y if y is not None else (0, ny)
        key = (slice(x0, x1), slice(y0, y1), slice(z0, z1)) + tuple(index)
        return self._proxy[key]

    def iter_slabs(self, *, index=(), x=None, y=None, z=None):
        """Yield ``(z_start, raw_slab)`` covering *z* in increasing order."""
        _, _, nz = self.shape
        z0, z1 = z if z is not None else (0, nz)
        depth = self.slab_depth
        for za in range(z0, z1, depth):
            zb = min(za + depth, z1)
            yield za, self.read(za, zb, index=index, x=x, y=y)

    def to_gray(self, raw):
        """Scalar float32 grey levels of a raw slab.

        Applies the header's intensity scaling, averages the three colour
        channels of an RGB volume and the leading channels of a 4-D stack,
        so the values match what the viewer and the thresholds work with.
        """
        if self.is_rgb:
            return (
                raw['R'].astype(np.float32)
                + raw['G'].astype(np.float32)
                + raw['B'].astype(np.float32)
            ) / 3.0
        if raw.ndim == 4:
            gray = raw[..., :3].mean(axis=-1).astype(np.float32)
        else:
            gray = raw.astype(np.float32, copy=False)
        if self.slope != 1.0 or self.inter != 0.0:
            gray = gray * np.float32(self.slope) + np.float32(self.inter)
        return gray

    def solid_mask(self, raw, threshold: float):
        """``raw`` at or above the grey level *threshold*, as booleans.

        For the common case -- a scalar integer volume with no header
        scaling -- the threshold is moved into stored units instead, which
        avoids building a floating-point copy of the slab. For integers
        ``value >= t`` and ``value >= ceil(t)`` select the same voxels, so
        this is a shortcut and not an approximation.
        """
        plain = (
            not self.is_rgb
            and raw.ndim == 3
            and self.slope == 1.0
            and self.inter == 0.0
            and np.issubdtype(self.dtype, np.integer)
        )
        if plain:
            limits = np.iinfo(self.dtype)
            cut = int(np.ceil(threshold))
            if cut <= limits.min:
                return np.ones(raw.shape, dtype=bool)
            if cut > limits.max:
                return np.zeros(raw.shape, dtype=bool)
            return raw >= self.dtype.type(cut)
        return self.to_gray(raw) >= np.float32(threshold)

    def raw_value(self, scaled: float):
        """The stored value whose scaled grey level is *scaled*.

        Inverts ``scl_slope``/``scl_inter`` and rounds and clips to the
        on-disk dtype, so writing it back reproduces *scaled* as closely as
        the file's quantisation allows.
        """
        raw = (float(scaled) - self.inter) / (self.slope or 1.0)
        if self.is_rgb:
            field = self.dtype[0]
            limits = np.iinfo(field)
            value = int(np.clip(round(raw), limits.min, limits.max))
            return np.array((value, value, value), dtype=self.dtype)
        if np.issubdtype(self.dtype, np.integer):
            limits = np.iinfo(self.dtype)
            return self.dtype.type(np.clip(round(raw), limits.min, limits.max))
        return self.dtype.type(raw)

    def scaled_value(self, raw) -> float:
        """The grey level a stored value *raw* stands for."""
        if self.is_rgb:
            return float(np.mean([float(raw[name]) for name in ('R', 'G', 'B')]))
        return float(raw) * self.slope + self.inter


def build_header(source_header, affine, shape, dtype, *, slope=1.0, inter=0.0):
    """The NIfTI-1 header nibabel would write for this image.

    Mirrors :meth:`nibabel.nifti1.Nifti1Pair.update_header`: the affine is
    only pushed into the header when it differs from the one already there,
    and then lands in the sform (code ``aligned``) with the qform marked
    ``unknown``. The intensity scaling is carried over from the source,
    which is what makes writing the stored voxels lossless.
    """
    header = nib.Nifti1Header.from_header(source_header)
    header.set_data_dtype(dtype)
    header.set_data_shape(tuple(int(s) for s in shape))
    header.set_slope_inter(float(slope), float(inter))
    if affine is not None and not np.allclose(affine, header.get_best_affine()):
        header.set_sform(affine, code='aligned')
        header.set_qform(affine, code='unknown')
    header['magic'] = header.single_magic
    header.set_data_offset(NIFTI1_DATA_OFFSET)
    return header


class StreamingNiftiWriter:
    """Write a NIfTI-1 file one slab at a time.

    Use as a context manager and feed it blocks with :meth:`write_block`, in
    the order the format stores them: ``x`` fastest, then ``y``, then ``z``,
    then any further dimension. For a 4-D image that means the whole first
    volume, then the whole second, and so on.

    The file is complete and self-describing; :func:`nibabel.load` cannot
    tell it apart from one written by :func:`nibabel.save`.
    """

    def __init__(self, path, *, source_header, affine, shape, dtype=None,
                 slope=1.0, inter=0.0) -> None:
        self.path = Path(path)
        self.shape = tuple(int(s) for s in shape)
        self.header = build_header(
            source_header,
            affine,
            self.shape,
            source_header.get_data_dtype() if dtype is None else dtype,
            slope=slope,
            inter=inter,
        )
        self.dtype = self.header.get_data_dtype()
        self._fileobj = None
        self._written = 0

    @property
    def expected_bytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __enter__(self):
        if is_gzipped(self.path):
            self._fileobj = gzip.GzipFile(
                str(self.path), mode='wb', compresslevel=GZIP_COMPRESSLEVEL,
            )
        else:
            self._fileobj = open(self.path, 'wb')
        # write_to emits the 348-byte header and the 4-byte extension flag,
        # which is exactly NIFTI1_DATA_OFFSET.
        self.header.write_to(self._fileobj)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._fileobj.close()
        finally:
            self._fileobj = None
        # A half-written file is not a NIfTI file: remove it rather than
        # leave something that loads but is silently truncated.
        if exc_type is not None:
            self.path.unlink(missing_ok=True)
            return False
        if self._written != self.expected_bytes:
            self.path.unlink(missing_ok=True)
            raise IOError(
                f"{self.path.name}: wrote {self._written} voxel bytes, "
                f"expected {self.expected_bytes}."
            )
        return False

    def write_block(self, block) -> None:
        """Append *block*, a sub-volume whose first axis varies fastest."""
        block = np.asanyarray(block)
        if block.dtype != self.dtype:
            block = block.astype(self.dtype, copy=False)
        payload = block.tobytes(order='F')
        self._fileobj.write(payload)
        self._written += len(payload)
