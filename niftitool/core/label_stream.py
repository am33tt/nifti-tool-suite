"""Connected components and cavity filling under a fixed memory budget.

The specimen in a micro-CT scan is the largest connected region of solid
voxels, and the background is everything outside it. Finding that region
with :func:`scipy.ndimage.label` needs an ``int32`` label volume, four
bytes per voxel, on top of the volume itself -- which is why the operation
used to fail on the machines the scans are actually processed on.

This module computes the same result from two-dimensional labellings that
are merged across ``z`` with a union-find, and stores the masks one bit per
voxel. Nothing here scales with the volume except the packed masks, at
0.125 bytes per voxel each.

Equivalence with scipy
----------------------
A slice labelled with the 4-connected structuring element, joined to the
next slice wherever both are foreground at the same ``(x, y)``, is exactly
the 6-connected (``connectivity=1``) labelling of the volume: the two
element sets are the same set of neighbours. :func:`fill_cavities`
likewise reproduces :func:`scipy.ndimage.binary_fill_holes` with its
default structuring element, because that too floods the background with
6-connectivity from the array border. ``tests/test_background_streaming.py``
asserts the equality voxel by voxel on random volumes.

Six-connectivity is the right choice here regardless of the memory: under
26-connectivity two grains touching at a single corner count as one solid
body, so a speck of noise diagonally adjacent to the specimen would join
it and drag the bounding box outward.
"""

from __future__ import annotations

from ..deps import ndimage, np

#: In-plane neighbourhood. With the z-links added by :func:`_label_volume`
#: this is 6-connectivity in 3-D.
_STRUCT_2D = np.array(
    [[False, True, False],
     [True, True, True],
     [False, True, False]],
)

#: Refuse to allocate a union-find bigger than this. 50 million components
#: is ~0.8 GiB of bookkeeping and far beyond any real scan; hitting it means
#: the threshold has caught noise rather than material.
MAX_COMPONENTS = 50_000_000


class PackedMask:
    """A 3-D boolean volume stored as one bit per voxel.

    Slices are read and written whole with :meth:`get` and :meth:`set`; the
    voxel count and bounding box are maintained as slices are written, so
    neither costs a second pass.
    """

    __slots__ = ("shape", "_planes", "_counts", "_boxes")

    def __init__(self, shape) -> None:
        nx, ny, nz = (int(s) for s in shape)
        self.shape = (nx, ny, nz)
        self._planes: list = [None] * nz
        self._counts = np.zeros(nz, dtype=np.int64)
        self._boxes: list = [None] * nz

    @property
    def nbytes(self) -> int:
        """Memory held by the packed planes."""
        return sum(p.nbytes for p in self._planes if p is not None)

    @property
    def count(self) -> int:
        """Number of set voxels."""
        return int(self._counts.sum())

    def get(self, z: int):
        """Slice *z* as a 2-D boolean array."""
        nx, ny, _ = self.shape
        plane = self._planes[z]
        if plane is None:
            return np.zeros((nx, ny), dtype=bool)
        flat = np.unpackbits(plane, count=nx * ny).astype(bool)
        return flat.reshape(nx, ny)

    def set(self, z: int, values) -> None:
        """Replace slice *z* with the 2-D boolean array *values*."""
        values = np.ascontiguousarray(values, dtype=bool)
        count = int(values.sum())
        self._counts[z] = count
        if count == 0:
            self._planes[z] = None
            self._boxes[z] = None
            return
        self._planes[z] = np.packbits(values.ravel())
        rows = np.flatnonzero(values.any(axis=1))
        cols = np.flatnonzero(values.any(axis=0))
        self._boxes[z] = (int(rows[0]), int(rows[-1]) + 1,
                          int(cols[0]), int(cols[-1]) + 1)

    def bounding_box(self):
        """``((x0, x1), (y0, y1), (z0, z1))`` of the set voxels, or None."""
        zs = np.flatnonzero(self._counts > 0)
        if zs.size == 0:
            return None
        boxes = [self._boxes[z] for z in zs]
        x0 = min(b[0] for b in boxes)
        x1 = max(b[1] for b in boxes)
        y0 = min(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        return ((x0, x1), (y0, y1), (int(zs[0]), int(zs[-1]) + 1))


class _UnionFind:
    """Disjoint sets over component labels, grown as slices are labelled.

    Weighted union with path halving, so the total cost is effectively
    linear in the number of merges.
    """

    __slots__ = ("_parent", "_rank", "_count", "n")

    def __init__(self) -> None:
        self._parent = np.zeros(0, dtype=np.int64)
        self._rank = np.zeros(0, dtype=np.int64)
        self._count = np.zeros(0, dtype=np.int64)
        self.n = 0

    def extend(self, counts) -> int:
        """Append one set per entry of *counts*; returns the first index.

        *counts* are voxel counts and stay attached to the label that was
        added; they are never merged, so summing them per root gives the
        size of each component exactly once.
        """
        counts = np.asarray(counts, dtype=np.int64)
        first = self.n
        self.n += counts.size
        if self.n > MAX_COMPONENTS:
            raise MemoryError(
                f"More than {MAX_COMPONENTS:,} connected components. The "
                f"threshold is almost certainly picking up noise rather "
                f"than material; raise it, or denoise the scan first."
            )
        if self.n > self._parent.size:
            capacity = max(1024, self.n * 2)
            parent = np.arange(capacity, dtype=np.int64)
            parent[:first] = self._parent[:first]
            rank = np.zeros(capacity, dtype=np.int64)
            rank[:first] = self._rank[:first]
            count = np.zeros(capacity, dtype=np.int64)
            count[:first] = self._count[:first]
            self._parent, self._rank, self._count = parent, rank, count
        self._count[first:self.n] = counts
        return first

    def find(self, i: int) -> int:
        parent = self._parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self._rank[a] < self._rank[b]:
            a, b = b, a
        self._parent[b] = a
        if self._rank[a] == self._rank[b]:
            self._rank[a] += 1

    def roots(self):
        """Root of every label, as an array indexed by label."""
        root = self._parent[:self.n].copy()
        while True:
            nxt = root[root]
            if np.array_equal(nxt, root):
                return root
            root = nxt

    def component_sizes(self, roots):
        """Voxel count of every component, indexed by its root label."""
        return np.bincount(roots, weights=self._count[:self.n],
                           minlength=self.n).astype(np.int64)


def _label_slice(values):
    """4-connected labelling of one boolean slice."""
    return ndimage.label(values, structure=_STRUCT_2D)


def _link(previous, current, previous_offset, current_offset, finder) -> None:
    """Merge the components of two adjacent slices where they overlap."""
    overlap = (previous > 0) & (current > 0)
    if not overlap.any():
        return
    above = previous[overlap].astype(np.int64) + (previous_offset - 1)
    below = current[overlap].astype(np.int64) + (current_offset - 1)
    # One integer key per pair, so the de-duplication is a 1-D unique.
    base = np.int64(finder.n + 1)
    for key in np.unique(above * base + below):
        finder.union(int(key // base), int(key % base))


def _label_volume(slice_of, nz, *, on_slice=None, progress=None, cancel=None,
                  stage=""):
    """Label a volume slice by slice and merge the labels across ``z``.

    *slice_of* is a callable returning the boolean slice at a given ``z``.
    *on_slice*, when given, is called as ``on_slice(z, labels, count,
    offset)`` while the slice is still in hand.

    Returns ``(finder, offsets)``, where ``offsets[z]`` is the global index
    of local label ``1`` in slice ``z``.
    """
    finder = _UnionFind()
    offsets = np.zeros(nz, dtype=np.int64)
    previous, previous_offset = None, 0
    for z in range(nz):
        if cancel is not None and cancel():
            raise _cancelled()
        labels, count = _label_slice(slice_of(z))
        offsets[z] = finder.n
        if count:
            sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
            finder.extend(sizes)
            if previous is not None:
                _link(previous, labels, previous_offset, offsets[z], finder)
        if on_slice is not None:
            on_slice(z, labels, count, int(offsets[z]))
        previous, previous_offset = (labels, int(offsets[z])) if count else (None, 0)
        if progress is not None and (z % 32 == 0 or z == nz - 1):
            progress(stage, (z + 1) / nz)
    return finder, offsets


def _cancelled():
    from ..utils import OperationCancelled
    return OperationCancelled("cancelled")


def _selector(roots, offsets, z, count, keep):
    """Lookup table mapping a slice's local labels to *keep*.

    Index 0 is the background and always maps to False.
    """
    table = np.zeros(count + 1, dtype=bool)
    if count:
        table[1:] = keep[roots[offsets[z]:offsets[z] + count]]
    return table


def largest_component(solid, *, progress=None, cancel=None):
    """The largest 6-connected component of *solid*, as a new packed mask.

    Returns ``(mask, n_components)``, or ``(None, 0)`` when *solid* is
    empty. The input is read twice; the second pass re-derives the same
    per-slice labelling, which is cheaper than keeping four bytes per voxel
    of labels around.
    """
    nx, ny, nz = solid.shape
    finder, offsets = _label_volume(
        solid.get, nz, progress=progress, cancel=cancel,
        stage="labelling solid",
    )
    if finder.n == 0:
        return None, 0

    roots = finder.roots()
    sizes = finder.component_sizes(roots)
    best = int(np.argmax(sizes))
    keep = np.zeros(finder.n, dtype=bool)
    keep[best] = True
    n_components = int(np.unique(roots).size)

    mask = PackedMask(solid.shape)
    for z in range(nz):
        if cancel is not None and cancel():
            raise _cancelled()
        labels, count = _label_slice(solid.get(z))
        table = _selector(roots, offsets, z, count, keep)
        mask.set(z, table[labels])
        if progress is not None and (z % 32 == 0 or z == nz - 1):
            progress("extracting specimen", (z + 1) / nz)
    return mask, n_components


def fill_cavities(mask, *, progress=None, cancel=None) -> int:
    """Add every void fully enclosed by *mask* to it, in place.

    A void is enclosed when its connected component never reaches the edge
    of the array. This is :func:`scipy.ndimage.binary_fill_holes` with the
    default structuring element, and it is what keeps internal porosity
    inside the specimen instead of counting it as background.

    Returns the number of voxels added.
    """
    nx, ny, nz = mask.shape
    touches_border: set[int] = set()

    def _outside(z):
        return ~mask.get(z)

    def _note_border(z, labels, count, offset):
        """Record the components of this slice that reach the array edge."""
        if not count:
            return
        edges = [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        if z == 0 or z == nz - 1:
            edges.append(labels.ravel())
        seen = np.unique(np.concatenate(edges))
        touches_border.update(offset + int(local) - 1 for local in seen[seen > 0])

    finder, offsets = _label_volume(
        _outside, nz, on_slice=_note_border, progress=progress, cancel=cancel,
        stage="finding cavities",
    )
    if finder.n == 0:
        return 0

    roots = finder.roots()
    exterior = np.zeros(finder.n, dtype=bool)
    if touches_border:
        exterior[roots[np.fromiter(touches_border, dtype=np.int64)]] = True
    enclosed = ~exterior

    added = 0
    for z in range(nz):
        if cancel is not None and cancel():
            raise _cancelled()
        labels, count = _label_slice(_outside(z))
        table = _selector(roots, offsets, z, count, enclosed)
        cavity = table[labels]
        if cavity.any():
            added += int(cavity.sum())
            mask.set(z, mask.get(z) | cavity)
        if progress is not None and (z % 32 == 0 or z == nz - 1):
            progress("filling cavities", (z + 1) / nz)
    return added
