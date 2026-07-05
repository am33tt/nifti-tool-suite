"""Per-void (pore) analytics on a segmented volume.

Connected-component analysis of the void phase:

* label every disconnected air pocket inside the specimen,
* measure each one (volume, equivalent diameter, sphericity, elongation,
  centroid, bounding box),
* summarise the population (count, size distribution percentiles,
  nearest-neighbour spacing).

This is the analysis AM-concrete / porous-media papers report, so the
numbers here feed both the Porosity tab and the PDF report generator.

Design notes
------------
* The exterior air surrounding the specimen is (optionally, default on)
  removed by discarding components that touch the volume boundary —
  otherwise it would register as one giant "void" and swamp the stats.
* Sphericity needs a surface area.  A triangulated surface from marching
  cubes is accurate but costs time per void, so it's only computed for
  voids with at least ``sphericity_min_voxels`` voxels (below that the
  mesh is too coarse to mean anything anyway).
* Everything returns plain floats / numpy arrays so the module stays
  GUI-free and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..deps import np, ndimage, marching_cubes, HAS_SKIMAGE


# Voids smaller than this get NaN sphericity (mesh too coarse).
DEFAULT_SPHERICITY_MIN_VOXELS = 27          # 3×3×3
# Detailed per-void metrics are computed for at most this many voids
# (largest first); the rest still count toward totals / porosity.
DEFAULT_MAX_DETAILED_VOIDS = 20_000
# Working memory the labelling stage needs per analysed voxel:
# bool mask (1) + int32 labels (4) + border bool temp (1) + slack.
BYTES_PER_ANALYSIS_VOXEL = 10.0


def choose_stride(shape, available_mb=None, *, budget_frac=0.5,
                  bytes_per_voxel=BYTES_PER_ANALYSIS_VOXEL) -> int:
    """Smallest voxel stride so the void analysis fits in RAM.

    ``stride == 1`` analyses at full resolution; ``stride == n`` reads
    every n-th voxel along each axis (n³-fold memory reduction) and the
    caller scales the voxel spacing accordingly so all physical sizes
    stay correct.  Falls back to a conservative 4 GiB budget when psutil
    is unavailable.
    """
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if available_mb is None:
        available_mb = 4096.0
    budget = max(256.0, float(available_mb) * budget_frac) * 1024 ** 2
    stride = 1
    while total / stride**3 * bytes_per_voxel > budget:
        stride += 1
    return stride


def build_void_mask(volume, thresh, stride: int = 1, progress=None):
    """Stream a boolean void mask one Z-slice at a time.

    ``volume`` only needs ``shape`` and 2-D ``[:, :, z]`` slicing — an
    ndarray and a :class:`~niftitool.core.io.LazyGrayVolume` both work,
    so the full-resolution float32 volume is never materialised.
    """
    sx, sy, sz = volume.shape[:3]
    zids = range(0, sz, stride)
    mask = np.empty(
        (len(range(0, sx, stride)), len(range(0, sy, stride)), len(zids)),
        dtype=bool,
    )
    n = len(zids)
    for i, z in enumerate(zids):
        sl = np.asarray(volume[:, :, z])
        mask[:, :, i] = sl[::stride, ::stride] < thresh
        if progress is not None and (i % 25 == 0 or i == n - 1):
            progress(i + 1, n)
    return mask


@dataclass
class VoidRecord:
    """Measurements for one connected void."""
    label:        int
    voxels:       int
    volume_mm3:   float
    eq_diam_mm:   float          # diameter of the equal-volume sphere
    sphericity:   float          # π^⅓ (6V)^⅔ / A, 1.0 = perfect sphere (NaN if not computed)
    elongation:   float          # major/minor principal axis ratio (1.0 = isotropic)
    centroid_vox: tuple          # (x, y, z) in voxel indices
    centroid_mm:  tuple          # (x, y, z) in mm
    bbox:         tuple          # (x0, x1, y0, y1, z0, z1) voxel slab


@dataclass
class PorosityResult:
    """Population-level summary + per-void records."""
    n_voids:            int
    total_void_voxels:  int
    total_volume_mm3:   float    # specimen volume analysed
    void_volume_mm3:    float
    porosity_pct:       float    # closed porosity (border-touching removed if excluded)
    border_void_voxels: int      # voxels discarded as exterior air
    spacing_mm:         tuple
    connectivity:       int
    min_voxels:         int
    exclude_border:     bool
    stride:             int = 1   # >1 = analysed on a downsampled grid
    raw_void_voxels:    int = 0   # voxels below threshold BEFORE filtering
    analysed_voxels:    int = 0   # total voxels in the analysed grid
    voids:              list = field(default_factory=list)   # [VoidRecord], largest first
    # population stats over eq_diam_mm (NaN when n_voids == 0)
    d_mean_mm:   float = float("nan")
    d_median_mm: float = float("nan")
    d_p10_mm:    float = float("nan")
    d_p90_mm:    float = float("nan")
    d_max_mm:    float = float("nan")
    mean_nn_dist_mm: float = float("nan")   # mean centroid nearest-neighbour distance

    def diameters_mm(self):
        return np.array([v.eq_diam_mm for v in self.voids], dtype=np.float64)

    def volumes_mm3(self):
        return np.array([v.volume_mm3 for v in self.voids], dtype=np.float64)


def _mesh_surface_area(mask, spacing):
    """Triangulated surface area of a binary blob via marching cubes."""
    if not HAS_SKIMAGE:
        return float("nan")
    padded = np.pad(mask.astype(np.float32), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    except Exception:
        return float("nan")
    tri = verts[faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def _elongation(mask, spacing):
    """Major/minor axis ratio from the inertia-tensor eigenvalues."""
    idx = np.argwhere(mask).astype(np.float64)
    if idx.shape[0] < 4:
        return 1.0
    idx *= np.asarray(spacing, dtype=np.float64)
    idx -= idx.mean(axis=0)
    cov = np.cov(idx.T)
    try:
        evals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 1.0
    evals = np.clip(evals, 1e-12, None)
    return float(np.sqrt(evals[-1] / evals[0]))


def analyze_voids(
    void_mask,
    spacing,
    *,
    min_voxels: int = 8,
    connectivity: int = 1,
    exclude_border: bool = True,
    max_detailed: int = DEFAULT_MAX_DETAILED_VOIDS,
    sphericity_min_voxels: int = DEFAULT_SPHERICITY_MIN_VOXELS,
    compute_sphericity: bool = True,
    progress=None,
) -> PorosityResult:
    """Run the full per-void analysis.

    Parameters
    ----------
    void_mask
        Boolean volume, ``True`` where the voxel is void/air.  Build it
        with ``labels == 0`` from :mod:`.segmentation` or ``hu < thresh``.
    spacing
        Voxel size ``(sx, sy, sz)`` in mm.
    min_voxels
        Ignore components smaller than this (scanner noise).
    connectivity
        1 = faces only (6-connected), 2 = +edges (18), 3 = +corners (26).
    exclude_border
        Drop components touching the volume boundary (exterior air).
    max_detailed
        Cap on per-void records (largest first); everything still counts
        toward porosity totals.
    progress
        Optional ``fn(done, total)`` callback for UI progress.
    """
    void_mask = np.asarray(void_mask, dtype=bool)
    spacing = tuple(float(s) for s in spacing[:3])
    vox_mm3 = spacing[0] * spacing[1] * spacing[2]
    total_volume_mm3 = void_mask.size * vox_mm3
    raw_void_voxels = int(void_mask.sum())

    structure = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(void_mask, structure=structure)

    border_void_voxels = 0
    if exclude_border and n:
        border_labels = np.unique(np.concatenate([
            lab[0, :, :].ravel(),  lab[-1, :, :].ravel(),
            lab[:, 0, :].ravel(),  lab[:, -1, :].ravel(),
            lab[:, :, 0].ravel(),  lab[:, :, -1].ravel(),
        ]))
        border_labels = border_labels[border_labels != 0]
        if border_labels.size:
            border_mask = np.isin(lab, border_labels)
            border_void_voxels = int(border_mask.sum())
            lab[border_mask] = 0

    counts = np.bincount(lab.ravel())
    if counts.size:
        counts[0] = 0                      # background
    keep = np.flatnonzero(counts >= max(1, int(min_voxels)))
    total_void_voxels = int(counts[keep].sum())

    # Largest first; cap detailed measurement.
    order = keep[np.argsort(counts[keep])[::-1]]
    detailed = order[: max_detailed]

    slabs = ndimage.find_objects(lab)
    records: list[VoidRecord] = []
    sp = np.asarray(spacing)
    n_det = len(detailed)

    for i, lbl in enumerate(detailed):
        sl = slabs[lbl - 1]
        if sl is None:
            continue
        sub = lab[sl] == lbl
        cnt = int(counts[lbl])
        vol = cnt * vox_mm3
        eq_d = float((6.0 * vol / np.pi) ** (1.0 / 3.0))

        # centroid (voxel + mm)
        loc = np.argwhere(sub).mean(axis=0)
        cen_vox = tuple(float(loc[k] + sl[k].start) for k in range(3))
        cen_mm = tuple(cen_vox[k] * spacing[k] for k in range(3))

        sph = float("nan")
        if compute_sphericity and cnt >= sphericity_min_voxels:
            area = _mesh_surface_area(sub, spacing)
            if area and area > 0 and not np.isnan(area):
                sph = float(np.pi ** (1 / 3) * (6.0 * vol) ** (2 / 3) / area)

        records.append(VoidRecord(
            label=int(lbl), voxels=cnt, volume_mm3=vol, eq_diam_mm=eq_d,
            sphericity=sph, elongation=_elongation(sub, sp),
            centroid_vox=cen_vox, centroid_mm=cen_mm,
            bbox=(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop,
                  sl[2].start, sl[2].stop),
        ))
        if progress is not None and (i % 200 == 0 or i == n_det - 1):
            progress(i + 1, n_det)

    void_volume_mm3 = total_void_voxels * vox_mm3
    result = PorosityResult(
        n_voids=len(keep),
        total_void_voxels=total_void_voxels,
        total_volume_mm3=total_volume_mm3,
        void_volume_mm3=void_volume_mm3,
        porosity_pct=100.0 * total_void_voxels / void_mask.size,
        border_void_voxels=border_void_voxels,
        spacing_mm=spacing,
        connectivity=connectivity,
        min_voxels=min_voxels,
        exclude_border=exclude_border,
        raw_void_voxels=raw_void_voxels,
        analysed_voxels=int(void_mask.size),
        voids=records,
    )

    if records:
        d = result.diameters_mm()
        result.d_mean_mm   = float(d.mean())
        result.d_median_mm = float(np.median(d))
        result.d_p10_mm    = float(np.percentile(d, 10))
        result.d_p90_mm    = float(np.percentile(d, 90))
        result.d_max_mm    = float(d.max())
        if len(records) >= 2:
            try:
                from scipy.spatial import cKDTree
                pts = np.array([v.centroid_mm for v in records])
                dist, _ = cKDTree(pts).query(pts, k=2)
                result.mean_nn_dist_mm = float(dist[:, 1].mean())
            except Exception:
                pass

    return result


def voids_to_csv(result: PorosityResult, path) -> None:
    """Write one row per void, largest first."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "voxels", "volume_mm3", "eq_diam_mm", "sphericity",
            "elongation", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm",
            "centroid_x_vox", "centroid_y_vox", "centroid_z_vox",
            "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1", "bbox_z0", "bbox_z1",
        ])
        for v in result.voids:
            w.writerow([
                v.label, v.voxels, f"{v.volume_mm3:.6g}", f"{v.eq_diam_mm:.6g}",
                "" if np.isnan(v.sphericity) else f"{v.sphericity:.4f}",
                f"{v.elongation:.4f}",
                *(f"{c:.4f}" for c in v.centroid_mm),
                *(f"{c:.2f}" for c in v.centroid_vox),
                *v.bbox,
            ])


def summary_lines(result: PorosityResult) -> list[str]:
    """Human-readable summary block (used by the log pane and the report)."""
    r = result
    lines = [
        f"Voids detected      : {r.n_voids}  (≥ {r.min_voxels} vox, "
        f"conn={r.connectivity})",
    ]
    if r.stride > 1:
        lines.append(
            f"Downsampled ×{r.stride}      : analysed every {r.stride}. voxel "
            f"to fit in RAM — voids smaller than "
            f"~{2 * max(r.spacing_mm):.3f} mm are not resolved"
        )
    raw_pct = 100.0 * r.raw_void_voxels / max(1, r.analysed_voxels)
    lines += [
        f"Raw void fraction   : {raw_pct:.3f} %  "
        f"(all voxels below threshold, before filters)",
        f"Closed porosity     : {r.porosity_pct:.3f} %  "
        f"(voids ≥ {r.min_voxels} vox, border excluded)",
        f"Void volume         : {r.void_volume_mm3:.3f} mm³ "
        f"of {r.total_volume_mm3:.1f} mm³ analysed",
    ]
    if r.exclude_border:
        lines.append(
            f"Border air removed  : {r.border_void_voxels} voxels "
            f"(exterior, not counted)"
        )
    if r.voids:
        lines += [
            f"Eq. diameter        : mean {r.d_mean_mm:.4f}  "
            f"median {r.d_median_mm:.4f}  max {r.d_max_mm:.4f} mm",
            f"Diameter p10–p90    : {r.d_p10_mm:.4f} – {r.d_p90_mm:.4f} mm",
        ]
        if not np.isnan(r.mean_nn_dist_mm):
            lines.append(
                f"Mean NN spacing     : {r.mean_nn_dist_mm:.4f} mm "
                f"(centroid nearest-neighbour)"
            )
    return lines
