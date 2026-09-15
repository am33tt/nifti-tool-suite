"""Per void analytics on a thresholded volume.

Connected component analysis of the void phase: label each pocket,
measure it (volume, equivalent diameter, sphericity, elongation,
centroid, bounding box) and summarise the population.

Components touching the volume boundary are dropped by default, since
the exterior air would otherwise register as one giant void. Sphericity
needs a surface area from a marching cubes mesh and is computed only
above sphericity_min_voxels, below which the mesh is too coarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..deps import np, ndimage, marching_cubes, HAS_SKIMAGE
from . import accel


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
    """Smallest voxel stride for which the analysis fits in RAM.

    stride n reads every n-th voxel along each axis, so the caller must
    scale the voxel spacing by n to keep physical sizes correct. Falls back
    to a 4 GiB budget when psutil is missing.
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
    """Stream a boolean void mask one Z slice at a time.

    volume needs only shape and 2-D [:, :, z] slicing, so an ndarray and a
    LazyGrayVolume both work and the float32 volume is never materialised.
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


def specimen_from_void_mask(void_mask, erode: int = 0):
    """Specimen mask derived from an existing void mask.

    void_mask is intensity < threshold, so its complement is the solid
    phase. The specimen is the largest connected solid component with its
    holes filled, which puts the pores back inside it.

    Porosity measured against the whole array counts surrounding air in the
    denominator, so the same specimen in a wider field of view would report
    a lower figure. Measuring against the specimen removes that dependence.

    erode drops that many voxels from the surface. The outermost layer is a
    partial volume mixture and would otherwise count as a shell of pores.
    """
    solid = ~np.asarray(void_mask, dtype=bool)
    if not solid.any():
        return np.zeros(solid.shape, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    lab, n = accel.label(solid, structure=structure)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        solid = lab == int(counts.argmax())
    del lab
    spec = accel.binary_fill_holes(solid)
    if erode:
        eroded = accel.binary_erosion(spec, iterations=int(erode))
        if eroded.any():
            spec = eroded
    return np.asarray(spec, dtype=bool)


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
    porosity_pct:       float    # closed porosity, referred to the specimen when known
    border_void_voxels: int      # voxels discarded as exterior air
    spacing_mm:         tuple
    connectivity:       int
    min_voxels:         int
    exclude_border:     bool
    stride:             int = 1   # >1 = analysed on a downsampled grid
    # Denominator the porosity is referred to. 0 means the whole analysed
    # array, which over-counts whenever there is air around the specimen.
    specimen_voxels:    int = 0
    volume_porosity_pct: float = float("nan")  # same voids, whole-array basis
    raw_void_voxels:    int = 0   # voxels below threshold BEFORE filtering
    analysed_voxels:    int = 0   # total voxels in the analysed grid
    # Sizes of every component surviving the border filter, before the
    # minimum size filter, so the count can be re-derived without
    # relabelling the volume.
    component_sizes:    object = None
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
    specimen_voxels: int | None = None,
    max_detailed: int = DEFAULT_MAX_DETAILED_VOIDS,
    sphericity_min_voxels: int = DEFAULT_SPHERICITY_MIN_VOXELS,
    compute_sphericity: bool = True,
    progress=None,
) -> PorosityResult:
    """Run the full per void analysis.

    void_mask is True where the voxel is void, spacing is the voxel size in
    mm. connectivity is 1 for faces, 2 for edges, 3 for corners, and
    components below min_voxels or touching the border are dropped.

    specimen_voxels is the denominator the porosity is referred to. Pass it
    whenever there is air around the specimen, from specimen_from_void_mask;
    without it the air inflates the denominator. max_detailed caps the per
    void records, largest first, but everything counts towards the totals.
    """
    void_mask = np.asarray(void_mask, dtype=bool)
    spacing = tuple(float(s) for s in spacing[:3])
    vox_mm3 = spacing[0] * spacing[1] * spacing[2]
    total_volume_mm3 = void_mask.size * vox_mm3
    raw_void_voxels = int(void_mask.sum())

    structure = ndimage.generate_binary_structure(3, connectivity)
    lab, n = accel.label(void_mask, structure=structure)

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
    all_sizes = counts[counts > 0].astype(np.int64)
    keep = np.flatnonzero(counts >= max(1, int(min_voxels)))
    total_void_voxels = int(counts[keep].sum())

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
    volume_porosity = 100.0 * total_void_voxels / max(1, void_mask.size)
    if specimen_voxels:
        porosity = 100.0 * total_void_voxels / max(1, int(specimen_voxels))
        basis_mm3 = int(specimen_voxels) * vox_mm3
    else:
        porosity = volume_porosity
        basis_mm3 = total_volume_mm3
    result = PorosityResult(
        n_voids=len(keep),
        total_void_voxels=total_void_voxels,
        total_volume_mm3=basis_mm3,
        void_volume_mm3=void_volume_mm3,
        porosity_pct=porosity,
        volume_porosity_pct=volume_porosity,
        specimen_voxels=int(specimen_voxels or 0),
        border_void_voxels=border_void_voxels,
        spacing_mm=spacing,
        connectivity=connectivity,
        min_voxels=min_voxels,
        exclude_border=exclude_border,
        raw_void_voxels=raw_void_voxels,
        analysed_voxels=int(void_mask.size),
        component_sizes=all_sizes,
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
            f"to fit in RAM; voids smaller than "
            f"~{2 * max(r.spacing_mm):.3f} mm are not resolved"
        )
    raw_pct = 100.0 * r.raw_void_voxels / max(1, r.analysed_voxels)
    basis = "specimen" if r.specimen_voxels else "whole analysed array"
    lines += [
        f"Raw void fraction   : {raw_pct:.3f} %  "
        f"(all voxels below threshold, before filters)",
        f"Closed porosity     : {r.porosity_pct:.3f} % of the {basis}  "
        f"(voids ≥ {r.min_voxels} vox, border excluded)",
        f"Void volume         : {r.void_volume_mm3:.4f} mm³ "
        f"of {r.total_volume_mm3:.2f} mm³ of {basis}",
    ]
    if r.specimen_voxels:
        lines.append(
            f"Specimen analysed   : {r.specimen_voxels:,} voxels "
            f"({100.0 * r.specimen_voxels / max(1, r.analysed_voxels):.1f} % "
            f"of the array; the surrounding air is excluded from the "
            f"denominator, so the porosity does not depend on how much "
            f"air happens to be in the field of view)"
        )
    else:
        lines.append(
            "Basis warning       : porosity is referred to the whole "
            "array because no specimen mask was supplied; any air "
            "around the specimen inflates the denominator and "
            "understates the porosity."
        )
    if r.exclude_border:
        lines.append(
            f"Border air removed  : {r.border_void_voxels} voxels "
            f"(exterior, not counted)"
        )
    if r.component_sizes is not None and len(r.component_sizes):
        sizes = np.asarray(r.component_sizes)
        sweep = {m: int((sizes >= m).sum()) for m in (1, 2, 4, 8, 16, 32)}
        pairs = "  ".join(f"\u2265{m}: {n}" for m, n in sweep.items())
        lines.append(f"Count vs min size   : {pairs}")
        single = 100.0 * float((sizes < 2).sum()) / max(1, sizes.size)
        lines.append(
            f"Single-voxel objects: {single:.1f} % of all dark objects \u2014 "
            f"a lone dark voxel is noise or partial volume, not a pore, so "
            f"the count is only meaningful together with the minimum size "
            f"it was measured at."
        )
    if r.voids:
        lines += [
            f"Eq. diameter        : mean {r.d_mean_mm:.4f}  "
            f"median {r.d_median_mm:.4f}  max {r.d_max_mm:.4f} mm",
            f"Diameter p10-p90    : {r.d_p10_mm:.4f} - {r.d_p90_mm:.4f} mm",
        ]
        if not np.isnan(r.mean_nn_dist_mm):
            lines.append(
                f"Mean NN spacing     : {r.mean_nn_dist_mm:.4f} mm "
                f"(centroid nearest-neighbour)"
            )
    return lines
