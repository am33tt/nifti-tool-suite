"""Boundary surface STL export from a NIfTI material mask.

The NIfTI stays the specimen geometry; the full specimen is never
converted to STL. Only the selected exposed surface regions are written
out, as boundary condition markers for the simulation.

Vertices are written in the same physical frame as the NIfTI affine and
the JSON bounding box. The full affine is used rather than
origin + index * positive_spacing, since NIfTI axes can be flipped.
Axis aligned affines with flips and permutations are supported; a
rotated or sheared affine is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import scipy.ndimage as ndi
except Exception:
    ndi = None


VALID_DIRECTIONS = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}


@dataclass
class BCSurfaceExportResult:
    """Small summary returned after a boundary surface STL is exported."""

    output_path: Path
    num_faces: int
    num_triangles: int
    roi_index_min: tuple[int, int, int]
    roi_index_max: tuple[int, int, int]


@dataclass(frozen=True)
class AffineAxisMapping:
    """Mapping between physical axes and voxel index axes.

    physical_axis_to_index_axis gives the voxel axis controlling each of
    X, Y, Z. physical_axis_sign is +1 when increasing that voxel index
    increases the physical coordinate and -1 when it decreases it.
    """

    physical_axis_to_index_axis: tuple[int, int, int]
    physical_axis_sign: tuple[int, int, int]


def nifti_physical_bounds(img) -> tuple[np.ndarray, np.ndarray]:
    """Physical bounds from all eight voxel grid corners.

    Uses the full affine rather than assuming positive spacing, so axis
    flips and other orientations are handled.
    """
    shape = np.asarray(img.shape[:3], dtype=np.int64)
    affine = np.asarray(img.affine, dtype=np.float64)

    # Grid corners, not voxel centres: indices run 0..shape-1 but the
    # physical domain runs from corner 0 to corner shape.
    corners_ijk = np.array(
        [
            [0, 0, 0, 1],
            [shape[0], 0, 0, 1],
            [0, shape[1], 0, 1],
            [0, 0, shape[2], 1],
            [shape[0], shape[1], 0, 1],
            [shape[0], 0, shape[2], 1],
            [0, shape[1], shape[2], 1],
            [shape[0], shape[1], shape[2], 1],
        ],
        dtype=np.float64,
    )

    corners_xyz = (affine @ corners_ijk.T).T[:, :3]

    bounds_min = corners_xyz.min(axis=0)
    bounds_max = corners_xyz.max(axis=0)

    return bounds_min, bounds_max


def _axis_mapping_from_affine(img, tolerance: float = 1e-6) -> AffineAxisMapping:
    """Build the physical to voxel axis mapping from the affine.

    Axes may be flipped or permuted; if increasing voxel index i decreases
    physical X then physical X has sign -1. A rotated or sheared affine is
    rejected, since face selection by '+X' or '-Y' is only defined for an
    axis aligned volume.
    """
    affine = np.asarray(img.affine, dtype=np.float64)
    matrix = affine[:3, :3]

    column_norms = np.linalg.norm(matrix, axis=0)
    if np.any(column_norms <= 0):
        raise ValueError("Invalid NIfTI affine: one or more voxel axes have zero length.")

    # Each affine column tells how one voxel-index axis moves in physical space.
    unit_columns = matrix / column_norms
    abs_unit_columns = np.abs(unit_columns)

    physical_axis_to_index_axis: list[int] = []
    physical_axis_sign: list[int] = []
    used_index_axes: set[int] = set()

    for physical_axis in range(3):
        index_axis = int(np.argmax(abs_unit_columns[physical_axis, :]))
        dominance = float(abs_unit_columns[physical_axis, index_axis])

        if dominance < 1.0 - tolerance:
            raise ValueError(
                "Unsupported NIfTI affine for boundary STL export: "
                "the affine appears to contain rotation/shear. "
                "Please reorient/resample to an axis-aligned image first."
            )

        if index_axis in used_index_axes:
            raise ValueError(
                "Unsupported NIfTI affine for boundary STL export: "
                "could not build a unique physical-axis to voxel-axis mapping."
            )

        used_index_axes.add(index_axis)

        sign_value = float(unit_columns[physical_axis, index_axis])
        sign = 1 if sign_value >= 0.0 else -1

        physical_axis_to_index_axis.append(index_axis)
        physical_axis_sign.append(sign)

    return AffineAxisMapping(
        physical_axis_to_index_axis=tuple(physical_axis_to_index_axis),
        physical_axis_sign=tuple(physical_axis_sign),
    )


def _ijk_to_xyz(affine: np.ndarray, ijk: Iterable[float]) -> np.ndarray:
    """Convert one voxel-grid coordinate to physical XYZ using the affine."""
    ijk_h = np.array(
        [float(ijk[0]), float(ijk[1]), float(ijk[2]), 1.0],
        dtype=np.float64,
    )
    return (affine @ ijk_h)[:3]


def _roi_mm_to_index_bounds(
    img,
    roi_bounds_mm: Optional[tuple[Iterable[float], Iterable[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert physical ROI bounds to half open voxel index bounds.

    The region is given in millimetres and mapped back through the inverse
    affine, which handles flipped axes that origin + spacing would not.
    """
    shape = np.asarray(img.shape[:3], dtype=np.int64)

    if roi_bounds_mm is None:
        return np.array([0, 0, 0], dtype=np.int64), shape.copy()

    affine = np.asarray(img.affine, dtype=np.float64)
    inv_affine = np.linalg.inv(affine)

    roi_min_mm = np.asarray(tuple(roi_bounds_mm[0]), dtype=np.float64)
    roi_max_mm = np.asarray(tuple(roi_bounds_mm[1]), dtype=np.float64)

    lo = np.minimum(roi_min_mm, roi_max_mm)
    hi = np.maximum(roi_min_mm, roi_max_mm)

    # Convert all eight physical ROI corners into voxel-index space.
    # Using all corners is important when axes are flipped.
    corners_xyz = np.array(
        [
            [lo[0], lo[1], lo[2], 1.0],
            [hi[0], lo[1], lo[2], 1.0],
            [lo[0], hi[1], lo[2], 1.0],
            [lo[0], lo[1], hi[2], 1.0],
            [hi[0], hi[1], lo[2], 1.0],
            [hi[0], lo[1], hi[2], 1.0],
            [lo[0], hi[1], hi[2], 1.0],
            [hi[0], hi[1], hi[2], 1.0],
        ],
        dtype=np.float64,
    )

    corners_ijk = (inv_affine @ corners_xyz.T).T[:, :3]

    index_min = np.floor(corners_ijk.min(axis=0)).astype(np.int64)
    index_max = np.ceil(corners_ijk.max(axis=0)).astype(np.int64)

    index_min = np.clip(index_min, 0, shape)
    index_max = np.clip(index_max, 0, shape)

    # Ensure each selected axis has at least one voxel.
    for axis in range(3):
        if index_max[axis] <= index_min[axis]:
            index_max[axis] = min(shape[axis], index_min[axis] + 1)

    return index_min, index_max


def _as_scalar_float_array(data: np.ndarray) -> np.ndarray:
    """Reduce NIfTI voxel data to a scalar float array for thresholding.

    Handles scalar volumes, structured RGB dtypes and channel last
    RGB/RGBA arrays.
    """
    arr = np.asarray(data)

    # Structured RGB dtype, for example [('R','u1'), ('G','u1'), ('B','u1')].
    if arr.dtype.fields is not None:
        names = arr.dtype.names or ()
        lower_names = {name.lower(): name for name in names}

        if all(ch in lower_names for ch in ("r", "g", "b")):
            r = arr[lower_names["r"]].astype(np.float32)
            g = arr[lower_names["g"]].astype(np.float32)
            b = arr[lower_names["b"]].astype(np.float32)

            # Standard luminance conversion.
            return 0.299 * r + 0.587 * g + 0.114 * b

        # Fallback for other structured dtypes: use the first field.
        if names:
            return arr[names[0]].astype(np.float32)

        raise ValueError("Unsupported structured NIfTI dtype.")

    # Channel-last RGB/RGBA array.
    if arr.ndim >= 4 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        return (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        )

    # Normal scalar array.
    return arr.astype(np.float32, copy=False)


def _make_material_mask(
    data: np.ndarray,
    lower_threshold: float,
    upper_threshold: Optional[float],
) -> np.ndarray:
    """Boolean material mask: True is material, False is void or air."""
    if upper_threshold is None:
        return np.asarray(data >= lower_threshold, dtype=bool)

    return np.asarray(
        (data >= lower_threshold) & (data <= upper_threshold),
        dtype=bool,
    )


def _clean_material_mask(
    mask: np.ndarray,
    enabled: bool = True,
    close_iterations: int = 0,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Clean thresholding artefacts in the material mask.

    close_iterations defaults to 0: binary closing moves the outer boundary
    inward by a voxel where the material touches the ROI edge, which puts
    the exported surface inside the specimen. The default keeps the largest
    component and leaves real pores alone.
    """
    if not enabled:
        return np.asarray(mask, dtype=bool)

    if ndi is None:
        return np.asarray(mask, dtype=bool)

    clean = np.asarray(mask, dtype=bool)

    # 6-neighbour connectivity is conservative and avoids diagonal-only links.
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)

    if keep_largest_component:
        labels, num = ndi.label(clean, structure=structure)
        if num > 1:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            largest_label = int(np.argmax(counts))
            clean = labels == largest_label

    if close_iterations > 0:
        clean = ndi.binary_closing(
            clean,
            structure=structure,
            iterations=int(close_iterations),
        )

    if keep_largest_component:
        labels, num = ndi.label(clean, structure=structure)
        if num > 1:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            largest_label = int(np.argmax(counts))
            clean = labels == largest_label

    return np.asarray(clean, dtype=bool)


def _axis_for_direction(direction: str) -> int:
    """Return physical axis index for a direction string.

    X -> 0, Y -> 1, Z -> 2.
    """
    if direction in {"+X", "-X"}:
        return 0
    if direction in {"+Y", "-Y"}:
        return 1
    if direction in {"+Z", "-Z"}:
        return 2
    raise ValueError(f"Invalid direction: {direction}")


def _normal_for_direction(direction: str) -> tuple[float, float, float]:
    """Return the requested physical outward normal direction."""
    normals = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0),
        "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    return normals[direction]


def _neighbor_offset(
    direction: str,
    axis_mapping: AffineAxisMapping,
) -> tuple[int, int, int]:
    """Voxel index neighbour offset for a physical face direction.

    Physical '+X' may correspond to an increasing or a decreasing voxel
    index depending on the affine.
    """
    physical_axis = _axis_for_direction(direction)
    physical_sign = 1 if direction.startswith("+") else -1

    index_axis = axis_mapping.physical_axis_to_index_axis[physical_axis]
    index_sign = axis_mapping.physical_axis_sign[physical_axis]

    offset = np.array([0, 0, 0], dtype=np.int64)
    offset[index_axis] = physical_sign * index_sign

    return tuple(int(v) for v in offset)


def _quad_normal(verts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Compute a unit normal from the first three vertices of a quad."""
    v0, v1, v2, _ = verts
    normal = np.cross(v1 - v0, v2 - v0)
    norm = float(np.linalg.norm(normal))

    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    return normal / norm


def _orient_quad_towards_direction(
    verts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    direction: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[float, float, float]]:
    """Orient quad vertices so the triangle normal points outward.

    ASCII STL stores a facet normal, but many readers recompute it from the
    vertex order, so both are kept consistent.
    """
    desired_normal = np.asarray(_normal_for_direction(direction), dtype=np.float64)
    normal = _quad_normal(verts)

    if float(np.linalg.norm(normal)) <= 0.0:
        return verts, tuple(float(v) for v in desired_normal)

    if float(np.dot(normal, desired_normal)) < 0.0:
        v0, v1, v2, v3 = verts
        verts = (v0, v3, v2, v1)
        normal = _quad_normal(verts)

    return verts, tuple(float(v) for v in normal)


def _voxel_face_vertices(
    global_index: np.ndarray,
    affine: np.ndarray,
    axis_mapping: AffineAxisMapping,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four physical vertices of one exposed voxel face.

    Finds the voxel index face matching the requested physical direction,
    builds its four grid corners and maps them through the affine. This is
    what keeps the STL coordinates consistent with the JSON bounding box.
    """
    base = np.asarray(global_index, dtype=np.int64)

    physical_axis = _axis_for_direction(direction)
    physical_sign = 1 if direction.startswith("+") else -1

    index_axis = axis_mapping.physical_axis_to_index_axis[physical_axis]
    index_sign = axis_mapping.physical_axis_sign[physical_axis]

    # If moving outward in this physical direction means increasing the voxel
    # index, the face is at index + 1. Otherwise, it is at index.
    index_step = physical_sign * index_sign
    use_high_index_face = index_step > 0

    face_index_value = int(base[index_axis] + (1 if use_high_index_face else 0))

    other_axes = [axis for axis in range(3) if axis != index_axis]
    a0, a1 = other_axes

    c0 = base.copy()
    c1 = base.copy()
    c2 = base.copy()
    c3 = base.copy()

    c0[index_axis] = face_index_value
    c1[index_axis] = face_index_value
    c2[index_axis] = face_index_value
    c3[index_axis] = face_index_value

    c0[a0] = base[a0]
    c0[a1] = base[a1]

    c1[a0] = base[a0] + 1
    c1[a1] = base[a1]

    c2[a0] = base[a0] + 1
    c2[a1] = base[a1] + 1

    c3[a0] = base[a0]
    c3[a1] = base[a1] + 1

    return tuple(
        np.asarray(_ijk_to_xyz(affine, corner), dtype=np.float64)
        for corner in (c0, c1, c2, c3)
    )


def _write_triangle(f, normal, tri):
    """Write one ASCII STL triangle."""
    f.write(
        f"  facet normal "
        f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
    )
    f.write("    outer loop\n")

    for v in tri:
        f.write(
            f"      vertex "
            f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
        )

    f.write("    endloop\n")
    f.write("  endfacet\n")


def _write_quad_as_stl_triangles(f, normal, verts):
    """Write one quad as two STL triangles."""
    v0, v1, v2, v3 = verts
    _write_triangle(f, normal, (v0, v1, v2))
    _write_triangle(f, normal, (v0, v2, v3))


def _filter_by_surface_band(
    quads: list[dict],
    surface_band_mm: Optional[float],
) -> list[dict]:
    """Keep only faces within surface_band_mm of the outermost one.

    The region may expose several faces, around pores and side edges as
    well as the surface. For '+Y' the highest Y is kept, for '-Y' the
    lowest.
    """
    if surface_band_mm is None:
        return quads

    if surface_band_mm <= 0:
        return quads

    filtered = []

    for direction in ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]:
        group = [q for q in quads if q["direction"] == direction]
        if not group:
            continue

        axis = _axis_for_direction(direction)
        coords = np.array([q["center"][axis] for q in group], dtype=np.float64)

        if direction.startswith("+"):
            extreme = float(coords.max())
            keep_limit = extreme - surface_band_mm
            filtered.extend([q for q in group if q["center"][axis] >= keep_limit])
        else:
            extreme = float(coords.min())
            keep_limit = extreme + surface_band_mm
            filtered.extend([q for q in group if q["center"][axis] <= keep_limit])

    return filtered


def _filter_largest_connected_component(quads: list[dict]) -> list[dict]:
    """Keep the largest group of quads connected through shared vertices.

    Removes speckles left by threshold noise, after face extraction.
    """
    if len(quads) <= 1:
        return quads

    vertex_to_quads: dict[tuple[float, float, float], list[int]] = {}

    for quad_idx, quad in enumerate(quads):
        for v in quad["verts"]:
            key = tuple(np.round(v.astype(np.float64), 6))
            vertex_to_quads.setdefault(key, []).append(quad_idx)

    visited = set()
    largest_component = set()

    for start_idx in range(len(quads)):
        if start_idx in visited:
            continue

        component = set()
        stack = [start_idx]

        while stack:
            idx = stack.pop()

            if idx in visited:
                continue

            visited.add(idx)
            component.add(idx)

            for v in quads[idx]["verts"]:
                key = tuple(np.round(v.astype(np.float64), 6))
                for neighbor_idx in vertex_to_quads.get(key, []):
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)

        if len(component) > len(largest_component):
            largest_component = component

    return [q for idx, q in enumerate(quads) if idx in largest_component]


def export_nifti_boundary_region_as_stl(
    img,
    output_path: str | Path,
    lower_threshold: float,
    upper_threshold: Optional[float] = None,
    roi_bounds_mm: Optional[tuple[Iterable[float], Iterable[float]]] = None,
    face_directions: Optional[Iterable[str]] = None,
    solid_name: str = "bc_surface",
    pad_voxels: int = 1,
    max_faces: Optional[int] = None,
    surface_band_mm: Optional[float] = None,
    normal_offset_mm: float = 0.0,
    keep_largest_component: bool = False,
    clean_material_mask: bool = True,
    mask_cleanup_iterations: int = 0,
) -> BCSurfaceExportResult:
    """Export selected exposed boundary faces as an ASCII STL.

    The physical region is converted to voxel bounds, the padded region is
    read and thresholded, material voxels whose neighbour in the requested
    direction is background give the exposed faces, and those are written
    as triangles through the full affine. The surface band and largest
    component filters are optional, and normal_offset_mm finally shifts the
    surface along its normal, outward for positive values.
    """
    output_path = Path(output_path)

    if face_directions is None:
        directions = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    else:
        directions = [str(d).upper().strip() for d in face_directions]

    for direction in directions:
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid face direction '{direction}'. "
                f"Valid directions are: {sorted(VALID_DIRECTIONS)}"
            )

    normal_offset_mm = float(normal_offset_mm)
    if not np.isfinite(normal_offset_mm):
        raise ValueError("normal_offset_mm must be a finite number.")

    shape = np.asarray(img.shape[:3], dtype=np.int64)
    affine = np.asarray(img.affine, dtype=np.float64)
    axis_mapping = _axis_mapping_from_affine(img)

    roi_index_min, roi_index_max = _roi_mm_to_index_bounds(
        img=img,
        roi_bounds_mm=roi_bounds_mm,
    )

    # Padding allows neighbour checks just outside the region, so a face on
    # its edge is not called exposed merely because the neighbour was not
    # loaded.
    pad = max(0, int(pad_voxels))

    padded_min = np.maximum(roi_index_min - pad, 0)
    padded_max = np.minimum(roi_index_max + pad, shape)

    sx = slice(int(padded_min[0]), int(padded_max[0]))
    sy = slice(int(padded_min[1]), int(padded_max[1]))
    sz = slice(int(padded_min[2]), int(padded_max[2]))

    data_roi = _as_scalar_float_array(np.asarray(img.dataobj[sx, sy, sz]))
    mask = _make_material_mask(data_roi, lower_threshold, upper_threshold)

    mask_voxels_before_cleanup = int(np.count_nonzero(mask))

    mask = _clean_material_mask(
        mask,
        enabled=clean_material_mask,
        close_iterations=mask_cleanup_iterations,
        keep_largest_component=True,
    )

    mask_voxels_after_cleanup = int(np.count_nonzero(mask))

    # Neighbour checks use the padded mask, but exported faces come only
    # from voxels inside the selected region.
    rel_min = roi_index_min - padded_min
    rel_max = roi_index_max - padded_min

    inner_mask = np.zeros_like(mask, dtype=bool)
    inner_mask[
        int(rel_min[0]):int(rel_max[0]),
        int(rel_min[1]):int(rel_max[1]),
        int(rel_min[2]):int(rel_max[2]),
    ] = mask[
        int(rel_min[0]):int(rel_max[0]),
        int(rel_min[1]):int(rel_max[1]),
        int(rel_min[2]):int(rel_max[2]),
    ]

    material_voxels = np.argwhere(inner_mask)

    if material_voxels.size == 0:
        raise RuntimeError(
            "No material voxels found inside selected ROI. "
            "Try changing threshold values or ROI bounds."
        )

    quads: list[dict] = []
    mask_shape = np.asarray(mask.shape[:3], dtype=np.int64)

    for local_idx in material_voxels:
        global_idx = padded_min + local_idx

        for direction in directions:
            offset = np.asarray(
                _neighbor_offset(direction, axis_mapping),
                dtype=np.int64,
            )

            neighbor_local = local_idx + offset
            neighbor_global = global_idx + offset

            neighbor_inside_volume = np.all(
                (neighbor_global >= 0) & (neighbor_global < shape)
            )

            if not neighbor_inside_volume:
                exposed = True
            else:
                neighbor_inside_mask = np.all(
                    (neighbor_local >= 0) & (neighbor_local < mask_shape)
                )

                if neighbor_inside_mask:
                    exposed = not bool(mask[tuple(neighbor_local)])
                else:
                    # Rare with pad_voxels >= 1. Treat the face as exposed
                    # rather than dropping a possible boundary face.
                    exposed = True

            if not exposed:
                continue

            verts = _voxel_face_vertices(
                global_index=global_idx,
                affine=affine,
                axis_mapping=axis_mapping,
                direction=direction,
            )

            verts, normal = _orient_quad_towards_direction(
                verts=verts,
                direction=direction,
            )

            center = (verts[0] + verts[1] + verts[2] + verts[3]) / 4.0

            quads.append(
                {
                    "direction": direction,
                    "normal": normal,
                    "verts": verts,
                    "center": center,
                }
            )

    if not quads:
        raise RuntimeError(
            "No exposed boundary faces found in the selected ROI. "
            "Try changing face direction, ROI bounds, or threshold."
        )

    before_filter_count = len(quads)

    quads = _filter_by_surface_band(
        quads=quads,
        surface_band_mm=surface_band_mm,
    )

    if keep_largest_component:
        quads = _filter_largest_connected_component(quads)

    if not quads:
        raise RuntimeError(
            "Surface cleanup removed all boundary faces. "
            "Try increasing surface band or disabling largest-component cleanup."
        )

    # Offset applied after extraction and cleanup, so the region, band,
    # shape, normals and face count are unchanged. Positive moves outward.
    if normal_offset_mm != 0.0:
        for quad in quads:
            shift = (
                np.asarray(
                    _normal_for_direction(quad["direction"]),
                    dtype=np.float64,
                )
                * normal_offset_mm
            )
            quad["verts"] = tuple(
                np.asarray(vertex, dtype=np.float64) + shift
                for vertex in quad["verts"]
            )
            quad["center"] = (
                np.asarray(quad["center"], dtype=np.float64) + shift
            )

    if max_faces is not None and len(quads) > int(max_faces):
        raise RuntimeError(
            f"Too many boundary faces selected ({len(quads)}). "
            f"Maximum allowed is {max_faces}. "
            "Try selecting a smaller ROI."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"solid {solid_name}\n")

        for quad in quads:
            _write_quad_as_stl_triangles(
                f=f,
                normal=quad["normal"],
                verts=quad["verts"],
            )

        f.write(f"endsolid {solid_name}\n")

    print(f"Material voxels before mask cleanup: {mask_voxels_before_cleanup}")
    print(f"Material voxels after mask cleanup:  {mask_voxels_after_cleanup}")
    print(f"Boundary faces before surface cleanup: {before_filter_count}")
    print(f"Boundary faces after surface cleanup:  {len(quads)}")

    return BCSurfaceExportResult(
        output_path=output_path,
        num_faces=len(quads),
        num_triangles=2 * len(quads),
        roi_index_min=tuple(int(v) for v in roi_index_min),
        roi_index_max=tuple(int(v) for v in roi_index_max),
    )