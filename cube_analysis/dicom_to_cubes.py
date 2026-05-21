"""
dicom_to_cubes.py
=================

Stream a folder of micro-CT DICOM slices (one column of N stacked cubes),
auto-detect the in-plane rotation and the air gaps between cubes, and write
one rotated + cropped NIfTI per detected cube.

Pipeline (no full volume ever held in RAM):

  1. List + sort DICOMs by filename (which matches monotonic Z here).
  2. Pass 1: compute mean intensity per slice (with a stride) -> Z-profile.
  3. Detect cube Z-ranges from air gaps in the Z-profile.
  4. Sample one middle slice from the longest band -> auto-detect:
       - in-plane rotation angle (PCA on the material mask), and
       - XY bounding box of the cube after rotation.
  5. Pass 2: for each band, allocate a memmapped int16 volume,
     stream slices through (read -> rotate -> crop -> store), then save
     the volume as a gzipped NIfTI to the output directory.

The output NIfTIs have the same int16 dtype as the input pixel data and an
affine built from PixelSpacing + Z-spacing inferred from ImagePositionPatient.

Run:
    python cube_analysis/dicom_to_cubes.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
import pydicom
from scipy import ndimage


# --- Configuration ----------------------------------------------------------

DEFAULT_DICOM_DIR = Path(r"C:/Users/amyth/Desktop/multiscan_dcm")
DEFAULT_OUT_DIR = Path(r"D:/Result")

EXPECTED_N_CUBES = 3
Z_PROFILE_STRIDE = 4          # speed/precision tradeoff for the gap detection
XY_MARGIN = 20                # voxels added around the detected XY bbox
ROTATION_INTERP_ORDER = 1     # 1 = bilinear (fast, fine for int16 CT data)

# Threshold for the "material" mask used to find the rotation angle and bbox.
# Computed adaptively per probe slice: lo + MATERIAL_QUANTILE_MIX * (hi - lo)
# where (lo, hi) are the 5th and 95th intensity percentiles of the probe slice.
MATERIAL_QUANTILE_MIX = 0.5


# --- DICOM stack ------------------------------------------------------------


@dataclass
class DicomStack:
    files: List[Path]
    rows: int
    cols: int
    pixel_spacing_xy: float        # mm
    pixel_spacing_z: float         # mm
    rescale_slope: float
    rescale_intercept: float
    origin_xyz: Tuple[float, float, float]  # ImagePositionPatient of first file

    @classmethod
    def from_dir(cls, dicom_dir: Path) -> "DicomStack":
        files = sorted(dicom_dir.glob("*.dcm"))
        if not files:
            raise FileNotFoundError(f"No .dcm files in {dicom_dir}")
        ds0 = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        ds1 = pydicom.dcmread(str(files[1]), stop_before_pixels=True)
        rows = int(ds0.Rows)
        cols = int(ds0.Columns)
        ps_xy = float(ds0.PixelSpacing[0])
        z0 = float(ds0.ImagePositionPatient[2])
        z1 = float(ds1.ImagePositionPatient[2])
        ps_z = abs(z1 - z0)
        slope = float(getattr(ds0, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds0, "RescaleIntercept", 0.0) or 0.0)
        origin = tuple(float(x) for x in ds0.ImagePositionPatient)
        return cls(
            files=files,
            rows=rows,
            cols=cols,
            pixel_spacing_xy=ps_xy,
            pixel_spacing_z=ps_z,
            rescale_slope=slope,
            rescale_intercept=intercept,
            origin_xyz=origin,  # type: ignore[arg-type]
        )

    def __len__(self) -> int:
        return len(self.files)

    def read_pixels(self, k: int) -> np.ndarray:
        """Return the (rows, cols) pixel array for slice k, with RescaleSlope/
        Intercept applied. Cast back to int16 to keep the same dtype as the
        input (slope=1, intercept=0 here, so this is effectively a no-op)."""
        ds = pydicom.dcmread(str(self.files[k]))
        arr = ds.pixel_array
        if self.rescale_slope != 1.0 or self.rescale_intercept != 0.0:
            arr = arr.astype(np.float32) * self.rescale_slope + self.rescale_intercept
        return arr.astype(np.int16, copy=False)


# --- Z-profile + cube detection --------------------------------------------


def build_z_profile(stack: DicomStack, stride: int) -> np.ndarray:
    """Mean intensity per slice along Z. Strided pass, gaps interpolated."""
    nz = len(stack)
    profile = np.full(nz, np.nan, dtype=np.float64)
    t0 = time.time()
    for k in range(0, nz, stride):
        # Read pixels only (skip the bulky header decode work where possible).
        ds = pydicom.dcmread(str(stack.files[k]))
        profile[k] = float(ds.pixel_array.mean())
        if k % max(1, (200 // stride) * stride) == 0:
            elapsed = time.time() - t0
            print(
                f"  Z-profile: slice {k}/{nz}  "
                f"(elapsed {elapsed:.0f}s, mean={profile[k]:.1f})"
            )
    if stride > 1:
        idx = np.arange(nz)
        good = ~np.isnan(profile)
        profile = np.interp(idx, idx[good], profile[good])
    return profile


def detect_cubes_along_z(
    profile: np.ndarray, n_expected: int
) -> List[Tuple[int, int]]:
    """Detect cube Z-ranges as contiguous bands above an adaptive threshold."""
    p_low = float(np.percentile(profile, 5))
    p_high = float(np.percentile(profile, 95))
    threshold = p_low + 0.4 * (p_high - p_low)

    mask = profile > threshold
    mask = ndimage.binary_closing(mask, iterations=5)
    mask = ndimage.binary_opening(mask, iterations=3)

    labels, n_found = ndimage.label(mask)
    if n_found == 0:
        raise RuntimeError(
            "No cubes detected. Inspect the saved z_profile.npy/png."
        )

    bands = []
    for i in range(1, n_found + 1):
        zs = np.where(labels == i)[0]
        bands.append((int(zs.min()), int(zs.max()) + 1))

    bands.sort(key=lambda b: -(b[1] - b[0]))
    bands = bands[:n_expected]
    bands.sort(key=lambda b: b[0])

    print(
        f"  Detected {n_found} bands above threshold={threshold:.1f}; "
        f"keeping top {len(bands)} by thickness."
    )
    for i, (a, b) in enumerate(bands):
        print(f"    Cube {i + 1}: Z in [{a}, {b})  ({b - a} slices)")
    return bands


# --- Rotation + bbox detection ---------------------------------------------


def _largest_component_mask(img: np.ndarray, threshold: float) -> np.ndarray:
    mask = img > threshold
    if not mask.any():
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labels == keep


def detect_rotation_and_bbox(
    probe_img: np.ndarray,
    margin: int,
) -> Tuple[float, Tuple[int, int, int, int], float]:
    """Given a single probe slice, return (angle_deg, (x0, x1, y0, y1), thr).
    The angle is the rotation (degrees, ccw, around the slice centre) needed
    to bring the cube edges parallel to the image axes."""
    lo, hi = np.percentile(probe_img, [5, 95])
    thr = float(lo + MATERIAL_QUANTILE_MIX * (hi - lo))
    mask = _largest_component_mask(probe_img, thr)
    if not mask.any():
        raise RuntimeError(
            f"Probe slice has no above-threshold pixels (thr={thr:.1f})."
        )

    # PCA on mask point cloud -> orientation of the major axis (degrees).
    ys, xs = np.where(mask)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    _, v = np.linalg.eigh(cov)
    major = v[:, -1]
    angle_major = np.degrees(np.arctan2(major[1], major[0]))
    # For a square cube the orientation is ambiguous mod 90°; pick the smallest
    # rotation magnitude in (-45, 45].
    angle = ((angle_major + 45.0) % 90.0) - 45.0

    # Rotate and re-bbox.
    rot = ndimage.rotate(
        probe_img, angle, reshape=False, order=ROTATION_INTERP_ORDER,
        mode="constant", cval=0,
    )
    mask_rot = _largest_component_mask(rot, thr)
    if not mask_rot.any():
        raise RuntimeError(
            "After rotation the material mask is empty -- check threshold."
        )
    ys2, xs2 = np.where(mask_rot)
    h, w = probe_img.shape
    x0 = max(0, int(xs2.min()) - margin)
    x1 = min(w, int(xs2.max()) + 1 + margin)
    y0 = max(0, int(ys2.min()) - margin)
    y1 = min(h, int(ys2.max()) + 1 + margin)
    return float(angle), (x0, x1, y0, y1), thr


# --- Per-cube streaming write ----------------------------------------------


def _build_affine(stack: DicomStack, x0: int, y0: int, z_start: int) -> np.ndarray:
    """Diagonal affine in mm. Rotation is baked into the pixel grid itself
    (so the affine stays axis-aligned), and the origin is translated to the
    crop corner in the *unrotated* DICOM frame -- this is approximate (the
    crop is in rotated pixels) but consistent with how analyze_cubes.py
    treats its inputs."""
    sx = sy = stack.pixel_spacing_xy
    sz = stack.pixel_spacing_z
    ox, oy, oz = stack.origin_xyz
    affine = np.array([
        [sx, 0,  0,  ox + x0 * sx],
        [0,  sy, 0,  oy + y0 * sy],
        [0,  0,  sz, oz + z_start * sz],
        [0,  0,  0,  1.0],
    ], dtype=np.float64)
    return affine


def process_cube(
    stack: DicomStack,
    index: int,
    z_start: int,
    z_end: int,
    angle: float,
    bbox: Tuple[int, int, int, int],
    out_dir: Path,
    tmp_dir: Path,
) -> dict:
    """Stream one cube's slices through rotate+crop and save as NIfTI."""
    x0, x1, y0, y1 = bbox
    nx_c = x1 - x0
    ny_c = y1 - y0
    nz_c = z_end - z_start

    out_nii = out_dir / f"cube_{index:02d}.nii.gz"
    gb = nx_c * ny_c * nz_c * 2 / 1e9
    print(
        f"  Cube {index}: bbox X[{x0}:{x1}] Y[{y0}:{y1}] Z[{z_start}:{z_end}]"
        f"  -> shape ({nx_c}, {ny_c}, {nz_c})  (~{gb:.2f} GB int16)"
    )

    # Use a memmap so we don't have to hold the whole cube in RAM. Nibabel
    # will iterate over it once on save() to gzip the bytes out.
    tmp_path = tmp_dir / f"cube_{index:02d}.raw"
    if tmp_path.exists():
        tmp_path.unlink()
    cropped = np.memmap(
        str(tmp_path), dtype=np.int16, mode="w+",
        shape=(nx_c, ny_c, nz_c),
    )

    t0 = time.time()
    for k, z in enumerate(range(z_start, z_end)):
        img = stack.read_pixels(z)
        # ndimage.rotate uses (row, col) -> rotation in the image plane.
        rot = ndimage.rotate(
            img, angle, reshape=False, order=ROTATION_INTERP_ORDER,
            mode="constant", cval=0,
        )
        # img is indexed (rows=Y, cols=X). We store as (X, Y, Z) to match the
        # convention used by analyze_cubes.py (nibabel-style).
        slab = rot[y0:y1, x0:x1]              # shape (ny_c, nx_c)
        cropped[:, :, k] = slab.T              # store as (nx_c, ny_c)
        if k % 50 == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / max(elapsed, 1e-6)
            eta = (nz_c - k - 1) / max(rate, 1e-6)
            print(
                f"    cube {index} slice {k}/{nz_c}  "
                f"({elapsed:.0f}s elapsed, {rate:.1f} sl/s, eta {eta:.0f}s)"
            )

    cropped.flush()
    print(f"  Saving cropped cube to {out_nii} (gzip, this can take minutes)...")
    affine = _build_affine(stack, x0, y0, z_start)
    nib.Nifti1Image(np.asarray(cropped), affine).to_filename(str(out_nii))

    # Drop the memmap and delete its temp backing file.
    del cropped
    try:
        tmp_path.unlink()
    except OSError:
        pass

    return {
        "cube_index": index,
        "z_start": z_start,
        "z_end": z_end,
        "x_min": x0, "x_max": x1,
        "y_min": y0, "y_max": y1,
        "rotation_deg": angle,
        "output_nii": str(out_nii),
        "shape": [nx_c, ny_c, nz_c],
    }


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-cubes", type=int, default=EXPECTED_N_CUBES)
    parser.add_argument("--z-stride", type=int, default=Z_PROFILE_STRIDE)
    parser.add_argument("--xy-margin", type=int, default=XY_MARGIN)
    parser.add_argument(
        "--angle", type=float, default=None,
        help="Override auto-detected rotation angle (degrees, ccw).",
    )
    parser.add_argument(
        "--tmp-dir", type=Path, default=None,
        help="Directory for per-cube memmap scratch files. "
             "Defaults to the system temp dir.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.tmp_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing DICOMs in {args.dicom_dir}...")
    stack = DicomStack.from_dir(args.dicom_dir)
    print(
        f"  {len(stack)} slices, each {stack.rows}x{stack.cols} "
        f"@ {stack.pixel_spacing_xy:.4f} mm/px (Z spacing {stack.pixel_spacing_z:.4f} mm)"
    )
    print(f"  RescaleSlope/Intercept = {stack.rescale_slope}/{stack.rescale_intercept}")

    print(f"\nBuilding Z-profile (stride {args.z_stride})...")
    profile = build_z_profile(stack, stride=args.z_stride)
    np.save(args.out / "z_profile.npy", profile)

    print("\nDetecting cubes along Z...")
    bands = detect_cubes_along_z(profile, args.n_cubes)

    # Diagnostic plot of the Z-profile + detected bands.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(profile, lw=0.8, color="black")
        for i, (a, b) in enumerate(bands):
            ax.axvspan(a, b, color=f"C{i}", alpha=0.25, label=f"cube {i+1}")
        ax.set_xlabel("Slice index"); ax.set_ylabel("Mean intensity")
        ax.set_title("Z-profile and detected cubes")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(args.out / "z_profile.png", dpi=140)
        plt.close(fig)
    except ImportError:
        pass

    print("\nProbing a middle slice for rotation + bbox...")
    biggest_band = max(bands, key=lambda b: b[1] - b[0])
    z_probe = (biggest_band[0] + biggest_band[1]) // 2
    probe_img = stack.read_pixels(z_probe)
    if args.angle is not None:
        # Honour the user override but still detect the bbox.
        angle = float(args.angle)
        lo, hi = np.percentile(probe_img, [5, 95])
        thr = float(lo + MATERIAL_QUANTILE_MIX * (hi - lo))
        rot = ndimage.rotate(
            probe_img, angle, reshape=False, order=ROTATION_INTERP_ORDER,
            mode="constant", cval=0,
        )
        m = _largest_component_mask(rot, thr)
        ys, xs = np.where(m)
        h, w = probe_img.shape
        bbox = (
            max(0, int(xs.min()) - args.xy_margin),
            min(w, int(xs.max()) + 1 + args.xy_margin),
            max(0, int(ys.min()) - args.xy_margin),
            min(h, int(ys.max()) + 1 + args.xy_margin),
        )
    else:
        angle, bbox, thr = detect_rotation_and_bbox(probe_img, args.xy_margin)
    print(
        f"  Probe slice z={z_probe}, threshold={thr:.1f}, "
        f"rotation={angle:+.3f}°, bbox X[{bbox[0]}:{bbox[1]}] Y[{bbox[2]}:{bbox[3]}]"
    )

    cube_records = []
    for i, (z0, z1) in enumerate(bands, start=1):
        print(f"\n--- Processing cube {i} ---")
        rec = process_cube(
            stack, i, z0, z1, angle, bbox, args.out, tmp_dir,
        )
        cube_records.append(rec)

    # Run summary.
    run_info = {
        "dicom_dir": str(args.dicom_dir),
        "out_dir": str(args.out),
        "n_files": len(stack),
        "rows": stack.rows, "cols": stack.cols,
        "pixel_spacing_xy_mm": stack.pixel_spacing_xy,
        "pixel_spacing_z_mm": stack.pixel_spacing_z,
        "rescale_slope": stack.rescale_slope,
        "rescale_intercept": stack.rescale_intercept,
        "rotation_deg": angle,
        "bbox_xyxy": list(bbox),
        "z_stride": args.z_stride,
        "xy_margin": args.xy_margin,
        "cubes": cube_records,
    }
    (args.out / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Outputs in {args.out.resolve()}")


if __name__ == "__main__":
    main()
