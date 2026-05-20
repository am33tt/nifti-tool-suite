"""
analyze_cubes.py
================

Analyse a stack of cement / concrete cubes scanned by micro-CT.

The input is two NIfTI files that together form one tall column of stacked
cubes (the second file is the upper half of the same scan). The script:

  1. Streams both files slice-by-slice (no full load -- the combined volume
     is ~14 GB) and builds a Z-profile of mean intensity.
  2. Detects individual cubes as contiguous high-intensity bands along Z,
     separated by air gaps.
  3. For each detected cube:
       - finds an XY bounding box from a middle slice,
       - streams the cube voxels to accumulate a histogram and statistics,
       - saves a tightly cropped NIfTI of just that cube,
       - saves a per-cube histogram plot.
  4. Writes a CSV of per-cube statistics, a diagnostic Z-profile plot, and
     a short markdown interpretation that ranks the cubes by likely
     water-to-cement (W/C) ratio.

Physical scientific notes
-------------------------
* The scanner exports values as 16-bit integers on an HU-like scale. Whether
  they are true Hounsfield Units depends on whether a calibration phantom was
  in the field of view. Without a phantom, treat them as "HU-like" -- the
  *relative* ordering between cubes is still meaningful even if the absolute
  numbers are off by a global scale/offset.
* Higher W/C ratio at the time of casting -> more residual capillary water
  during hydration -> more capillary porosity in the hardened paste -> lower
  bulk density -> lower mean attenuation -> lower mean HU. So the cube with
  the *highest* mean HU is the lowest-W/C cube, and vice versa.
* Histograms also encode microstructure: a wider distribution and a heavier
  left tail mean more porosity / phase heterogeneity. Aggregate (if present)
  shows up as a separate high-HU peak.
* Young's modulus is intentionally left as an open question. There is no
  universal HU -> E mapping for concrete the way there is for bone. A
  defensible mapping requires either a calibration phantom plus a published
  density-modulus correlation, or direct mechanical testing on the same mix.
  This script reports relative ordering and stops short of GPa numbers.

Run:
    python cube_analysis/analyze_cubes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy import ndimage


# --- Configuration ----------------------------------------------------------

DEFAULT_FILES = [
    r"D:/CT_output/MS_Probe1/MS_Probe4_cropped.nii.gz",
    r"D:/CT_output/MS_Probe1/MS_Probe4_cropped_2.nii.gz",
]
DEFAULT_OUT = Path(__file__).parent / "results"

# Expected number of cubes in the stack. The detector will keep the N largest
# Z-bands if it finds more than this.
EXPECTED_N_CUBES = 3

# Histogram accumulator: wide enough to cover air (~ -1000) and dense concrete
# (well above +2000). Bin width = (HIST_MAX - HIST_MIN) / HIST_BINS.
HIST_MIN = -2000
HIST_MAX = 5000
HIST_BINS = 1400  # 5 HU per bin

# Voxels above this threshold are considered "material" (not air). Used for
# masked statistics so the giant air peak does not dominate the mean/std.
MATERIAL_THRESHOLD = -200

# Margin (in voxels) added around the detected XY bounding box of each cube.
XY_MARGIN = 10

# Sample every Nth slice when building the Z-profile. 1 = every slice (slow
# but accurate). 4 is usually plenty for finding cube boundaries.
Z_PROFILE_STRIDE = 1


# --- Lazy slice access ------------------------------------------------------


@dataclass
class Stack:
    """Two (or more) NIfTI files concatenated along Z, accessed lazily."""

    imgs: list
    offsets: list  # cumulative Z offsets

    @classmethod
    def load(cls, files: List[str]) -> "Stack":
        imgs = [nib.load(f) for f in files]
        # Sanity: XY shapes and spacings must match for a clean concatenation.
        x0, y0, _ = imgs[0].shape
        for img in imgs[1:]:
            if img.shape[0] != x0 or img.shape[1] != y0:
                raise ValueError(
                    f"XY shapes differ: {imgs[0].shape} vs {img.shape}. "
                    "Cannot concatenate along Z."
                )
        offsets = [0]
        for img in imgs[:-1]:
            offsets.append(offsets[-1] + img.shape[2])
        return cls(imgs=imgs, offsets=offsets)

    @property
    def shape(self) -> Tuple[int, int, int]:
        nx, ny, _ = self.imgs[0].shape
        nz = sum(img.shape[2] for img in self.imgs)
        return (nx, ny, nz)

    def slice_z(self, z: int) -> np.ndarray:
        """Return XY slice at combined-Z index z."""
        for img, off in zip(self.imgs, self.offsets):
            if off <= z < off + img.shape[2]:
                return np.asarray(img.dataobj[:, :, z - off])
        raise IndexError(z)

    def base_affine(self) -> np.ndarray:
        """Affine of the lowest file -- valid for the combined volume since
        Z stacking is contiguous and starts at the lowest file's origin."""
        return self.imgs[0].affine.copy()


# --- Z-profile + cube detection --------------------------------------------


def build_z_profile(stack: Stack, stride: int = 1) -> np.ndarray:
    """Mean intensity per Z-slice. One streaming pass over all slices."""
    nz = stack.shape[2]
    profile = np.full(nz, np.nan, dtype=np.float64)
    t0 = time.time()
    for z in range(0, nz, stride):
        sl = stack.slice_z(z)
        profile[z] = float(sl.mean())
        if z % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  Z-profile: slice {z}/{nz}  "
                f"(elapsed {elapsed:.0f}s, mean={profile[z]:.1f})"
            )
    # If we strided, fill gaps by interpolation so the detector sees a smooth
    # 1-D signal.
    if stride > 1:
        idx = np.arange(nz)
        good = ~np.isnan(profile)
        profile = np.interp(idx, idx[good], profile[good])
    return profile


def detect_cubes_along_z(
    profile: np.ndarray, n_expected: int
) -> List[Tuple[int, int]]:
    """Detect cube Z-ranges as contiguous bands where mean intensity is well
    above the air baseline. Returns a list of (z_start, z_end_exclusive)."""
    # Heuristic threshold: midway between the minimum (air gap) and the
    # robust upper quantile (cube body) of the profile.
    p_low = float(np.percentile(profile, 5))
    p_high = float(np.percentile(profile, 95))
    threshold = p_low + 0.4 * (p_high - p_low)

    mask = profile > threshold
    # Closing fills tiny dips inside a cube (e.g. internal pores showing up
    # in the mean). Opening removes thin spurious bands.
    mask = ndimage.binary_closing(mask, iterations=5)
    mask = ndimage.binary_opening(mask, iterations=3)

    labels, n_found = ndimage.label(mask)
    if n_found == 0:
        raise RuntimeError(
            "No cubes detected. Try lowering MATERIAL_THRESHOLD or "
            "inspecting cube_analysis/results/z_profile.png."
        )

    bands = []
    for i in range(1, n_found + 1):
        zs = np.where(labels == i)[0]
        bands.append((int(zs.min()), int(zs.max()) + 1))

    # Keep the N largest bands if more were found (drops chips / noise).
    bands.sort(key=lambda b: -(b[1] - b[0]))
    bands = bands[:n_expected]
    # Order by Z position (bottom to top).
    bands.sort(key=lambda b: b[0])

    print(
        f"  Detected {n_found} bands above threshold={threshold:.1f}; "
        f"keeping top {len(bands)} by thickness."
    )
    for i, (a, b) in enumerate(bands):
        print(f"    Cube {i + 1}: Z in [{a}, {b})  ({b - a} slices)")
    return bands


def detect_xy_bbox(stack: Stack, z_mid: int) -> Tuple[int, int, int, int]:
    """Find the XY bounding box of the material in a single middle slice."""
    sl = stack.slice_z(z_mid)
    mask = sl > MATERIAL_THRESHOLD
    if not mask.any():
        # Fall back to full XY -- better than crashing.
        nx, ny = sl.shape
        return (0, nx, 0, ny)

    # Largest connected component to avoid speckle dragging the bbox out.
    labels, n = ndimage.label(mask)
    if n == 0:
        nx, ny = sl.shape
        return (0, nx, 0, ny)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labels == keep)
    x_min = max(0, int(xs.min()) - XY_MARGIN)
    x_max = min(sl.shape[1], int(xs.max()) + 1 + XY_MARGIN)
    y_min = max(0, int(ys.min()) - XY_MARGIN)
    y_max = min(sl.shape[0], int(ys.max()) + 1 + XY_MARGIN)
    # Note: numpy axis 0 is rows (Y), axis 1 is cols (X). We store both.
    return (x_min, x_max, y_min, y_max)


# --- Per-cube streaming statistics ------------------------------------------


@dataclass
class CubeStats:
    index: int
    z_start: int
    z_end: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    n_voxels_total: int = 0
    n_voxels_material: int = 0
    sum_material: float = 0.0
    sumsq_material: float = 0.0
    sumcub_material: float = 0.0  # for skewness
    hist_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(HIST_BINS, dtype=np.int64)
    )
    hist_counts_material: np.ndarray = field(
        default_factory=lambda: np.zeros(HIST_BINS, dtype=np.int64)
    )

    def add_slice(self, slab: np.ndarray) -> None:
        """slab is an int16 XY slab cropped to (y_min:y_max, x_min:x_max)."""
        flat = slab.ravel()
        self.n_voxels_total += flat.size
        # Histogram (all voxels, including air).
        counts, _ = np.histogram(
            flat, bins=HIST_BINS, range=(HIST_MIN, HIST_MAX)
        )
        self.hist_counts += counts
        # Material-only stats.
        material = flat[flat > MATERIAL_THRESHOLD].astype(np.float64)
        self.n_voxels_material += material.size
        if material.size:
            self.sum_material += material.sum()
            self.sumsq_material += (material * material).sum()
            counts_m, _ = np.histogram(
                material, bins=HIST_BINS, range=(HIST_MIN, HIST_MAX)
            )
            self.hist_counts_material += counts_m

    def finalize(self) -> dict:
        n = self.n_voxels_material
        if n == 0:
            mean = std = median = p05 = p95 = skew = float("nan")
        else:
            mean = self.sum_material / n
            var = max(0.0, self.sumsq_material / n - mean * mean)
            std = float(np.sqrt(var))
            # Percentiles from the material histogram CDF.
            edges = np.linspace(HIST_MIN, HIST_MAX, HIST_BINS + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            cdf = np.cumsum(self.hist_counts_material) / n
            median = float(np.interp(0.5, cdf, centers))
            p05 = float(np.interp(0.05, cdf, centers))
            p95 = float(np.interp(0.95, cdf, centers))
            # Skewness via histogram (approximate, plenty accurate at this
            # bin count). E[(x-mu)^3] / sigma^3
            if std > 0:
                m3 = float(
                    np.sum(
                        self.hist_counts_material * (centers - mean) ** 3
                    )
                    / n
                )
                skew = m3 / (std ** 3)
            else:
                skew = float("nan")
        return {
            "cube_index": self.index,
            "z_start": self.z_start,
            "z_end": self.z_end,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "voxels_total": int(self.n_voxels_total),
            "voxels_material": int(self.n_voxels_material),
            "material_fraction": (
                self.n_voxels_material / self.n_voxels_total
                if self.n_voxels_total
                else 0.0
            ),
            "mean_hu_material": mean,
            "median_hu_material": median,
            "std_hu_material": std,
            "p05_hu_material": p05,
            "p95_hu_material": p95,
            "skewness_material": skew,
        }


def process_cube(
    stack: Stack,
    index: int,
    z_start: int,
    z_end: int,
    out_dir: Path,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Stream one cube, write its cropped NIfTI, return stats + histograms."""
    z_mid = (z_start + z_end) // 2
    x_min, x_max, y_min, y_max = detect_xy_bbox(stack, z_mid)
    nx_c = x_max - x_min
    ny_c = y_max - y_min
    nz_c = z_end - z_start

    stats = CubeStats(
        index=index,
        z_start=z_start,
        z_end=z_end,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    # Allocate the cropped volume. dtype int16 to match input -- a single
    # cube of e.g. 1400 x 1400 x 500 = ~2 GB.
    print(
        f"  Cube {index}: bbox X[{x_min}:{x_max}] Y[{y_min}:{y_max}] "
        f"Z[{z_start}:{z_end}]  -> shape ({nx_c}, {ny_c}, {nz_c})"
    )
    print(f"  Allocating cropped volume (~{nx_c * ny_c * nz_c * 2 / 1e9:.2f} GB int16)...")
    cropped = np.empty((nx_c, ny_c, nz_c), dtype=np.int16)

    t0 = time.time()
    for k, z in enumerate(range(z_start, z_end)):
        sl = stack.slice_z(z)  # full XY slice
        # Note axis convention: nibabel returns (X, Y) for [:, :, z].
        slab = sl[x_min:x_max, y_min:y_max]
        cropped[:, :, k] = slab
        stats.add_slice(slab)
        if k % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"    cube {index} slice {k}/{nz_c}  "
                f"({elapsed:.0f}s elapsed)"
            )

    # Write cropped NIfTI with affine shifted to the cube origin.
    base = stack.base_affine()
    new_affine = base.copy()
    # base is diagonal-ish for this scan: spacings on diagonal, origin in
    # last column. Translate the origin by R @ [x_min, y_min, z_start].
    new_affine[:3, 3] = base[:3, 3] + base[:3, :3] @ np.array(
        [x_min, y_min, z_start], dtype=float
    )
    cubes_dir = out_dir / "cubes"
    cubes_dir.mkdir(parents=True, exist_ok=True)
    out_nii = cubes_dir / f"cube_{index:02d}.nii.gz"
    print(f"  Saving cropped cube to {out_nii} (gzip, this can take a few minutes)...")
    nib.Nifti1Image(cropped, new_affine).to_filename(str(out_nii))
    del cropped  # free RAM before next cube

    final = stats.finalize()
    final["output_nii"] = str(out_nii)
    return final, stats.hist_counts, stats.hist_counts_material


# --- Plotting + reporting ---------------------------------------------------


def plot_z_profile(profile, bands, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(profile, lw=0.8, color="black")
    for i, (a, b) in enumerate(bands):
        ax.axvspan(a, b, color=f"C{i}", alpha=0.25, label=f"cube {i + 1}")
    ax.set_xlabel("Combined Z slice index")
    ax.set_ylabel("Mean intensity (HU-like)")
    ax.set_title("Z-profile and detected cubes")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_histogram(hist_counts, hist_counts_material, stats: dict,
                   out_path: Path) -> None:
    edges = np.linspace(HIST_MIN, HIST_MAX, HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, scale in [(ax_lin, "linear"), (ax_log, "log")]:
        ax.plot(centers, hist_counts, color="grey", lw=0.7,
                label="all voxels")
        ax.plot(centers, hist_counts_material, color="C0", lw=0.9,
                label=f"material (>{MATERIAL_THRESHOLD} HU)")
        ax.axvline(stats["mean_hu_material"], color="C3", lw=1.0,
                   label=f"mean = {stats['mean_hu_material']:.0f}")
        ax.axvline(stats["median_hu_material"], color="C2", lw=1.0, ls="--",
                   label=f"median = {stats['median_hu_material']:.0f}")
        ax.set_yscale(scale)
        ax.set_ylabel(f"count ({scale})")
        ax.grid(True, alpha=0.3)
    ax_log.set_xlabel("HU (HU-like, int16)")
    ax_lin.set_title(f"Cube {stats['cube_index']} histogram")
    ax_lin.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_combined(hists_material, all_stats, out_path: Path) -> None:
    edges = np.linspace(HIST_MIN, HIST_MAX, HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(9, 5))
    for h, s in zip(hists_material, all_stats):
        # Normalize to PDF so cubes of different size are comparable.
        total = h.sum()
        pdf = h / total if total else h
        ax.plot(centers, pdf, lw=1.0,
                label=f"cube {s['cube_index']} (mean {s['mean_hu_material']:.0f})")
    ax.set_xlabel("HU (material voxels only)")
    ax.set_ylabel("normalized density")
    ax.set_title("Overlaid material histograms per cube")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(all_stats, out_path: Path) -> None:
    if not all_stats:
        return
    fieldnames = list(all_stats[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_stats:
            writer.writerow(s)


def write_interpretation(all_stats, out_path: Path) -> None:
    # Rank cubes by mean HU descending -- highest density first.
    ranked = sorted(all_stats, key=lambda s: -s["mean_hu_material"])
    wc_guess = [0.3, 0.4, 0.5]  # densest -> least dense
    lines = []
    lines.append("# Cube analysis -- interpretation\n")
    lines.append(
        "## Assumptions\n"
        "* Intensities are read as the int16 values stored in the NIfTI "
        f"(threshold for 'material' voxels: > {MATERIAL_THRESHOLD}). They "
        "are HU-like; absolute calibration depends on whether a phantom was "
        "in the scan.\n"
        "* Higher water-to-cement ratio at casting -> more capillary "
        "porosity in the hardened paste -> lower bulk density -> lower mean "
        "attenuation -> lower mean HU.\n"
        "* Therefore the cube with the *highest* mean HU is assigned the "
        "*lowest* W/C, and so on. This ordering is robust to a global HU "
        "miscalibration because miscalibration shifts all three cubes by "
        "the same amount.\n"
    )

    lines.append("## Per-cube statistics (material voxels only)\n")
    lines.append(
        "| rank | cube | mean HU | median HU | std HU | p05 | p95 | skew | "
        "material voxels | likely W/C |\n"
        "|------|------|---------|-----------|--------|-----|-----|------|"
        "------------------|------------|\n"
    )
    for rank, s in enumerate(ranked):
        wc = wc_guess[rank] if rank < len(wc_guess) else "?"
        lines.append(
            f"| {rank + 1} | {s['cube_index']} | "
            f"{s['mean_hu_material']:.0f} | "
            f"{s['median_hu_material']:.0f} | "
            f"{s['std_hu_material']:.0f} | "
            f"{s['p05_hu_material']:.0f} | "
            f"{s['p95_hu_material']:.0f} | "
            f"{s['skewness_material']:+.2f} | "
            f"{s['voxels_material']:,} | "
            f"{wc} |\n"
        )

    lines.append("\n## Reading the histograms\n")
    lines.append(
        "* A *broader* distribution and a heavier left tail suggest more "
        "porosity / phase heterogeneity, both of which scale with W/C.\n"
        "* A distinct high-HU shoulder/peak indicates aggregate inclusions "
        "(if any -- pure mortar samples will not show this).\n"
        "* The grey 'all voxels' curve includes the air around the cube; "
        "the blue 'material' curve is what to interpret physically.\n"
    )

    lines.append("\n## Young's modulus -- open question\n")
    lines.append(
        "There is no universal HU -> E mapping for concrete the way "
        "Carter-Hayes / Keller equations provide for trabecular bone. A "
        "defensible mapping needs one of:\n\n"
        "1. **A calibration phantom in the scan** (e.g. a SiC / Al / water "
        "rod set) to convert HU to physical density rho, plus a published "
        "rho -> E correlation for the specific mix family. Without the "
        "phantom, only relative E ordering is defensible.\n"
        "2. **Direct mechanical testing** on companion specimens cast from "
        "the same batch (uniaxial compression with strain gauges, or "
        "ultrasonic pulse velocity for the dynamic modulus). Then HU and E "
        "are correlated empirically.\n\n"
        "Until one of the above is in place, this report stops at: "
        "*ranking cubes by likely W/C from mean HU, and showing the "
        "distribution shapes that drive that ranking.*\n"
    )

    out_path.write_text("".join(lines), encoding="utf-8")


# --- Main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+", default=DEFAULT_FILES,
        help="Input NIfTI files (stacked along Z, bottom first).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output directory.",
    )
    parser.add_argument(
        "--n-cubes", type=int, default=EXPECTED_N_CUBES,
        help="Number of cubes expected in the stack.",
    )
    parser.add_argument(
        "--z-stride", type=int, default=Z_PROFILE_STRIDE,
        help="Stride for the Z-profile pass (>1 = faster, less precise).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    print(f"Loading NIfTI headers for {len(args.files)} files...")
    stack = Stack.load(args.files)
    print(f"  Combined shape: {stack.shape}")
    print(f"  Spacing: {stack.imgs[0].header.get_zooms()} mm")

    print("\nBuilding Z-profile (one streaming pass)...")
    profile = build_z_profile(stack, stride=args.z_stride)
    np.save(out_dir / "z_profile.npy", profile)

    print("\nDetecting cubes along Z...")
    bands = detect_cubes_along_z(profile, args.n_cubes)
    plot_z_profile(profile, bands, out_dir / "z_profile.png")

    all_stats = []
    hists_material = []
    for i, (z0, z1) in enumerate(bands, start=1):
        print(f"\n--- Processing cube {i} ---")
        stats, hist_all, hist_mat = process_cube(stack, i, z0, z1, out_dir)
        all_stats.append(stats)
        hists_material.append(hist_mat)
        plot_histogram(
            hist_all, hist_mat, stats,
            out_dir / "plots" / f"cube_{i:02d}_histogram.png",
        )

    print("\nWriting summary outputs...")
    write_csv(all_stats, out_dir / "stats.csv")
    plot_combined(
        hists_material, all_stats, out_dir / "plots" / "all_cubes_overlay.png"
    )
    write_interpretation(all_stats, out_dir / "interpretation.md")

    # Also dump raw JSON of stats + run config for reproducibility.
    run_info = {
        "inputs": [str(f) for f in args.files],
        "combined_shape": stack.shape,
        "spacing_mm": [float(x) for x in stack.imgs[0].header.get_zooms()],
        "config": {
            "material_threshold": MATERIAL_THRESHOLD,
            "hist_range": [HIST_MIN, HIST_MAX],
            "hist_bins": HIST_BINS,
            "xy_margin": XY_MARGIN,
            "z_profile_stride": args.z_stride,
        },
        "cubes": all_stats,
    }
    (out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
