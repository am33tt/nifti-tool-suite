#!/usr/bin/env python3
"""Pore threshold, porosity and pore-size statistics from the command line.

Runs the same code as the Porosity tab of the GUI.

    python scripts/porosity.py scan.nii
    python scripts/porosity.py scan.nii --method sigma --k 3 --min-voxels 4
    python scripts/porosity.py *.nii --csv

The output always states the threshold the porosity was measured at, what
the same volume would report under the other candidate thresholds, the
minimum connected size a dark object had to reach to count as a pore, and
how the count changes with that filter.

Porosity is referred to the specimen rather than the whole array, so the
result does not depend on how much air surrounds the specimen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niftitool.core import porosity as po                # noqa: E402
from niftitool.core import pore_threshold as pt          # noqa: E402
from niftitool.core.io import LazyGrayVolume, load_nifti  # noqa: E402
from niftitool.utils import available_ram_mb             # noqa: E402


def process(path: Path, args) -> dict:
    img = load_nifti(path)
    vol = LazyGrayVolume(img.dataobj)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    print(f"\n{path.name}: {tuple(int(s) for s in img.shape[:3])} voxels, "
          f"voxel {spacing[0]:.4f} x {spacing[1]:.4f} x {spacing[2]:.4f} mm")

    masked, why = pt.resolve_masking(vol, args.masked)
    print(f"  background: {'already masked' if masked else 'raw scan'} ({why})")

    z_lo, z_hi, _zs, _areas = pt.find_valid_zrange(vol, masked)
    print(f"  valid slice range: {z_lo} .. {z_hi} of {vol.shape[2]}")

    sample = pt.sample_interior(vol, (z_lo, z_hi), masked,
                                erode=args.erode)
    th = pt.auto_threshold(sample, k_sigma=args.k, method=args.method,
                           manual=args.threshold)
    sens = pt.porosity_sensitivity(sample, th)
    for line in pt.summary_lines(th, sens, spacing):
        print("  " + line)

    stride = args.stride or po.choose_stride(vol.shape, available_ram_mb())
    if stride > 1:
        print(f"  analysis grid: every {stride}. voxel")
    mask = po.build_void_mask(vol, th.T, stride)
    spacing_eff = tuple(s * stride for s in spacing)

    specimen_voxels = None
    if not args.whole_array:
        specimen = po.specimen_from_void_mask(mask, erode=args.erode)
        specimen_voxels = int(specimen.sum()) or None
        if specimen_voxels:
            mask &= specimen
        del specimen

    result = po.analyze_voids(
        mask, spacing_eff, min_voxels=args.min_voxels,
        connectivity=args.connectivity, exclude_border=True,
        specimen_voxels=specimen_voxels,
        compute_sphericity=not args.no_sphericity,
    )
    result.stride = stride
    for line in po.summary_lines(result):
        print("  " + line)

    out_dir = path.parent
    stem = path.name[:-7] if path.name.lower().endswith(".nii.gz") \
        else path.stem
    summary = {
        "nifti": str(path.resolve()),
        "voxel_spacing_mm": list(spacing),
        "analysis_stride": stride,
        "background_already_masked": bool(masked),
        "valid_z_range": [int(z_lo), int(z_hi)],
        "threshold": th.to_dict(),
        "porosity_sensitivity_pct": sens,
        "porosity_pct_of_specimen": round(result.porosity_pct, 5),
        "porosity_pct_of_whole_array": round(result.volume_porosity_pct, 5),
        "specimen_voxels": int(result.specimen_voxels),
        "n_voids": int(result.n_voids),
        "min_voxels": int(result.min_voxels),
        "count_by_min_voxels": (
            pt.count_sensitivity(result.component_sizes)
            if result.component_sizes is not None else {}
        ),
        "d_mean_mm": round(result.d_mean_mm, 6),
        "d_median_mm": round(result.d_median_mm, 6),
        "d_p10_mm": round(result.d_p10_mm, 6),
        "d_p90_mm": round(result.d_p90_mm, 6),
        "d_max_mm": round(result.d_max_mm, 6),
    }
    json_path = out_dir / f"{stem}_porosity.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  summary -> {json_path}")

    if args.csv and result.voids:
        csv_path = out_dir / f"{stem}_voids.csv"
        po.voids_to_csv(result, csv_path)
        print(f"  voids   -> {csv_path}  ({len(result.voids)} rows)")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pore threshold and porosity for NIfTI CT volumes.")
    ap.add_argument("nifti", nargs="+", help="one or more .nii / .nii.gz files")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "sigma", "otsu", "valley"])
    ap.add_argument("--k", type=float, default=3.0,
                    help="noise widths below the solid peak (sigma method)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="force a fixed grey-value threshold")
    ap.add_argument("--min-voxels", type=int, default=8, dest="min_voxels",
                    help="smallest connected object counted as a pore")
    ap.add_argument("--connectivity", type=int, default=1, choices=[1, 2, 3],
                    help="1 = faces, 2 = +edges, 3 = +corners")
    ap.add_argument("--erode", type=int, default=1,
                    help="voxels trimmed off the specimen surface")
    ap.add_argument("--stride", type=int, default=0,
                    help="analysis grid step; 0 chooses one from free RAM")
    ap.add_argument("--masked", default="auto",
                    choices=["auto", "yes", "no"],
                    help="is the background already zero in the file?")
    ap.add_argument("--whole-array", action="store_true",
                    help="report porosity against the whole array instead of "
                         "the specimen (not recommended)")
    ap.add_argument("--no-sphericity", action="store_true",
                    help="skip the marching-cubes surface area (much faster)")
    ap.add_argument("--csv", action="store_true",
                    help="also write one CSV row per void")
    args = ap.parse_args()
    args.masked = {"auto": "auto", "yes": True, "no": False}[args.masked]

    failures = 0
    for p in (Path(x) for x in args.nifti):
        if not p.exists():
            print(f"{p}: not found", file=sys.stderr)
            failures += 1
            continue
        try:
            process(p, args)
        except Exception as ex:
            print(f"{p}: {ex}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
