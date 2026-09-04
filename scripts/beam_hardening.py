#!/usr/bin/env python3
"""Beam-hardening (cupping) correction from the command line.

Runs the same code as the Beam Hardening tab, so a batch of specimens is
corrected identically to one processed in the GUI.

    python scripts/beam_hardening.py scan.nii
    python scripts/beam_hardening.py scan.nii --out corrected.nii --degree 4
    python scripts/beam_hardening.py *.nii --measure-only

Depth below the specimen surface is measured with a Euclidean distance
transform of the specimen mask at the true voxel spacing, so cylinders,
notched and T-shaped parts and anisotropic grids are handled correctly.
Pass ``--no-border-surface`` when the volume is a cut through a larger
part, so the edge of the array is not treated as a physical surface.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niftitool.core import beam_hardening as bh          # noqa: E402
from niftitool.core import pore_threshold as pt          # noqa: E402
from niftitool.core.io import LazyGrayVolume, load_nifti  # noqa: E402
from niftitool.core.segmentation import otsu_threshold   # noqa: E402
from niftitool.utils import available_ram_mb             # noqa: E402


def process(path: Path, args) -> dict:
    img = load_nifti(path)
    vol = LazyGrayVolume(img.dataobj)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    print(f"\n{path.name}: {tuple(int(s) for s in img.shape[:3])} voxels, "
          f"voxel {spacing[0]:.4f} x {spacing[1]:.4f} x {spacing[2]:.4f} mm")

    stride = args.stride or bh.choose_stride(vol.shape, available_ram_mb())
    if stride > 1:
        print(f"  analysis grid: every {stride}. voxel")
    coarse = bh.coarse_volume(vol, stride)

    flat = coarse.ravel()
    if flat.size > 4_000_000:
        flat = flat[:: flat.size // 4_000_000]
    t_specimen = (args.specimen_threshold if args.specimen_threshold is not None
                  else float(otsu_threshold(flat)))

    interior = coarse[coarse >= t_specimen]
    if interior.size > 4_000_000:
        interior = interior[:: interior.size // 4_000_000]
    th = pt.auto_threshold(interior, method="auto")
    t_solid = float(th.t_valley) if th.t_valley is not None else float(t_specimen)
    print(f"  specimen threshold {t_specimen:.4g}, solid threshold {t_solid:.4g}")

    fit = bh.fit_cupping(
        vol, spacing,
        specimen_threshold=t_specimen, solid_threshold=t_solid,
        stride=stride, poly_degree=args.degree, bin_mm=args.bin_mm,
        border_is_surface=not args.no_border_surface, coarse=coarse,
    )
    for line in bh.summary_lines(fit):
        print("  " + line)

    out_dir = Path(args.out).parent if args.out else path.parent
    stem = path.name[:-7] if path.name.lower().endswith(".nii.gz") \
        else path.stem
    fit_path = out_dir / f"{stem}_beam_hardening_fit.json"
    fit_path.write_text(json.dumps(fit.to_dict(), indent=2), encoding="utf-8")
    print(f"  fit  -> {fit_path}")

    if args.measure_only:
        return fit.to_dict()

    out_path = Path(args.out) if args.out else out_dir / f"{stem}_corrected.nii"
    mask = bh.specimen_mask(coarse, t_specimen)

    def _progress(done, total):
        pct = 100.0 * done / max(1, total)
        print(f"\r  writing {done}/{total} slices ({pct:.0f} %)",
              end="", flush=True)

    info = bh.apply_correction(
        img, vol, out_path, fit, coarse_mask=mask, stride=stride,
        remove_background=args.remove_background, out_dtype=args.dtype,
        progress=_progress,
    )
    print(f"\r  volume -> {out_path}  "
          f"({info['bytes'] / 1e6:.0f} MB, {info['dtype']})".ljust(60))
    return info["fit"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Beam-hardening correction for NIfTI CT volumes.")
    ap.add_argument("nifti", nargs="+", help="one or more .nii / .nii.gz files")
    ap.add_argument("--out", help="output path (single input only)")
    ap.add_argument("--degree", type=int, default=bh.DEFAULT_POLY_DEGREE,
                    help="polynomial degree of the depth fit (default 4)")
    ap.add_argument("--bin-mm", type=float, default=bh.DEFAULT_BIN_MM,
                    dest="bin_mm", help="depth bin width in mm (default 0.25)")
    ap.add_argument("--stride", type=int, default=0,
                    help="analysis grid step; 0 chooses one from free RAM")
    ap.add_argument("--specimen-threshold", type=float, default=None,
                    help="force the specimen/air grey level")
    ap.add_argument("--no-border-surface", action="store_true",
                    help="the array edge is an arbitrary cut, not a surface")
    ap.add_argument("--remove-background", action="store_true",
                    help="set everything outside the specimen to zero")
    ap.add_argument("--dtype", choices=["float32", "int16"], default="float32")
    ap.add_argument("--measure-only", action="store_true",
                    help="fit and report, but write no corrected volume")
    args = ap.parse_args()

    paths = [Path(p) for p in args.nifti]
    if args.out and len(paths) > 1:
        ap.error("--out can only be used with a single input file")

    failures = 0
    for p in paths:
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
