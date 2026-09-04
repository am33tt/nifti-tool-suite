#!/usr/bin/env python3
"""Binarise a CT NIfTI volume into an FCM-ready indicator field.

Runs the same code as the Binarise tab. The output is uint8, ``1`` where
material and ``0`` where pore or background: the indicator alpha(x) the
finite cell method integrates against, so the file can be sampled at the
quadrature points of the embedding mesh.

Two notes for the solver side:

* Use alpha = eps (1e-6 to 1e-10) in the void, never exactly 0, or a cell
  lying entirely outside the material gives a singular element stiffness
  matrix. The file stores 0/1; the eps substitution belongs in the FCM
  code.
* Material not connected to the main body is an unconstrained rigid body
  in the analysis. ``--largest-component`` removes those islands and the
  report counts them either way.

Thresholding runs on a subsample and the binarisation streams slab by
slab, so a volume larger than RAM can be processed.

Examples
--------
    python3 scripts/binarize.py scan.nii
    python3 scripts/binarize.py scan.nii --method valley --smooth 1.0
    python3 scripts/binarize.py scan.nii --downsample 4 --largest-component
    python3 scripts/binarize.py scan.nii --method manual --threshold 13500
    python3 scripts/binarize.py scan.nii --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                        # noqa: E402

from niftitool.core import binarize as bz                 # noqa: E402
from niftitool.core.io import LazyGrayVolume, load_nifti   # noqa: E402


def _threshold(sample, args):
    """Threshold for *sample*, refined inside the specimen envelope.

    Also returns the interior values. The material peak and the
    separability check must be measured there, because on the whole volume
    the tallest histogram peak is the air around the specimen.
    """
    smoothed = sample
    if args.smooth > 0:
        from scipy import ndimage
        # Threshold the data the way it will actually be binarised.
        smoothed = ndimage.gaussian_filter(sample, args.smooth,
                                           mode="nearest")
    global_t = bz.compute_threshold(smoothed, args.method,
                                    sigma_k=args.sigma_k,
                                    manual=args.threshold)
    threshold, envelope_fraction = global_t, None
    inside = smoothed
    envelope = bz.specimen_envelope(smoothed, global_t)
    if envelope is not None and envelope.any():
        envelope_fraction = float(envelope.mean())
        candidate = smoothed[envelope]
        if candidate.size >= 1000:
            inside = candidate
            if args.refine_inside and args.method != "manual":
                refined = bz.compute_threshold(inside, args.method,
                                               sigma_k=args.sigma_k,
                                               manual=args.threshold)
                if np.isfinite(refined):
                    threshold = float(refined)
    return (float(threshold), float(global_t), envelope_fraction, smoothed,
            inside)


def process(path: Path, args) -> int:
    img = load_nifti(path)
    vol = LazyGrayVolume(img.dataobj)
    print(f"\n{path.name}  shape={tuple(int(s) for s in img.shape[:3])}  "
          f"dtype={img.get_data_dtype()}  "
          f"zooms={tuple(np.round(img.header.get_zooms()[:3], 4))}")

    sample, stride = bz.sample_volume(vol, hist_crop=args.hist_crop)
    finite = sample[np.isfinite(sample)]
    print(f"  sampled {finite.size:,} voxels (stride {stride})   "
          f"range {finite.min():.1f} .. {finite.max():.1f}")

    if args.sweep:
        inside = sample
        envelope = bz.specimen_envelope(sample, bz.otsu_threshold(sample))
        if envelope is not None and envelope.any():
            inside = sample[envelope]
        mode, sigma, rows = bz.porosity_sweep(inside)
        print(f"\n  material mode {mode:.0f}, sigma {sigma:.0f} "
              f"(inside the specimen)\n")
        print("       k    threshold   void fraction   drop vs previous")
        for row in rows:
            drop = ("        -" if row["drop_from_previous"] is None
                    else f"{row['drop_from_previous'] * 100:8.3f} pt")
            print(f"    {row['k']:4.1f}    {row['threshold']:9.0f}   "
                  f"{row['void_fraction'] * 100:9.3f} %   {drop}")
        print("\n  Pick k where the drop flattens: that plateau is the pore\n"
              "  population, and everything steeper than it is matrix noise.")
        return 0

    threshold, global_t, env_frac, smoothed, inside = _threshold(sample, args)
    warning, mode, sigma = bz.separability_note(inside, threshold)
    below = float((smoothed[np.isfinite(smoothed)] < threshold).mean())
    if env_frac is not None:
        print(f"  specimen fills {env_frac * 100:.1f} % of the box; "
              f"threshold recomputed inside it")
    print(f"  {args.method} threshold = {threshold:.2f}   "
          f"({below * 100:.1f} % of the sample below it"
          + (", after smoothing" if args.smooth > 0 else "") + ")"
          + (f"   [before refinement: {global_t:.2f}]"
             if threshold != global_t else ""))
    if warning:
        print(f"  WARNING: {warning}")

    out_path = Path(args.output) if args.output else \
        path.with_name((path.name[:-7] if path.name.lower().endswith(".nii.gz")
                        else path.stem) + "_binary.nii")

    def _progress(done, total):
        print(f"\r  binarising {100 * done // max(1, total):3d} %",
              end="", flush=True)

    info = bz.binarize_stream(img, vol, threshold, out_path,
                              smooth=args.smooth, downsample=args.downsample,
                              slab=args.slab, progress=_progress)
    print(f"\r  wrote {out_path.name}  shape={info['shape']}  "
          f"({info['seconds']:.1f} s)".ljust(60))

    total = int(np.prod(info["shape"]))
    result = bz.BinarizeResult(
        method=args.method, threshold=threshold,
        threshold_before_refinement=global_t,
        specimen_fraction_of_box=env_frac, separability_warning=warning,
        material_mode=mode, material_sigma=sigma,
        shape=info["shape"], voxel_size_mm=info["zooms"],
        material_voxels=info["material_voxels"], total_voxels=total,
        material_fraction_of_box=info["material_voxels"] / max(1, total),
        smooth_sigma=args.smooth, downsample=args.downsample,
        output_path=str(out_path),
    )

    if total <= 600_000_000:
        import nibabel as nib
        mask = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.uint8)
        changed = False
        if args.min_void > 0:
            mask, filled = bz.remove_small_voids(mask, args.min_void)
            result.voids_filled_voxels = filled
            changed = changed or filled > 0
        mask, comp = bz.component_report(mask, args.largest_component)
        result.material_components = comp.get("material_components", 0)
        result.largest_component_fraction = comp.get(
            "largest_component_fraction")
        result.removed_voxels = comp.get("removed_voxels", 0)
        changed = changed or result.removed_voxels > 0
        if changed:
            out = nib.Nifti1Image(mask, info["affine"])
            out.set_data_dtype(np.uint8)
            out.header.set_zooms(tuple(float(z) for z in info["zooms"]))
            out.header['descrip'] = b'FCM indicator: 1=material 0=void'
            nib.save(out, str(out_path))
            result.material_voxels = int(mask.sum())
            result.material_fraction_of_box = float(mask.mean())
        pores = bz.porosity_inside_specimen(mask)
        if pores:
            result.porosity_inside_specimen = pores["porosity"]
            result.specimen_voxels = pores["specimen_voxels"]
        del mask
    else:
        print("  volume too large for the component analysis; "
              "use --downsample for that step")

    for line in bz.summary_lines(result):
        print("  " + line)

    if not args.no_report:
        json_path = out_path.with_name(out_path.stem + "_report.json")
        json_path.write_text(json.dumps(result.to_dict(), indent=2),
                             encoding="utf-8")
        print(f"  report -> {json_path.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nifti", nargs="+", help="one or more .nii / .nii.gz files")
    ap.add_argument("-o", "--output",
                    help="output path (single input only)")
    ap.add_argument("--method", default="otsu", choices=list(bz.METHOD_NAMES),
                    help="threshold rule (default: otsu)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="threshold for --method manual")
    ap.add_argument("--sigma-k", type=float, default=3.0, dest="sigma_k",
                    help="k for --method sigma (default: 3)")
    ap.add_argument("--no-refine-inside", dest="refine_inside",
                    action="store_false",
                    help="do not recompute the threshold inside the specimen "
                         "envelope (refinement is on by default)")
    ap.add_argument("--hist-crop", type=float, default=1.0, dest="hist_crop",
                    help="central fraction of each axis used for the "
                         "histogram, e.g. 0.8 to exclude surrounding air")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian sigma in voxels applied before "
                         "thresholding")
    ap.add_argument("--downsample", type=int, default=1,
                    help="block-average by this factor before writing")
    ap.add_argument("--largest-component", action="store_true",
                    dest="largest_component",
                    help="keep only the largest connected material body")
    ap.add_argument("--min-void", type=int, default=0, dest="min_void",
                    help="fill pores smaller than this many voxels")
    ap.add_argument("--sweep", action="store_true",
                    help="report the void fraction against threshold and "
                         "exit, without writing a volume")
    ap.add_argument("--no-report", action="store_true", dest="no_report",
                    help="skip the JSON report")
    ap.add_argument("--slab", type=int, default=64,
                    help="slices per streaming slab (default: 64)")
    args = ap.parse_args()

    if args.method == "manual" and args.threshold is None:
        ap.error("--method manual requires --threshold")
    paths = [Path(p) for p in args.nifti]
    if args.output and len(paths) > 1:
        ap.error("--output can only be used with a single input file")

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
