# cube_analysis

Standalone scripts to turn a raw micro-CT DICOM stack into per-cube NIfTIs
and then analyse them.

Two scripts:

- **`dicom_to_cubes.py`** — preprocessing: take a folder of raw DICOM slices,
  auto-detect the in-plane rotation of the sample and the air gaps between
  the stacked cubes, then write one rotated + cropped NIfTI per cube.
- **`analyze_cubes.py`** — analysis: take NIfTI input(s) of stacked cubes
  and produce per-cube statistics, histograms, and a W/C ranking.

## dicom_to_cubes.py — DICOM → per-cube NIfTI

Defaults point at:

- input: `C:/Users/amyth/Desktop/multiscan_dcm` (5868 .dcm files)
- output: `D:/Result` (one `cube_NN.nii.gz` per detected cube, plus
  `z_profile.png`/`.npy` and `run_info.json`)

```powershell
python cube_analysis/dicom_to_cubes.py
```

To override:

```powershell
python cube_analysis/dicom_to_cubes.py `
  --dicom-dir C:/path/to/dcm `
  --out D:/Result `
  --n-cubes 3 `
  --z-stride 4
```

Useful flags:

- `--angle <deg>` — skip auto-detection and use this rotation angle (ccw).
- `--xy-margin <px>` — padding around the detected cube bbox (default 20).
- `--tmp-dir <path>` — where to put the per-cube memmap scratch file
  (defaults to the system temp dir; pick a fast SSD with ≥ ~12 GB free).

How it works (no full volume in RAM):

1. Sort `.dcm` files (filename matches monotonic Z here).
2. Strided pass: mean intensity per slice → 1-D Z-profile.
3. Detect cube Z-ranges as contiguous high-mean bands between air gaps.
4. Probe one middle slice from the longest band → PCA on the material mask
   gives the in-plane rotation; re-mask after rotation gives the XY bbox.
5. For each band: stream slices through `read → rotate → crop`, store into
   a `numpy.memmap`, then save as a gzipped NIfTI.

The output dtype is `int16`. The affine uses `PixelSpacing` in plane and the
`ImagePositionPatient` Z-difference for the Z-spacing (≈ 0.025 mm here).

## analyze_cubes.py — analysis

## What it does

1. Streams two NIfTI files (the lower + upper halves of one tall scan)
   slice-by-slice — never loads the full 14 GB volume into RAM.
2. Builds a Z-profile of mean intensity and detects the individual cubes as
   contiguous high-intensity bands separated by air gaps.
3. For each cube:
   - finds a tight XY bounding box,
   - streams the voxels to compute a histogram + statistics (mean, median,
     std, 5th/95th percentile, skewness),
   - saves a cropped NIfTI of just that cube,
   - saves a per-cube histogram plot.
4. Writes `stats.csv`, an overlay plot of all three cubes, and a short
   markdown interpretation that ranks cubes by likely W/C ratio.

## How to run

From the project root:

```powershell
python cube_analysis/analyze_cubes.py
```

Defaults point at:

- `D:/CT_output/MS_Probe1/MS_Probe4_cropped.nii.gz`
- `D:/CT_output/MS_Probe1/MS_Probe4_cropped_2.nii.gz`

To override:

```powershell
python cube_analysis/analyze_cubes.py `
  --files D:/path/lower.nii.gz D:/path/upper.nii.gz `
  --out cube_analysis/results `
  --n-cubes 3
```

Useful flags:

- `--n-cubes N` — change if the stack has more or fewer than 3 cubes.
- `--z-stride 4` — sub-sample the Z-profile pass for a much faster initial
  detection. Final per-cube processing always uses every slice.

## Expected runtime

The two files together are ~14 GB on disk (gzipped) and ~9 GB after
decompression per file in memory if loaded whole — which the script avoids.
Wall-clock time is dominated by disk I/O:

- Z-profile pass: ~5–15 min (one read of every slice).
- Per cube: another ~5–10 min (read + write of the cropped output).

Total roughly **30–60 min** on a typical workstation. Each cube output is
~1–2 GB on disk (gzipped int16).

## Outputs

```
cube_analysis/results/
├── z_profile.png           # diagnostic: profile + detected cube bands
├── z_profile.npy           # raw Z-profile (for replotting)
├── stats.csv               # per-cube statistics
├── run_info.json           # config + stats dump
├── interpretation.md       # written ranking + caveats
├── cubes/
│   ├── cube_01.nii.gz      # cropped cube (bottom)
│   ├── cube_02.nii.gz
│   └── cube_03.nii.gz      # cropped cube (top)
└── plots/
    ├── cube_01_histogram.png
    ├── cube_02_histogram.png
    ├── cube_03_histogram.png
    └── all_cubes_overlay.png
```

## How the W/C ranking is derived

Higher water-to-cement ratio at the time of casting leaves more residual
free water that ends up as **capillary porosity** in the hardened paste.
More porosity → lower bulk density → lower X-ray attenuation → lower mean
HU. The script therefore ranks cubes by mean HU and assigns the densest
cube to the lowest W/C ratio.

The ranking is robust to a global HU miscalibration (an additive/scale
error shifts all three cubes equally and does not change their order), so
the conclusion *"cube X is the 0.3 mix"* is defensible even without a
calibration phantom.

## Young's modulus — open question

There is **no universal HU → E mapping for concrete** the way Carter–Hayes
or Keller equations provide for trabecular bone. Closing this gap needs
one of:

1. **Calibration phantom in the scan** (e.g. a set of rods of known
   density: SiC / Al / water / air) to convert HU to physical density ρ.
   Then apply a published ρ → E correlation for the specific binder family
   (e.g. Feldman–Sereda type models for cement paste, or empirical fits
   from the relevant literature). Without the phantom, only **relative** E
   ordering between the three cubes is defensible.
2. **Direct mechanical testing** of companion specimens cast from the same
   batch — uniaxial compression with strain gauges for the static modulus,
   or ultrasonic pulse velocity for the dynamic modulus. Then HU and E are
   tied together empirically for this exact mix.

The interpretation file the script emits stops at the relative-ordering
claim and explicitly flags this as the next step.

## Tuning knobs (top of `analyze_cubes.py`)

- `MATERIAL_THRESHOLD` — anything above this HU is treated as "material",
  not air. Default `-200`. Lower if a cube appears very porous and the air
  / material peaks are not well separated; check `z_profile.png` and the
  per-cube histograms after a first run.
- `HIST_MIN`, `HIST_MAX`, `HIST_BINS` — histogram extent and resolution.
  Defaults give 5 HU per bin across [-2000, +5000], wide enough for any
  cement matrix and most aggregates.
- `XY_MARGIN` — padding (voxels) added around the detected cube bounding
  box before cropping.

## Notes

- The script assumes the two input files share XY shape and that file 2 is
  directly stacked on top of file 1 along Z (true for `MS_Probe4_cropped*`).
- All outputs are reproducible from the saved `run_info.json` + the input
  files.
