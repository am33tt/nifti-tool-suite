# NIfTI Tool Suite

A preprocessing, measurement and simulation-export toolkit for industrial and
research CT volumes stored as NIfTI: concrete, ceramics, metals and
additively manufactured parts.

It covers three stages. Preparing a scan (inspect, reorient, crop, remove
background, correct beam hardening), measuring it (thresholds, porosity,
pore-size statistics, binarisation into an FCM indicator field), and
exporting what a finite-cell simulation needs (boundary-condition surfaces as
STL and `simulation_input.json`).

No step assumes a particular specimen shape. Cylinders, notched and T-shaped
parts, scans with air around the specimen, anisotropic voxels and flipped or
permuted NIfTI affines are handled through the header geometry rather than an
assumed one.

## Running it

```bash
pip install -r requirements.txt
python run.py
```

`vtk` provides the GPU 3-D viewer and `psutil` the memory monitor. Both are
optional and the application runs without them.

The scientific core (`niftitool.core`) is pure NumPy and SciPy and imports
without Qt, so it can be used from a notebook, a batch job or a cluster with
no GUI installed.

## The tabs

| Tab | Purpose |
|---|---|
| Tri-Planar | Windowed axial, coronal and sagittal views at true physical proportions, with crosshairs, probe, distance measurement and region of interest |
| 3-D View | Isosurface and volume rendering (VTK when available, marching cubes otherwise) |
| Histogram | Intensity distribution and per-slice mean, with the threshold marked |
| Porosity | Automatic pore threshold, connected-component void analysis, size distribution, per-void table and CSV |
| Binarise | The 0/1 indicator field the finite cell method integrates against, with the histogram and the cut it was made at |
| Beam Hardening | Cupping profile against depth below the surface, fit, correction, before and after slices |
| Metadata | Full header, affine, orientation, physical size |
| Export | PDF report, boundary-surface STL, `simulation_input.json`, `export_info.json` |

Clicking a plot opens that plot alone in its own window as a live matplotlib
figure with the navigation toolbar, so it can be panned, zoomed and saved at
any resolution. The Porosity tab shows one plot at a time behind a selector,
so each gets the whole panel.

The left-hand tool panels (Viewer, Metadata, Stats, Threshold, Binarise, Beam
Hardening, Background, Reorient, Rotate, Crop) hold the settings each action
uses. One grey level, set in the Threshold panel, feeds the porosity
analysis, the background removal and the binarisation, so the three agree.

## Methods

### Beam-hardening (cupping) correction

A polychromatic beam loses its soft photons first, so its effective
attenuation rises with penetration depth and the surface layer of a specimen
reconstructs brighter than its core. The correction measures the mean solid
grey value as a function of depth, fits a low-order polynomial `I_fit(d)`,
and rescales every voxel by `I_ref / I_fit(d)` with `I_ref` the deep-interior
value. Deep material is left untouched and the over-bright rim is pulled down
to the interior level.

Depth is the shortest distance to the specimen surface, obtained from a
Euclidean distance transform of the specimen mask evaluated at the real,
possibly anisotropic, voxel spacing. A depth defined as distance to the
nearest image border is correct only for a block cropped flush to a
rectangular field of view. For a cylinder, a notched part or a scan with air
around it that coordinate is wrong, and the fit then removes a gradient that
was never there.

For a specimen that is cropped flush, the distance transform reduces exactly
to the border distance, so nothing is lost where the simpler definition was
valid. The "Image border counts as specimen surface" option controls this:
leave it on when the volume was cropped to the specimen, turn it off when the
volume is a cut through a larger part and its edges are not physical
surfaces.

Pores are excluded from the profile. Otherwise a pore fraction that varies
with depth is read as a grey-value gradient and corrected away.

The cupping index, how much brighter the surface layer is than the core, is
reported before and after. It shows whether the correction did anything and
whether it overshot. Below about 1 % the scan shows no meaningful beam
hardening and the tool reports that instead of producing a correction.

The corrected volume is written slice by slice with the source affine,
orientation and voxel spacing copied unchanged, so it stays registered with
the original and with any boundary surfaces already exported.

### Pore threshold

The grey level separating a pore from solid material is the most influential
choice in a porosity measurement: porosity can double or halve within the
width of the noise. The tool therefore:

* samples interior voxels only, at full resolution. Sampling the whole volume
  lets the surrounding air, usually the largest population in the histogram,
  drag every automatic threshold towards it. The specimen is found per slice
  as the filled largest connected component of the material and eroded by a
  voxel or two, which drops the partial-volume surface skin that is darker
  than solid material for geometric reasons;
* fits the solid peak from quantiles of the bright flank rather than from a
  histogram half-maximum, which is biased low on small samples and would
  inflate the porosity directly;
* offers three thresholds and selects between them:

  | method | definition | when it applies |
  |---|---|---|
  | `sigma` | `mode - k*sigma` of the solid peak | always defined; conservative, noise-referenced |
  | `otsu` | maximum between-class variance | two comparable populations; biased when porosity is small |
  | `valley` | histogram minimum between the pore and solid peaks | most physical, but only when both peaks are resolved |

  `auto` uses the valley when one exists and the sigma threshold otherwise.
  With no separated pore population the tool reports `valley = none found`
  rather than a number.

* reports a sensitivity sweep: what the same volume would have measured at
  every other candidate threshold. A porosity that swings from 0.4 % to 4 %
  between `k = 2` and `k = 4` is not a measurement, and the sweep shows this
  instead of hiding it behind a single figure.

### Porosity and pore statistics

Porosity is reported as a fraction of the specimen, not of the array.
Counting the air around the specimen in the denominator would make the same
specimen scanned with a wider field of view report a lower porosity. The
specimen is the largest connected solid component with its internal holes
filled, so pores count as specimen and only exterior air does not. The
whole-array figure is recorded alongside it.

Each void is measured for volume, volume-equivalent diameter
`d_eq = (6V/pi)^(1/3)`, sphericity, elongation, centroid and bounding box,
with anisotropic voxel spacing carried through. Two limits are reported with
the results:

* A single dark voxel is not a pore. Noise, partial volume and ring artefacts
  all produce them. The count is meaningful only together with the minimum
  connected size it was measured at, so the count-versus-minimum-size sweep
  and the single-voxel fraction are always given.
* Sphericity from marching cubes reads a few percent low on small voids,
  because the triangulated surface of a voxelised sphere is longer than the
  ideal one. It is computed only for voids of at least 27 voxels.

### Tri-planar views

The three views are drawn in voxel indices but at the aspect their voxel
spacing implies, so a slice is never stretched to fill its panel. An
anisotropic grid, and a specimen much taller than it is wide, both come out
geometrically true.

By default each panel takes the shape of its own slice, the three share one
height, and every view fills its panel.

"Same scale in all views" instead gives every panel an identical box shaped
like the physical window that contains all three views, and frames each slice
in that window. One millimetre is then the same number of pixels in all
three, so a feature is the same size everywhere and one scale bar reads for
all of them. It costs space whenever the views differ in size: on a specimen
several times taller than it is wide the common window is a tall narrow strip
and the axial view ends up as a small square inside it. It is therefore off
by default and useful when the views have to be compared.

### Binarisation for the finite cell method

The Binarise tab writes the indicator field alpha(x): uint8, `1` where
material, `0` where pore or background. This is what the embedding-domain
formulation integrates against, so the file can be sampled at the quadrature
points of the FCM mesh.

Two notes for the solver side, which the tool also prints:

* Use alpha = eps (1e-6 to 1e-10) in the void, never exactly zero. A cell
  lying entirely outside the material gives a singular element stiffness
  matrix at alpha = 0. The file stores 0/1; the eps substitution belongs in
  the FCM code, not in the image.
* Material not connected to the main body is an unconstrained rigid body in
  the analysis. The islands are always counted, and "Keep only the largest
  body" removes them.

Six threshold rules are offered:

| method | definition | when it applies |
|---|---|---|
| `otsu` | between-class variance, refined by isodata | the default, and the rule the porosity analysis uses, so the two agree |
| `isodata` | Ridler-Calvard iterative intermeans | equivalent to Otsu on well-separated peaks |
| `valley` | minimum of the smoothed histogram on the material peak lower flank | closest to picking the valley by eye; needs a bimodal histogram |
| `sigma` | material mode minus k standard deviations | robust when the pore population is too small for Otsu to resolve |
| `triangle` | Zack's rule on the material flank | a dominant material peak with a long tail towards the pores |
| `manual` | a value you type | when the value is already decided |

By default the threshold is then recomputed inside the specimen envelope. Air
around the specimen is not a phase of the material, and on a
background-removed or corrected volume it is often a spike at exactly zero.
Either way it inflates the low class and pulls a global threshold away from
the pore/material boundary. The same envelope is why the material peak and
the separability check are measured on the interior: on the whole volume the
tallest histogram peak is the air.

The threshold sweep reports the void fraction against
threshold = mode - k*sigma. A real pore population appears as a plateau: once
k passes the material noise tail the fraction stops falling steeply. Choosing
k in that plateau is defensible; choosing it on the steep part measures the
material noise. If the cut ends up within two standard deviations of the
material mode the tool reports it, since that histogram is not bimodal and
the fraction that follows is meaningless.

Downsampling averages each block and re-thresholds at one half, which
preserves the material fraction. Taking every n-th voxel would discard thin
features.

### Boundary surfaces for simulation

The NIfTI file remains the specimen geometry and is never converted wholesale
to STL. Only the selected boundary regions are extracted, as the exposed
voxel faces of the material mask in one physical direction, and written
through the full NIfTI affine. That keeps a surface on the finite-cell domain
when the affine flips or permutes axes. `simulation_input.json` carries
`specimen.bounding_box_mm` derived from the same affine, so the solver domain
and the surfaces share one coordinate frame.

The region of interest is given as a percentage of the physical bounding box
and is therefore independent of voxel count. "Surface band" keeps only the
faces closest to the outermost one in the chosen direction, and "Normal
offset" shifts the finished surface along its own normal (positive outward).

## Command line

The same code paths, for batches and reproducibility.

```bash
# measure and correct beam hardening
python scripts/beam_hardening.py scan.nii
python scripts/beam_hardening.py *.nii --measure-only
python scripts/beam_hardening.py scan.nii --degree 4 --no-border-surface

# pore threshold, porosity and pore-size statistics
python scripts/porosity.py scan.nii --csv
python scripts/porosity.py scan.nii --method sigma --k 3 --min-voxels 4

# indicator field for the finite cell method
python scripts/binarize.py scan.nii --method valley --largest-component
```

Each writes a JSON summary next to the input recording every parameter, so a
reported number can be traced back to the settings that produced it.

## Output files

| File | Contents |
|---|---|
| `*_report.pdf` | Slices, histogram, void analytics, pore-threshold sensitivity, beam-hardening correction, binarisation |
| `*_voids.csv` | One row per void: volume, equivalent diameter, sphericity, elongation, position |
| `*_binary.nii` | uint8 FCM indicator field (1 material, 0 void), source affine preserved |
| `*_binarize_report.json` | Method, threshold, material fraction, components, porosity |
| `*_corrected.nii` | Beam-hardening-corrected volume, source affine preserved |
| `*_beam_hardening_fit.json` | The cupping fit, reusable on other scans |
| `*_porosity.json` | Threshold, sensitivity sweep, porosity and size statistics |
| `*.stl` | One boundary-condition surface in NIfTI world coordinates |
| `simulation_input.json` | Specimen bounding box, material, boundary conditions, solver and load stepping |
| `export_info.json` | Preprocessing record: regions, thresholds, affine, rotations, corrections applied |

## Memory

Volumes larger than RAM are supported throughout. Slices are read through the
file memory map, masks and distance fields are built on a strided grid whose
size is chosen from free memory, and the corrected volume is streamed to disk
one slice at a time. Whenever the tool downsamples it reports the stride and
the smallest feature the reduced grid can still resolve.

## Layout

```
run.py                     entry point
niftitool/
  config.py                theme, presets, limits
  deps.py                  optional-import flags
  core/                    science, no GUI
    beam_hardening.py      depth field, cupping fit, streaming correction
    pore_threshold.py      interior sampling, solid peak, threshold candidates
    porosity.py            void labelling, per-void metrics, specimen basis
    binarize.py            threshold rules, streaming indicator field, components
    bc_surface_export.py   affine-aware boundary surface STL
    segmentation.py        shared automatic void/solid threshold
    report.py              multi-page PDF
    io.py geometry.py metadata.py histogram.py windowing.py slice_cache.py
    background.py simulation.py
  ui_qt/                   PyQt6 front end, one file per tab
    figure_viewer.py       click-to-enlarge, zoomable figure windows
scripts/                   command-line entry points
```
