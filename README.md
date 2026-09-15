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

`cupy` is optional too and enables GPU acceleration for the filter kernels;
see [Using the graphics card](#using-the-graphics-card).

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

### Background removal

The specimen is the largest 6-connected component of the solid phase --
every voxel at or above the threshold from the Threshold panel -- with its
internal cavities returned to it, so internal porosity stays part of the
specimen and only exterior air is removed. Six-connectivity is deliberate:
under 26-connectivity two bodies touching at a single corner count as one,
so a speck of reconstruction noise diagonally adjacent to the part would
join it and drag the bounding box outward.

Voxels outside the specimen are set to the fill value, zero by default.
Zero is what the rest of the tool expects: the pore threshold recognises an
already-masked volume by its exactly zero background, and the simulation
exports read zero as void. "Air median" writes the median grey level of the
removed background instead, which keeps a realistic air peak in the
histogram; a value can also be typed. Where `scl_slope`/`scl_inter` are
set, the fill is converted into stored units before it is written, so it
means the grey level it is displayed as.

Nothing is resampled. Surviving voxels are copied in the file's own dtype
and byte order, the header's intensity scaling is carried over unchanged,
and cropping shifts the affine origin so world coordinates are preserved. A
cropped, background-removed file is therefore bit-identical to the input
wherever the specimen is.

The volume is never loaded. It is read in 64 MiB slabs, the two masks the
algorithm needs are bit-packed at 0.125 bytes per voxel, and the output is
written slab by slab as it is produced, so the working set is

```
0.25 bytes/voxel  +  one slab  +  a few thousand component records
```

whatever the file size: about 0.9 GiB for a 6 GB scan and 7 GiB for a 54 GB
one, so 16 GB of RAM is enough for volumes far larger than it. The
connected components come from per-slice 2-D labellings merged across `z`
with a union-find, which is what removes the `int32` label volume
`scipy.ndimage.label` would need. An internal equivalence test asserts
voxel-for-voxel agreement with the in-memory scipy result the streaming
version replaces, and byte-for-byte agreement between the header it writes
and nibabel's.

#### When the specimen was scanned inside something

A sleeve, mould or wrapping shares the specimen's grey level and touches it
somewhere over the height of almost any scan. One contact is enough: the
largest connected component then holds both, and filling the specimen's
cavities fills the air gap between them as well, so the shell survives
untouched. On the scan this was developed against -- a printed cylinder,
1000 x 1000 x 1825 -- the two are fused over the first ~185 slices, and
99.7% of all solid voxels end up in one component.

Nor does a single cut radius separate them. The printed layers leave ribs,
so within one slice the surface radius runs from 423 to 469 voxels while the
shell's inner face sits at 456: ribs and gap overlap in radius, and any
radius that clears the shell everywhere cuts material somewhere.

Setting **Outer boundary** to *radial envelope* finds the surface direction
by direction instead. For every slice and every one of 720 rays from the
axis, the surface is the first sustained run of void on the way out;
everything beyond it goes. Three things make that robust:

* The axis is anchored to the surface, not to the mask's centre of area,
  which a shell pulls sideways and which cannot be recovered by iterating a
  centroid -- any point inside a round specimen is a fixed point of that
  iteration. It is found instead by nulling the first angular harmonic of
  the measured radius, the standard centring of a roundness measurement,
  fitted alongside the specimen's own shape so the ribs cannot leak into it.
* Rays that cross a contact find no void until beyond the shell and read too
  large. They are identified against a separable model, R(theta, z) =
  scale(z) * shape(theta), fitted with medians so that neither a minority of
  bad rays nor a minority of bad slices can carry it, and replaced by it. A
  median filter alone will not do: where a sleeve rests against the specimen
  over a stretch of its length, the contaminated slices are the majority
  inside the filter window.
* The fit is reported with the material it cut, which is the number to read
  before trusting a run. On the reference scan: radius 447 voxels (p5 424,
  p95 471), mean diameter 895, axis drift 27 x 21 voxels, 5.0% of rays
  fitted rather than measured, and 14.3% of the solid voxels removed -- the
  shell. What is left is one specimen: 2383 components instead of 922580,
  the largest holding 100.00% of the remaining material.

The method assumes the specimen is star-shaped about its axis, so that every
ray leaves it exactly once. That holds for a cylinder, a prism or a core,
and fails for a specimen with an undercut or a surface-breaking cavity wider
than the search band; for those, keep the largest-solid-body boundary.

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

## Sub-voxel geometry for FCM

The finite cell method's advantage is that the mesh need not conform to the
specimen; its accuracy comes from how well the quadrature resolves the
boundary of the indicator `alpha(x)`. A hard `uint8` mask hands the solver a
staircase whose error is set by the voxel size no matter how finely the
quadrature is refined, so two optional companion fields are written on the
**same grid, affine and zooms** as the indicator.

`*_fraction.nii` (`--fraction`, needs `--downsample > 1`)
: The material volume fraction of each exported voxel, float32 in [0, 1].
  The downsample already computes it and the old code discarded it on the
  next line by taking `mean >= 0.5`. It is counted from the fine mask, so
  it is exact: total material volume is preserved to the last bit. On a
  voxelised sphere at downsample 2 it recovers the analytic volume to
  0.005%, against 0.84% for the majority vote.

`*_sdf.nii` (`--sdf`)
: Signed distance to the material surface in mm, positive inside, float32.
  Same geometry as the indicator, written so a quadrature rule can find the
  interface between voxel centres instead of snapping it to a voxel face.
  Computed after post-processing, so the surface it measures is the one the
  solver sees. The log reports what fraction of voxels lie within one voxel
  of the surface — the share of the domain a cut-cell rule actually has to
  work on.

Neither changes the boundary-value problem. Taking `alpha` from the grey
level instead, so a boundary voxel becomes partly solid, *would* — that is a
graded-stiffness model, not a better-integrated version of the same one, and
it is deliberately not implemented here.

Two details that are easy to get wrong, and are checked by an internal
equivalence test:

- **Half-voxel convention.** A discrete transform measures to the nearest
  voxel *centre* of the opposite phase; the interface lies midway between
  centres. On an anisotropic grid the correction is half a step *along the
  surface normal*, not half the smallest spacing — using the latter puts a
  z-normal face at 0.30 mm spacing out by 0.10 mm across its whole area.
- **The array edge is a cut, not a surface.** A specimen cropped from a
  larger scan continues past the edge. Treating it as a free surface carves
  a spurious soft skin over the whole face. Override with
  `--sdf-border-is-surface` only when the volume holds the entire specimen
  with air around it.

The distance transform cannot be streamed — the nearest surface voxel may
lie anywhere — so it runs in one pass over the export grid, which is what
the solver receives anyway. An oversized grid is refused with the downsample
factor that would fit, rather than being allowed to swap the machine.

## Output files

Every science-side operation above (beam hardening, porosity, binarisation,
background removal) also has a batch/command-line entry point that runs the
same `core/` code as its tab, each writing a JSON summary of the parameters
used next to the output so a reported number can be traced back to the
settings that produced it. Those scripts, and the equivalence-test suite
that checks the streaming and GPU code paths against an in-memory reference,
are development tooling and are not part of this release.

| File | Contents |
|---|---|
| `*_report.pdf` | Slices, histogram, void analytics, pore-threshold sensitivity, beam-hardening correction, binarisation |
| `*_voids.csv` | One row per void: volume, equivalent diameter, sphericity, elongation, position |
| `*_binary.nii` | uint8 FCM indicator field (1 material, 0 void), source affine preserved |
| `*_binarize_report.json` | Method, threshold, material fraction, components, porosity |
| `*_nobg.nii.gz` | Volume with the exterior air removed, cropped to the specimen, voxels and scaling unchanged |
| `*_corrected.nii` | Beam-hardening-corrected volume, source affine preserved |
| `*_beam_hardening_fit.json` | The cupping fit, reusable on other scans |
| `*_porosity.json` | Threshold, sensitivity sweep, porosity and size statistics |
| `*.stl` | One boundary-condition surface in NIfTI world coordinates |
| `simulation_input.json` | Specimen bounding box, material, boundary conditions, solver and load stepping |
| `export_info.json` | Preprocessing record: regions, thresholds, affine, rotations, corrections applied |

## Using the graphics card

The graphics card is used in two independent places.

**Rendering.** The 3-D tab has always rendered through VTK's GPU volume
mapper. What changed is how the volume is sized for it: the scalar copy the
mapper holds lives in video memory, so the stride is now chosen against the
card's VRAM, queried from CUDA when available and otherwise from VTK's
adapter probe. It previously used free system RAM as a stand-in, which is
wrong in both directions -- a 4 GB card in a 64 GB workstation was handed a
budget it could not honour, and a card with more VRAM than the machine had
free RAM was downsampled for nothing. On integrated graphics there is no
dedicated pool to query and the system-RAM budget is restored, which is the
correct model there: its "VRAM" *is* system RAM.

**Computation.** `niftitool/core/accel.py` routes the neighbourhood filters
and morphology to `cupyx.scipy.ndimage` when a CUDA device is usable, and to
SciPy when it is not. The routed kernels are `gaussian_filter`,
`gaussian_filter1d`, `map_coordinates`, `distance_transform_edt`,
`median_filter`, `uniform_filter`, `label`, `rotate` and the `binary_*`
morphology. Each call decides for itself: the mode has to allow it, a device
has to be present and working, and the operands plus a working margin have
to fit in free VRAM. A CUDA out-of-memory error releases the pool, retries
once, and then returns the CPU result, so no caller has to handle a
GPU-specific failure.

Install the wheel matching the CUDA version the driver reports in the header
of `nvidia-smi` -- `cupy-cuda13x` for CUDA 13.x, `cupy-cuda12x` for CUDA
12.x. The `[ctk]` extra pulls NVIDIA's CUDA component wheels, so only the
driver is needed and no system-wide CUDA Toolkit:

```bash
python -m pip install "cupy-cuda13x[ctk]"
```

Use the same interpreter that runs `run.py`; installing into a different one
leaves the application reporting that CuPy is unavailable.

Nothing requires it. Without it, or on Intel/AMD integrated graphics where
CuPy has no backend, every kernel takes the SciPy path exactly as before.
That is the intended outcome rather than a degraded one: on integrated
graphics the "video memory" is system RAM behind a slower path, so
offloading would cost time rather than save it.

### What it does and does not speed up

The memory model is unchanged. Slabs are still read, filtered and written
one at a time, the bit-packed masks still cost a quarter byte per voxel, and
the 16 GB target still holds. The GPU is a coprocessor for one slab, never a
place a volume lives.

So the gain is confined to the steps whose cost is arithmetic:

| Step | Bound by | Effect |
|---|---|---|
| Beam-hardening depth field and correction | distance transform, bilinear resampling | large |
| Binarisation with smoothing | 3-D Gaussian per slab | large |
| Porosity labelling, hole filling, erosion | morphology | moderate |
| Envelope fitting | median filters on small radius maps | negligible, arrays stay below the transfer threshold |
| Background removal, masked write-out | disk I/O | none |

Background removal will not get faster, and that is not a defect. It reads
and writes tens of gigabytes; the arithmetic between them is a comparison
per voxel. Moving that comparison to the GPU only adds two PCIe transfers to
a step already waiting on the disk. Arrays under 4 MiB are kept on the CPU
for the same reason.

### Reproducibility

The GPU kernels implement the same algorithms, but floating-point reductions
run in a different order, so results agree to rounding rather than bit for
bit. Morphology, `median_filter` and the component partition from `label`
match exactly; `gaussian_filter`, `map_coordinates` and the anisotropic
scaling in the EDT agree to about 1e-6 relative. Label *numbering* is not
guaranteed to match, so never compare label ids across backends -- only
sizes and masks.

The one place this can reach a reported number is a voxel whose smoothed
grey level sits exactly on a threshold: it can fall to the other side of it.
An internal test measures that directly and fails if more than one voxel in
a million moves.

Measured on an RTX 3050 Ti with CuPy 14.2: `gaussian_filter` 3.3e-07 of the
data range with no voxel changing side of a threshold, `map_coordinates`
2.3e-07 and exactly zero at float64 (so the float32 difference is precision,
not a different algorithm), `distance_transform_edt` 1.2e-16, and exact
agreement for `median_filter`, the morphology and the `label` partition.

One backend detail: CuPy implements only the brute-force form of
multi-iteration binary morphology, so the GPU call sets `brute_force=True`.
That flag selects an algorithm, not an answer -- the test suite verifies
both settings give identical results for 1 to 5 iterations -- and the SciPy
path keeps the default, which is the faster of the two there.

Deviations are quoted as a fraction of the **data range**, not per
voxel. A pointwise relative error is the wrong metric for a CT volume: the
air peak sits near zero, so a difference of 2e-07 grey levels at a voxel
whose value is 2e-05 reads as a 1e-02 "relative" error while meaning
nothing physically. Judge a difference against the contrast in the scan.

For a run whose numbers go into a report or a paper, set the backend to
`off` -- the `GPU:` selector above the log pane, or `NIFTITOOL_GPU=off` in
the environment. The result is then reproducible on any machine regardless
of what hardware it has. Every run logs which backend produced it, so a
session transcript records this rather than leaving it to be inferred.

## Memory

Volumes larger than RAM are supported throughout. Slices are read through the
file memory map, masks and distance fields are built on a strided grid whose
size is chosen from free memory, and the corrected volume is streamed to disk
one slice at a time. Whenever the tool downsamples it reports the stride and
the smallest feature the reduced grid can still resolve.

Background removal is exact rather than strided: it visits every voxel at
full resolution and still runs in a fixed budget of roughly a quarter byte
per voxel plus one 64 MiB slab, because its masks are bit-packed and its
output is written as it is computed. Each run logs the working set it needs
against the RAM that is free, and the Stop button cancels it between slabs,
removing the partly written file.

## Layout

```
run.py                     entry point
niftitool/
  config.py                theme, presets, limits
  deps.py                  optional-import flags
  core/                    science, no GUI
    accel.py               GPU/CPU dispatch for the filter kernels
    beam_hardening.py      depth field, cupping fit, streaming correction
    pore_threshold.py      interior sampling, solid peak, threshold candidates
    porosity.py            void labelling, per-void metrics, specimen basis
    binarize.py            threshold rules, streaming indicator field, components
    fcm_field.py            sub-voxel volume-fraction and signed-distance fields for FCM
    bc_surface_export.py   affine-aware boundary surface STL
    segmentation.py        shared automatic void/solid threshold
    report.py              multi-page PDF
    background.py          specimen detection and background removal
    envelope.py            radial specimen boundary, for scans inside a sleeve
    volume_stream.py       slab-wise NIfTI reader and streaming writer
    label_stream.py        bit-packed masks, out-of-core connected components
    preview.py              resident decimated copy of the volume, for interactive viewing
    io.py geometry.py metadata.py histogram.py windowing.py slice_cache.py
    simulation.py
  ui_qt/                   PyQt6 front end, one file per tab
    figure_viewer.py       click-to-enlarge, zoomable figure windows
    help_content.py, help_dialog.py   in-app Help (Help menu), one topic per control
```
