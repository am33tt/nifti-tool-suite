"""The text shown in Help.

Every control that used to carry a hover tooltip is documented here
instead, alongside the workflow context a tooltip had no room for. Keeping
it as data rather than as strings scattered through the widget code means
one place to read, one place to correct, and no chance of a control's
explanation drifting away from the control.

Bodies are the small HTML subset :class:`QTextBrowser` renders. Write
plain prose in ``<p>``, term lists in ``<dl>``, and use :func:`terms` for
the option-by-option lists that most of these need -- several controls are
a combo box whose entries each need a sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    """One help entry: a title, the searchable body, and where it applies."""
    title: str
    body: str
    #: Extra words that should match this topic in the search box but do
    #: not appear in the text -- old names, synonyms, the control's label.
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Section:
    title: str
    topics: tuple[Topic, ...] = field(default_factory=tuple)


def terms(*pairs: tuple[str, str]) -> str:
    """A definition list from ``(term, description)`` pairs."""
    items = "".join(
        f"<dt><b>{term}</b></dt><dd>{description}</dd>" for term, description in pairs)
    return f"<dl>{items}</dl>"


def paragraphs(*texts: str) -> str:
    return "".join(f"<p>{t}</p>" for t in texts)


# ---------------------------------------------------------------------------

_GETTING_STARTED = Section("Getting started", (
    Topic(
        "Opening a volume",
        paragraphs(
            "Use <b>Open</b> on the top bar, or drag a <code>.nii</code> or "
            "<code>.nii.gz</code> file onto the window.",
            "A plain <code>.nii</code> is memory-mapped, so only the slices "
            "you look at are read. A <code>.nii.gz</code> cannot be: nibabel "
            "has to decompress the whole thing into RAM before any read. For "
            "a large compressed file, use <b>Gunzip</b> on the top bar first "
            "and open the result instead.",
            "On a machine with 16 GB or less, opening a file above about "
            "1.5 GB offers a pre-load crop. The crop reads only the bytes it "
            "keeps, writes a <code>_cropped</code> file beside the original, "
            "and loads that in place of it. The original is untouched.",
        ),
        ("open", "load", "drag", "drop", "gunzip", "decompress", "crop"),
    ),
    Topic(
        "Volumes larger than RAM",
        paragraphs(
            "Every step that touches all the voxels is streamed: the volume "
            "is read in slabs, the result is written as it is computed, and "
            "peak memory is set by the slab rather than by the file. A scan "
            "much larger than the machine's RAM is expected to work.",
            "Background removal is exact rather than strided. It visits every "
            "voxel at full resolution and still runs in roughly a quarter "
            "byte per voxel plus one 64 MiB slab, because its masks are "
            "bit-packed. A 3000&#179; scan needs about 7 GiB and runs on a "
            "16 GB machine.",
            "Where a step must downsample, it says so and reports the stride "
            "and the smallest feature the reduced grid can still resolve.",
        ),
        ("memory", "ram", "streaming", "slab", "large", "big"),
    ),
    Topic(
        "Keyboard shortcuts",
        terms(
            ("Ctrl+O", "Open a NIfTI file."),
            ("F1", "Open this Help window."),
            ("Ctrl+F1", "Open Help with the cursor in the search box."),
            ("Enter", "In a slice-index box, jump to that slice."),
        ),
        ("shortcut", "keyboard", "ctrl", "f1", "hotkey", "keys"),
    ),
    Topic(
        "Stopping a long task",
        paragraphs(
            "A <b>Stop</b> button appears on the top bar, beside "
            "<b>Open</b>, while a task is running. The current step "
            "finishes first, then the task gives up and leaves the volume "
            "unchanged. A partly written output file is removed.",
        ),
        ("stop", "cancel", "abort"),
    ),
))

_THRESHOLD = Section("Threshold", (
    Topic(
        "The void/solid threshold",
        paragraphs(
            "The grey level that separates void from solid. It is the input "
            "to background removal and to the 3-D view, and the starting "
            "point for the porosity and binarisation steps.",
            "<b>auto</b> finds the split with Otsu's method on a subsample of "
            "the volume. You can also type a grey value directly.",
        ),
        ("otsu", "auto", "void", "solid", "grey", "gray"),
    ),
))

_BACKGROUND = Section("Background removal", (
    Topic(
        "What it does",
        paragraphs(
            "Finds the specimen as the largest connected solid region, keeps "
            "its internal pores, removes everything outside it and saves a "
            "clean new NIfTI. It uses the grey level from the Threshold "
            "panel.",
            "Surviving voxels are copied in their stored dtype and byte "
            "order, and the header's scaling is carried over unchanged, so "
            "they are bit-identical to the input. Nothing is interpolated, "
            "resampled or re-quantised. Cropping shifts the affine's origin, "
            "so world coordinates are preserved.",
        ),
        ("background", "specimen", "air", "remove"),
    ),
    Topic(
        "Outer boundary",
        "<p>How the outside of the specimen is decided.</p>" + terms(
            ("largest solid body",
             "The specimen is the biggest connected piece of material. Right "
             "when it is the only thing in the scan above the threshold."),
            ("radial envelope",
             "The surface is found direction by direction, as the first void "
             "met on the way out from the axis. Use it when the specimen was "
             "scanned inside a sleeve, mould or wrapping: those touch it "
             "somewhere over the height of the scan, and one contact is "
             "enough to make the two a single connected body that no "
             "threshold can separate. Assumes the specimen is convex about "
             "its axis, as a cylinder or a core is."),
        ),
        ("envelope", "sleeve", "mould", "mold", "wrapping", "boundary"),
    ),
    Topic(
        "Fill outside",
        "<p>What is written outside the specimen.</p>" + terms(
            ("zero",
             "What the rest of the tool expects: the porosity and threshold "
             "steps recognise a masked volume by its exactly zero "
             "background, and the simulation exports read zero as void."),
            ("air median",
             "The median grey level of the removed background, which keeps a "
             "realistic air peak in the histogram."),
            ("grey level", "A value you choose."),
        ),
        ("fill", "zero", "air", "median", "outside"),
    ),
    Topic(
        "Margin and cropping",
        paragraphs(
            "<b>Margin</b> is the padding in voxels kept around the "
            "specimen's bounding box.",
            "<b>Crop to specimen</b> also trims the volume to that padded "
            "box. The affine origin is shifted to match, so the result still "
            "sits in the same place in world coordinates.",
        ),
        ("margin", "crop", "bounding box", "padding"),
    ),
))

_BEAM_HARDENING = Section("Beam-hardening correction", (
    Topic(
        "What it does",
        paragraphs(
            "A polychromatic beam hardens as it passes through material, so "
            "the reconstructed value of the same material falls with depth "
            "below the surface -- the cupping artefact. The correction fits "
            "the measured value against true depth and divides it out.",
            "Depth is a physical distance from an exact Euclidean transform, "
            "with anisotropic voxels handled through the voxel spacing, so it "
            "is in millimetres and not a voxel count.",
        ),
        ("cupping", "polychromatic", "hardening", "depth"),
    ),
    Topic(
        "Fit settings",
        terms(
            ("Polynomial degree",
             "Degree of the depth fit. 4 is a good default; above 6 it starts "
             "fitting noise in the sparse deep bins."),
            ("Depth bin width", "Bin width for the depth profile, in mm."),
            ("Analysis stride",
             "The step of the analysis grid. 0 chooses one from the free "
             "memory. The correction itself is applied at full resolution."),
        ),
        ("polynomial", "degree", "bin", "stride", "fit"),
    ),
    Topic(
        "Border is a real surface",
        paragraphs(
            "<b>On:</b> the specimen was cropped flush to its bounding box, "
            "so the edge of the volume is a real surface and material "
            "touching it sits at zero depth.",
            "<b>Off:</b> the volume is an arbitrary cut through a larger "
            "part, so depth is measured only from real material/air "
            "interfaces. Getting this wrong makes every voxel near the cut "
            "face look far shallower than it is.",
        ),
        ("border", "surface", "edge", "cut"),
    ),
    Topic(
        "Output dtype",
        paragraphs(
            "<b>float32</b> keeps the corrected values exactly. <b>int16</b> "
            "halves the file size but rounds and clips them.",
        ),
        ("dtype", "float32", "int16", "precision"),
    ),
))

_BINARISE = Section("Binarisation for the finite cell method", (
    Topic(
        "What it writes",
        paragraphs(
            "The indicator field the finite cell method integrates against: "
            "<code>uint8</code>, 1 where material, 0 where pore or "
            "background. The threshold is measured inside the specimen "
            "envelope, so air around the part cannot drag it.",
            "Two optional companion fields carry sub-voxel geometry the "
            "binary file cannot &mdash; see <i>Sub-voxel fields for FCM</i>.",
        ),
        ("binarise", "binarize", "indicator", "fcm", "uint8"),
    ),
    Topic(
        "Threshold method",
        terms(
            ("otsu",
             "Between-class variance, refined by isodata. The same rule the "
             "porosity step uses, so the two agree."),
            ("isodata", "Ridler&ndash;Calvard iterative intermeans."),
            ("valley",
             "Minimum of the smoothed histogram on the material peak's lower "
             "flank."),
            ("sigma",
             "Material mode minus k standard deviations. Robust when the pore "
             "peak is too small for Otsu."),
            ("triangle",
             "Zack's rule; suits a dominant material peak with a long tail."),
            ("manual", "The value typed below."),
        ),
        ("otsu", "isodata", "valley", "sigma", "triangle", "manual",
         "ridler", "calvard", "zack"),
    ),
    Topic(
        "Smoothing, downsampling and cleanup",
        terms(
            ("Smooth",
             "Gaussian sigma in voxels applied before thresholding. Smooths "
             "speckle, but also rounds thin features &mdash; keep it below "
             "one voxel unless the scan is very noisy."),
            ("Downsample",
             "Block-average by this factor before writing. A block becomes "
             "material when at least half of it is, which preserves the "
             "material fraction far better than taking every n-th voxel."),
            ("Minimum void",
             "Fill pores smaller than this many voxels. 0 keeps every pore."),
            ("Keep only the largest body",
             "Material not connected to the main body is an unconstrained "
             "rigid body in the analysis. The islands are counted either "
             "way."),
            ("Refine inside the specimen",
             "Recompute the threshold using only voxels inside the specimen "
             "envelope, so surrounding air cannot pull it away from the "
             "pore/material boundary."),
            ("k for sigma",
             "How many standard deviations below the material mode the sigma "
             "method cuts."),
        ),
        ("smooth", "gaussian", "downsample", "min void", "largest",
         "refine", "k", "sigma"),
    ),
    Topic(
        "Sub-voxel fields for FCM",
        paragraphs(
            "The finite cell method does not need a conforming mesh; its "
            "accuracy comes from how well the quadrature resolves the "
            "boundary of the indicator. A hard binary mask hands the solver a "
            "staircase whose error is fixed by the voxel size no matter how "
            "finely the quadrature is refined.",
            "Two optional fields give the boundary back. Both are written on "
            "the same grid, affine and zooms as the indicator, so the solver "
            "reads them in one frame.",
        ) + terms(
            ("Volume fraction  (<code>*_fraction.nii</code>)",
             "The material fraction of each exported voxel, float32 in "
             "[0,&nbsp;1]. Only meaningful when downsampling, where a coarse "
             "voxel really does contain a mixture. Counted from the fine "
             "mask, so it is exact: the total material volume is preserved."),
            ("Signed distance  (<code>*_sdf.nii</code>)",
             "Distance to the material surface in mm, positive inside. The "
             "same geometry the indicator describes, written so a quadrature "
             "rule can find the interface between voxel centres instead of "
             "snapping it to a voxel face."),
        ) + paragraphs(
            "Neither changes the boundary-value problem. Taking the "
            "indicator from the grey level instead, so a boundary voxel "
            "becomes partly solid, <i>would</i> &mdash; that is a "
            "graded-stiffness model, not a better-integrated version of the "
            "same one.",
            "The signed distance treats the edge of the array as a cut "
            "through the specimen, not as a free surface, because a specimen "
            "cropped out of a larger scan continues past it. Override that "
            "only when the volume holds the whole specimen with air around "
            "it.",
        ),
        ("sdf", "signed distance", "fraction", "level set", "quadrature",
         "sub-voxel", "subvoxel", "partial volume"),
    ),
))

_POROSITY = Section("Porosity", (
    Topic(
        "Threshold method",
        terms(
            ("auto",
             "Histogram valley when a real pore peak exists, otherwise the "
             "noise-referenced sigma threshold."),
            ("sigma",
             "Mode minus k &times; sigma of the solid peak. Conservative, and "
             "always defined."),
            ("otsu",
             "Maximum between-class variance. Biased when the porosity is "
             "small."),
            ("valley",
             "Minimum between the pore and solid peaks. The most physical, "
             "but it needs two resolved peaks."),
        ) + paragraphs(
            "<b>k sigma</b> sets how many noise widths below the solid peak "
            "the sigma threshold sits.",
            "The report includes a sensitivity sweep, so you can see how much "
            "the porosity figure depends on the threshold you chose. Quote "
            "that spread rather than a single number.",
        ),
        ("auto", "sigma", "otsu", "valley", "k", "sensitivity"),
    ),
    Topic(
        "What counts as a pore",
        terms(
            ("Exclude border voids",
             "Discard components touching the volume boundary &mdash; the "
             "exterior air around the specimen."),
            ("Specimen only",
             "Report porosity as a fraction of the specimen rather than of "
             "the whole array, and ignore voids outside it. Without it, air "
             "around the specimen counts towards the denominator, and the "
             "same specimen scanned with a wider field of view would report a "
             "lower porosity."),
            ("Sphericity",
             "Mesh each void with marching cubes to measure surface area. "
             "Disable for a much faster first pass on huge volumes. Voids too "
             "small to mesh reliably report no sphericity rather than a "
             "wrong one."),
        ),
        ("border", "specimen", "sphericity", "marching cubes", "denominator"),
    ),
))

_SIMULATION = Section("Simulation export", (
    Topic(
        "Volume and solver input",
        terms(
            ("NIfTI file name",
             "File name only &mdash; the C++ pipeline resolves it next to "
             "<code>simulation_input.json</code>."),
            ("Void threshold",
             "Normalised to [0,&nbsp;1] against the data range. Filled in "
             "automatically from the Otsu split when a file is loaded."),
        ),
        ("mlhp", "simulation", "solver", "json", "void threshold"),
    ),
    Topic(
        "Boundary-condition surfaces",
        terms(
            ("Name",
             "A name containing <i>load</i>, <i>support</i>, <i>roller</i>, "
             "<i>fixed</i> or <i>grip</i> pre-selects the matching "
             "boundary-condition preset."),
            ("Direction",
             "The physical outward normal of the wanted face, in world "
             "coordinates &mdash; not a voxel-index direction."),
            ("Auto threshold",
             "Estimate the material/background split from the loaded volume "
             "with Otsu, instead of typing grey values."),
            ("Band",
             "Keep only faces within this distance of the outermost face in "
             "the chosen direction. Set 0 to keep every exposed face in the "
             "region, including pore walls."),
            ("Normal offset",
             "Moves the finished STL along the selected face direction. "
             "Positive moves outward, negative inward."),
            ("Keep largest patch",
             "Drop speckles left by threshold noise, keeping only the largest "
             "vertex-connected patch."),
        ),
        ("stl", "boundary condition", "bc", "load", "support", "roller",
         "grip", "band", "offset", "normal"),
    ),
))

_VIEWS = Section("Views", (
    Topic(
        "Tri-planar",
        paragraphs(
            "Three orthogonal slices with linked crosshairs. Type a slice "
            "index in a box and press Enter to jump to it.",
            "<b>Same scale</b> off (the default): each view frames its own "
            "slice and fills its panel. The best use of the space, and the "
            "only sensible choice when the specimen is much taller than it is "
            "wide.",
            "<b>Same scale</b> on: every view shows the same physical window, "
            "so a feature is the same size in all three and one scale bar "
            "reads for all of them &mdash; at the cost of the smaller views "
            "sitting in a lot of empty space. The geometry is true either "
            "way: a slice is never stretched to fill a panel.",
        ),
        ("triplanar", "tri-planar", "crosshair", "scale bar", "slice"),
    ),
    Topic(
        "Why dragging is fast (the interactive preview)",
        paragraphs(
            "An axial slice is contiguous on disk and costs one read. A "
            "sagittal or coronal slice is not: it takes one row from every "
            "plane in the file, so collecting it touches the whole volume. "
            "On a scan larger than the page cache that is the entire cost "
            "of dragging a slider, and every new position pays it again.",
            "So after a volume is loaded, a decimated copy is read once, in "
            "the background, and kept in memory at the resolution the "
            "panels display. Dragging reads that copy; when you stop, the "
            "panels re-read at full resolution. The log says when it is "
            "ready.",
            "The preview is for interaction only. Grey values under the "
            "pointer, measurements, thresholds, histograms and exported "
            "images all come from the volume itself, never from it.",
            "No preview is built when the volume already fits in memory - "
            "a slice of a resident array is a memory read already - or when "
            "the volume is small enough that reading it directly is cheap.",
        ),
        ("slow", "lag", "slider", "drag", "preview", "fast", "smooth",
         "responsive", "performance"),
    ),
    Topic(
        "3-D view",
        terms(
            ("View preset", "Jump the camera to a preset angle."),
            ("Axes",
             "Show the labelled bounding-box axes on the volume. The small "
             "XYZ gizmo in the corner is always visible."),
            ("Move",
             "When on, click-drag translates the picked volume or STL instead "
             "of rotating the view. Useful for aligning an STL overlay with "
             "the volume."),
            ("Slice planes",
             "In sliced mode, each axis has a checkbox that shows or hides "
             "its plane, and a box to type a slice index."),
            ("STL focus", "Zoom the camera to that STL's bounds."),
        ) + paragraphs(
            "The volume is rendered on the graphics card. If it does not fit "
            "in video memory it is strided down, and the log reports the "
            "factor and the budget it was measured against.",
        ),
        ("3d", "vtk", "render", "camera", "stl", "gizmo", "volume"),
    ),
    Topic(
        "Figures",
        paragraphs(
            "Click a plot to open it on its own, full size and zoomable.",
        ),
        ("figure", "plot", "zoom", "chart"),
    ),
))

_PERFORMANCE = Section("Performance and reproducibility", (
    Topic(
        "GPU acceleration",
        paragraphs(
            "The <b>GPU</b> selector above the log pane decides where the "
            "filter kernels run.",
        ) + terms(
            ("auto",
             "Use the graphics card when one is available and the data fits. "
             "The default."),
            ("on",
             "The same, but say so loudly in the log when the card cannot be "
             "used, so a run meant to be accelerated does not quietly fall "
             "back."),
            ("off", "Always use the CPU."),
        ) + paragraphs(
            "Acceleration needs an NVIDIA card with CuPy installed. Without "
            "it, or on integrated graphics, everything runs on the CPU "
            "exactly as before &mdash; that is the intended behaviour, not a "
            "degraded one.",
            "It helps the steps whose cost is arithmetic: the "
            "beam-hardening depth field and correction, smoothed "
            "binarisation, and the morphology behind porosity. It does not "
            "help background removal, which is bounded by reading and writing "
            "the file rather than by computation.",
        ),
        ("gpu", "cuda", "cupy", "vram", "nvidia", "accelerate", "speed"),
    ),
    Topic(
        "Reproducible numbers",
        paragraphs(
            "The GPU kernels implement the same algorithms, but "
            "floating-point reductions run in a different order, so results "
            "agree to rounding rather than bit for bit. Morphology, median "
            "filtering and the connected-component partition match exactly; "
            "smoothing, resampling and the anisotropic distance scaling agree "
            "to about one part in a million of the data range.",
            "For a run whose numbers go into a report or a paper, set the "
            "selector to <b>off</b>. The result is then reproducible on any "
            "machine regardless of its hardware. Every run records which "
            "backend produced it in the log, so a session transcript states "
            "this rather than leaving it to be inferred.",
        ),
        ("reproducible", "determinism", "paper", "publication", "rounding"),
    ),
    Topic(
        "Memory monitor",
        paragraphs(
            "The status bar shows this process's resident memory and, when a "
            "graphics card is present, its video memory. Video memory is what "
            "the 3-D view and the accelerated kernels are budgeted against, "
            "so it is the number to watch when a render is being strided "
            "down.",
        ),
        ("ram", "vram", "memory", "status bar", "monitor"),
    ),
))

#: Every section, in the order they appear in the Help window.
SECTIONS: tuple[Section, ...] = (
    _GETTING_STARTED,
    _THRESHOLD,
    _BACKGROUND,
    _BEAM_HARDENING,
    _BINARISE,
    _POROSITY,
    _SIMULATION,
    _VIEWS,
    _PERFORMANCE,
)

#: Shown when the window opens, before a topic is chosen.
WELCOME = paragraphs(
    "Pick a topic on the left, or type in the search box to filter them.",
    "The sections follow the order of a normal run: open a volume, set a "
    "threshold, remove the background, correct beam hardening, binarise for "
    "the finite cell method, then measure porosity or export for simulation.",
)


def iter_topics():
    """Every ``(section, topic)`` pair, in display order."""
    for section in SECTIONS:
        for topic in section.topics:
            yield section, topic


def matches(topic: Topic, query: str) -> bool:
    """Whether *topic* should survive the search box holding *query*."""
    query = query.strip().lower()
    if not query:
        return True
    haystack = " ".join((topic.title, topic.body, " ".join(topic.keywords)))
    return all(word in haystack.lower() for word in query.split())
