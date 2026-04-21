"""Optional-dependency probes.

Imports the heavy scientific stack once and exposes :data:`HAS_*` flags so
the rest of the codebase can degrade gracefully when a package is missing.
The GUI uses these flags to show an inline warning instead of crashing.
"""

# ── numpy + nibabel + scipy ───────────────────────────────────────────────────
try:
    import numpy as np
    import nibabel as nib
    import nibabel.orientations as nio
    import scipy.ndimage as ndimage
    HAS_NIBABEL = True
except ImportError:
    np = None          # type: ignore
    nib = None         # type: ignore
    nio = None         # type: ignore
    ndimage = None     # type: ignore
    HAS_NIBABEL = False

# ── matplotlib ────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk,
    )
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    matplotlib = None            # type: ignore
    Figure = None                # type: ignore
    FigureCanvasTkAgg = None     # type: ignore
    NavigationToolbar2Tk = None  # type: ignore
    mcm = None                   # type: ignore
    mcolors = None               # type: ignore
    HAS_MPL = False

# ── matplotlib 3-D ────────────────────────────────────────────────────────────
try:
    from mpl_toolkits.mplot3d import Axes3D                 # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_3D = True
except ImportError:
    Poly3DCollection = None  # type: ignore
    HAS_3D = False

# ── scikit-image (marching cubes for 3-D surface view) ────────────────────────
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    marching_cubes = None  # type: ignore
    HAS_SKIMAGE = False

# ── VTK (GPU 3-D rendering, Slicer-style) ─────────────────────────────────────
try:
    import vtk  # noqa: F401
    from vtkmodules.tk.vtkTkRenderWidget import vtkTkRenderWidget  # noqa: F401
    from vtkmodules.util import numpy_support as vtk_numpy_support  # noqa: F401
    HAS_VTK = True
except Exception:
    vtk = None                 # type: ignore
    vtkTkRenderWidget = None   # type: ignore
    vtk_numpy_support = None   # type: ignore
    HAS_VTK = False

# ── psutil (optional RAM monitor) ─────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False


def missing_report() -> list[str]:
    """List of human-readable missing-package labels (empty if all present)."""
    missing: list[str] = []
    if not HAS_NIBABEL: missing.append("nibabel / numpy / scipy")
    if not HAS_MPL:     missing.append("matplotlib")
    if not HAS_PSUTIL:  missing.append("psutil  (optional)")
    if not HAS_3D:      missing.append("mpl_toolkits  (optional)")
    if not HAS_SKIMAGE: missing.append("scikit-image  (optional — for 3-D surface)")
    if not HAS_VTK:     missing.append("vtk  (recommended — GPU 3-D viewer)")
    return missing
