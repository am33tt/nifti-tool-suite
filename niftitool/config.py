"""Application-wide constants: theme, fonts, presets, limits.

No runtime logic. Numeric and stylistic settings are collected here so
they can be tuned in one place.
"""

# Theme
BG       = "#FDFAF6"
PANEL    = "#F5F0E8"
PANEL2   = "#EDE6D8"
BORDER   = "#D4C4A8"
ACCENT   = "#E07B2A"
ACCENT2  = "#6A9E3F"
WARN     = "#C08020"
ERR      = "#C0392B"
TEXT     = "#2C2416"
TEXT_DIM = "#7A6A52"
ENTRY_BG = "#FFFFFF"
BTN_BG   = "#EDE6D8"
BTN_HOV  = "#D4C4A8"
TEAL     = "#1A7A8A"

# Per-axis colours, shared by the tri-planar crosshairs, the sliders and the
# 3-D cube-axes labels so a colour always denotes the same axis.
AXIS_COLOR = {
    'X': "#E5484D",  # sagittal
    'Y': "#2FA84F",  # coronal
    'Z': "#3B82F6",  # axial
}

FONT_MONO  = ("Consolas", 10)
FONT_UI    = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_HEAD  = ("Segoe UI Semibold", 11)
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_CODE  = ("Consolas", 9)

# Colormaps
CMAPS = ["gray", "hot", "jet", "viridis", "bone", "plasma", "inferno", "turbo"]

# CT windowing presets: (window width, window center) in HU
CT_PRESETS = {
    "-- preset --": None,
    "AM Concrete" : (200,  100),
    "Bone"        : (2000, 400),
    "Soft Tissue" : (400,  40),
    "Lung"        : (1500, -600),
    "Brain"       : (80,   40),
}

# Performance limits
# Max voxels plotted in 3-D scatter (the viewer uses marching cubes).
MAX_3D_VOXELS   = 60_000
# Max voxels fed to np.histogram, anything bigger is subsampled.
MAX_HIST_VOXELS = 10_000_000
# Stats panel subsample threshold (for live Min/Max/Mean/p1/p99 display).
STATS_SUBSAMPLE_VOXELS = 5_000_000
# Triplanar slider debounce in milliseconds (caps matplotlib redraws).
SLIDER_DEBOUNCE_MS = 16
# Longest displayed side of a triplanar slice, in pixels. Slices are
# downsampled to this before windowing, bounding the per-frame cost at
# roughly 15-20 ms. Coordinates stay in voxel units via the imshow extent,
# and PNG export still uses full resolution.
MAX_TRI_DISPLAY_PX = 640
