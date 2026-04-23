"""Application-wide constants: theme, fonts, presets, limits.

Keep this file free of runtime logic. Anything numeric or stylistic that
the rest of the codebase needs should live here so it can be tuned in one
place.
"""

# ── THEME ─────────────────────────────────────────────────────────────────────
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

# Per-axis slider / crosshair colors. Used consistently by the tri-planar
# crosshairs and the 3-D cube-axes labels so a line's colour always maps
# back to the same slider.
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

# ── COLORMAPS ─────────────────────────────────────────────────────────────────
CMAPS   = ["gray", "hot", "jet", "viridis", "bone", "plasma", "inferno", "turbo"]
E_CMAPS = ["viridis", "plasma", "inferno", "hot", "jet", "turbo", "RdYlGn"]

# ── CT windowing presets  (Window Width, Window Center) ───────────────────────
CT_PRESETS = {
    "-- preset --": None,
    "AM Concrete" : (200,  100),
    "Bone"        : (2000, 400),
    "Soft Tissue" : (400,  40),
    "Lung"        : (1500, -600),
    "Brain"       : (80,   40),
}

# ── Performance limits ────────────────────────────────────────────────────────
# Max voxels plotted in 3-D scatter (legacy — the viewer uses marching cubes).
MAX_3D_VOXELS   = 60_000
# Max voxels fed to np.histogram — anything bigger is subsampled.
MAX_HIST_VOXELS = 10_000_000
# Stats panel subsample threshold (for live Min/Max/Mean/p1/p99 display).
STATS_SUBSAMPLE_VOXELS = 5_000_000
# Triplanar slider debounce in milliseconds (caps matplotlib redraws).
SLIDER_DEBOUNCE_MS = 16

# ── Material presets  — HU-to-Young's-modulus mappings ───────────────────────
# params for 'linear':   {'a': slope,  'b': intercept}     E [MPa] = a*HU + b
# params for 'power':    {'a': scale,  'b': exponent}      E [MPa] = a * HU^b
# params for 'bilinear': {'hu_thresh', 'E_void', 'E_solid'}
# params for 'table':    {'hu': [...], 'E': [...]}
#
# References:
#   Concrete (AM):  du Plessis et al. 2016; Garboczi & Berryman 2001
#   Limestone:      Landis & Keane 2010
#   Aluminium:      empirical density–E for Al alloys
#   Steel:          empirical
MATERIAL_PRESETS = {
    "AM Concrete (default)": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "bilinear",
        "params": {"hu_thresh": -200, "E_void": 0.001, "E_solid": 30_000.0},
        "notes": "Additively manufactured concrete. E_solid ≈ 30 GPa (paste). "
                 "Refs: Garboczi & Berryman 2001, du Plessis 2016.",
    },
    "AM Concrete (power-law)": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "power",
        "params": {"a": 0.09, "b": 1.92, "hu_min_clamp": 1.0},
        "notes": "Power-law E = 0.09·HU^1.92  [MPa]. Calibrate a,b to your mix.",
    },
    "Limestone / Rock": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "linear",
        "params": {"a": 20.0, "b": 5_000.0},
        "notes": "E [MPa] = 20·HU + 5000. Landis & Keane 2010 approximate.",
    },
    "Aluminium Alloy": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "linear",
        "params": {"a": 0.045, "b": 69_000.0},
        "notes": "E [MPa] ≈ 69 GPa base, small HU correction for porosity.",
    },
    "Steel": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "bilinear",
        "params": {"hu_thresh": -300, "E_void": 0.001, "E_solid": 200_000.0},
        "notes": "E_solid ≈ 200 GPa. Void ~ air inclusions.",
    },
    "Custom / Manual": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "linear",
        "params": {"a": 1.0, "b": 0.0},
        "notes": "Edit all parameters manually.",
    },
}
