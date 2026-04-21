"""
NIfTI Tool Suite — v5  "SimReady"
══════════════════════════════════════════════════════════════════════════════
Primary goal: Turn CT scan NIfTI files into simulation-ready material maps.
Designed for industrial/research CT (NOT medical): additively manufactured
concrete, ceramics, metals, composites, and any future material.

KEY NEW FEATURES over v4
─────────────────────────
MATERIAL MAPPING ENGINE
  • Intensity → HU calibration  (two-point linear: air + reference material)
  • HU → Young's modulus mapping with FOUR models:
      1. Linear       E = a·HU + b
      2. Power-law    E = a·HU^b          (popular for trabecular bone / porous)
      3. Bilinear     piecewise (void | matrix)
      4. Custom table user-defined breakpoints (piecewise linear interpolation)
  • Material PRESET library: Concrete (AM), Limestone, Aluminium, Steel, Ceramic
    — extensible via JSON sidecar file
  • Phase segmentation: threshold-based void/aggregate/paste or manual ranges
  • Per-phase statistics: volume fraction, E stats, porosity estimate
  • E-map export to:  NIfTI (.nii.gz), raw float32 binary, CSV summary
  • Generates ct.hpp snippet compatible with AMfracture / FCM pipeline

MATERIAL MAP VISUALIZER (new tab)
  • Tri-planar view of Young's modulus field (separate from intensity viewer)
  • Colorbar with MPa scale
  • Overlay mode: E-map overlaid on intensity with alpha blending
  • Per-slice E histogram

SIMULATION EXPORT TAB (new tab)
  • Export E-map NIfTI for direct FCM/FEM ingestion
  • Export raw binary (float32, C-order) + header JSON
  • Generate ct.hpp code snippet (AMfracture-compatible) in one click
  • Mesh info: bounding box, physical size in mm

ENHANCED HISTOGRAM
  • Optional HU axis  (dual-axis: raw intensity + HU after calibration)
  • Phase-shaded histogram  (void / paste / aggregate highlighted)
  • Percentile cursor on hover

ENHANCED METADATA
  • Physical volume size in mm³  and  cm³
  • Porosity estimate from threshold
  • Voxel count per detected phase
  • Scan quality indicator (SNR estimate)

THEME — unchanged: Mild Orange + White (FDFAF6 / E07B2A)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import gzip
import shutil
import os
import sys
import gc
import time
import json
import struct
from pathlib import Path

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import numpy as np
    import nibabel as nib
    import nibabel.orientations as nio
    import scipy.ndimage as ndimage
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from mpl_toolkits.mplot3d import Axes3D   # noqa
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_3D = True
except ImportError:
    HAS_3D = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =============================================================================
# COLOUR / STYLE CONSTANTS
# =============================================================================
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

FONT_MONO  = ("Consolas", 10)
FONT_UI    = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_HEAD  = ("Segoe UI Semibold", 11)
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_CODE  = ("Consolas", 9)

CMAPS = ["gray", "hot", "jet", "viridis", "bone", "plasma", "inferno", "turbo"]
E_CMAPS = ["viridis", "plasma", "inferno", "hot", "jet", "turbo", "RdYlGn"]

CT_PRESETS = {
    "-- preset --": None,
    "AM Concrete" : (200,  100),
    "Bone"        : (2000, 400),
    "Soft Tissue" : (400,  40),
    "Lung"        : (1500, -600),
    "Brain"       : (80,   40),
}

# Max voxels in 3-D scatter
MAX_3D_VOXELS  = 60_000
MAX_HIST_VOXELS = 10_000_000


# =============================================================================
# MATERIAL PRESETS  — (name, hu_air, hu_ref, E_void, E_solid, model, params)
# =============================================================================
# params for 'linear':  {'a': slope, 'b': intercept}   E [MPa] = a*HU + b
# params for 'power':   {'a': scale, 'b': exponent}    E [MPa] = a * HU^b
# params for 'bilinear':{'hu_thresh': t, 'E_void': ev, 'E_solid': es}
# params for 'table':   {'hu': [...], 'E': [...]}
#
# References:
#   Concrete AM:  approx from du Plessis et al. 2016, Garboczi & Berryman 2001
#   Limestone:    Landis & Keane 2010
#   Aluminium:    empirical density-E for Al alloys
#   Steel:        empirical

MATERIAL_PRESETS = {
    "AM Concrete (default)": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "bilinear",
        "params": {"hu_thresh": -200, "E_void": 0.001, "E_solid": 30_000.0},
        "notes": "Additively manufactured concrete. E_solid ≈ 30 GPa (paste). "
                 "Refs: Garboczi & Berryman 2001, du Plessis 2016."
    },
    "AM Concrete (power-law)": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "power",
        "params": {"a": 0.09, "b": 1.92, "hu_min_clamp": 1.0},
        "notes": "Power-law E = 0.09·HU^1.92  [MPa]. Calibrate a,b to your mix."
    },
    "Limestone / Rock": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "linear",
        "params": {"a": 20.0, "b": 5_000.0},
        "notes": "E [MPa] = 20·HU + 5000. Landis & Keane 2010 approximate."
    },
    "Aluminium Alloy": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "linear",
        "params": {"a": 0.045, "b": 69_000.0},
        "notes": "E [MPa] ≈ 69 GPa base, small HU correction for porosity."
    },
    "Steel": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "bilinear",
        "params": {"hu_thresh": -300, "E_void": 0.001, "E_solid": 200_000.0},
        "notes": "E_solid ≈ 200 GPa. Void ~ air inclusions."
    },
    "Custom / Manual": {
        "hu_air": -1000, "hu_ref": 0,
        "void_thresh": -500,
        "model": "linear",
        "params": {"a": 1.0, "b": 0.0},
        "notes": "Edit all parameters manually."
    },
}


# =============================================================================
# BACKEND — CORE FUNCTIONS
# =============================================================================

def _load(path):
    return nib.load(str(path))

def _raw(img):
    return np.asanyarray(img.dataobj)

def _to_gray(data):
    """Convert raw voxel data to float32 scalar volume."""
    if data.dtype.names and 'R' in data.dtype.names:
        # Structured RGB dtype — average channels (no luma for CT)
        return ((data['R'].astype(np.float32) +
                 data['G'].astype(np.float32) +
                 data['B'].astype(np.float32)) / 3.0)
    elif data.ndim == 4:
        return data[..., :3].mean(axis=-1).astype(np.float32)
    else:
        return data.astype(np.float32)

def _apply_window(gray, ww, wc):
    lo = wc - ww / 2.0; hi = wc + ww / 2.0
    clipped = np.clip(gray, lo, hi)
    if hi == lo:
        return np.zeros_like(clipped)
    return (clipped - lo) / (hi - lo)

def _get_axis_labels(affine):
    try:
        codes = nio.aff2axcodes(affine)
        flips = {'R':'L','L':'R','A':'P','P':'A','S':'I','I':'S'}
        return {
            'x': (codes[0], flips.get(codes[0], '?')),
            'y': (codes[1], flips.get(codes[1], '?')),
            'z': (codes[2], flips.get(codes[2], '?')),
        }
    except Exception:
        return {'x': ('X','-'), 'y': ('Y','-'), 'z': ('Z','-')}

def _process_rss_mb():
    if not HAS_PSUTIL:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        return None


# ── HU calibration ────────────────────────────────────────────────────────────

def calibrate_to_hu(gray, hu_air_intensity, hu_ref_intensity,
                     hu_air_value=-1000.0, hu_ref_value=0.0):
    """
    Two-point linear calibration:  HU = m * raw_intensity + c
    hu_air_intensity : raw voxel value that corresponds to air
    hu_ref_intensity : raw voxel value that corresponds to hu_ref_value (water=0, or known material)
    """
    if hu_ref_intensity == hu_air_intensity:
        raise ValueError("Air and reference intensities must differ.")
    m = (hu_ref_value - hu_air_value) / (hu_ref_intensity - hu_air_intensity)
    c = hu_air_value - m * hu_air_intensity
    return gray.astype(np.float32) * m + c, m, c


# ── Young's Modulus mappings ──────────────────────────────────────────────────

def hu_to_E_linear(hu_vol, a, b, void_thresh=None, E_void=0.001):
    """E [MPa] = a*HU + b  (clamp to E_void below threshold)"""
    E = a * hu_vol + b
    if void_thresh is not None:
        E = np.where(hu_vol < void_thresh, E_void, E)
    return np.clip(E, E_void, None).astype(np.float32)

def hu_to_E_power(hu_vol, a, b, hu_min_clamp=1.0, void_thresh=None, E_void=0.001):
    """E [MPa] = a * max(HU, hu_min_clamp)^b"""
    hu_safe = np.clip(hu_vol, hu_min_clamp, None)
    E = a * np.power(hu_safe, b)
    if void_thresh is not None:
        E = np.where(hu_vol < void_thresh, E_void, E)
    return np.clip(E, E_void, None).astype(np.float32)

def hu_to_E_bilinear(hu_vol, hu_thresh, E_void, E_solid):
    """Piecewise: below hu_thresh -> E_void, else -> E_solid"""
    return np.where(hu_vol < hu_thresh,
                    np.float32(E_void),
                    np.float32(E_solid)).astype(np.float32)

def hu_to_E_table(hu_vol, hu_breakpoints, E_breakpoints):
    """Piecewise linear interpolation from user table."""
    hu_arr = np.array(hu_breakpoints, dtype=np.float64)
    E_arr  = np.array(E_breakpoints,  dtype=np.float64)
    return np.interp(hu_vol.ravel(), hu_arr, E_arr).reshape(hu_vol.shape).astype(np.float32)


def compute_E_map(hu_vol, model, params, void_thresh, E_void_default=0.001):
    """Dispatch to the correct mapping function."""
    if model == "linear":
        return hu_to_E_linear(hu_vol, params['a'], params['b'],
                               void_thresh, params.get('E_void', E_void_default))
    elif model == "power":
        return hu_to_E_power(hu_vol, params['a'], params['b'],
                              params.get('hu_min_clamp', 1.0),
                              void_thresh, params.get('E_void', E_void_default))
    elif model == "bilinear":
        return hu_to_E_bilinear(hu_vol, params['hu_thresh'],
                                 params['E_void'], params['E_solid'])
    elif model == "table":
        return hu_to_E_table(hu_vol, params['hu'], params['E'])
    else:
        raise ValueError(f"Unknown model: {model}")


# ── Phase segmentation ────────────────────────────────────────────────────────

def segment_phases(hu_vol, void_thresh, solid_thresh=None):
    """
    Returns label volume: 0=void, 1=paste/matrix, 2=aggregate (if solid_thresh set)
    """
    labels = np.zeros(hu_vol.shape, dtype=np.uint8)
    labels[hu_vol >= void_thresh] = 1
    if solid_thresh is not None:
        labels[hu_vol >= solid_thresh] = 2
    return labels


def phase_statistics(hu_vol, E_vol, labels):
    """Return per-phase dict with volume fraction, E stats, voxel count."""
    total = hu_vol.size
    stats = {}
    phase_names = {0: "Void/Air", 1: "Matrix/Paste", 2: "Aggregate/Solid"}
    for pid, name in phase_names.items():
        mask = labels == pid
        count = int(mask.sum())
        if count == 0:
            continue
        E_vals = E_vol[mask]
        hu_vals = hu_vol[mask]
        stats[name] = {
            "voxels": count,
            "vol_frac": count / total,
            "HU_mean": float(hu_vals.mean()),
            "HU_std":  float(hu_vals.std()),
            "E_mean_MPa": float(E_vals.mean()),
            "E_min_MPa":  float(E_vals.min()),
            "E_max_MPa":  float(E_vals.max()),
            "E_std_MPa":  float(E_vals.std()),
        }
    porosity = stats.get("Void/Air", {}).get("vol_frac", 0.0)
    return stats, porosity


# ── Metadata ──────────────────────────────────────────────────────────────────

def read_metadata(img, gray=None, hu_vol=None, labels=None):
    from nibabel.orientations import aff2axcodes
    affine  = img.affine
    header  = img.header
    orient  = "".join(aff2axcodes(affine))
    spacing = header.get_zooms()
    m       = affine[:3, :3]
    scale   = np.sqrt(np.sum(m**2, axis=0))

    vox_mm3 = float(np.prod(scale[:3]))
    total_vox = int(np.prod(img.shape[:3]))
    phys_vol_mm3 = vox_mm3 * total_vox
    phys_vol_cm3 = phys_vol_mm3 / 1000.0

    phys_size_mm = tuple(round(float(img.shape[i]) * float(scale[i]), 2)
                         for i in range(min(3, len(img.shape))))

    rot   = m / scale
    pitch = np.degrees(np.arctan2(rot[2,1], rot[2,2]))
    roll  = np.degrees(np.arctan2(-rot[2,0], np.sqrt(rot[2,1]**2 + rot[2,2]**2)))
    yaw   = np.degrees(np.arctan2(rot[1,0], rot[0,0]))

    lines = [
        "══ VOLUME GEOMETRY ══════════════════════════════════",
        f"  Shape            : {img.shape}",
        f"  Dtype            : {img.get_data_dtype()}",
        f"  Orientation      : {orient}",
        f"  Voxel spacing    : {tuple(round(float(s),4) for s in spacing)} mm",
        f"  Scale factors    : {tuple(round(float(s),4) for s in scale)}",
        f"  Physical size    : {phys_size_mm} mm",
        f"  Voxel volume     : {vox_mm3:.4f} mm³",
        f"  Total volume     : {phys_vol_mm3:.2f} mm³  ({phys_vol_cm3:.4f} cm³)",
        f"  Origin (world)   : {tuple(round(float(v),3) for v in affine[:3,3])}",
        f"  Euler angles     : Pitch={pitch:.2f}°  Roll={roll:.2f}°  Yaw={yaw:.2f}°",
        "",
        "══ INTENSITY STATISTICS ════════════════════════════",
    ]

    if gray is not None:
        flat = gray.ravel()
        lines += [
            f"  Min              : {float(flat.min()):.2f}",
            f"  Max              : {float(flat.max()):.2f}",
            f"  Mean             : {float(flat.mean()):.4f}",
            f"  Std              : {float(flat.std()):.4f}",
            f"  p1 / p99         : {float(np.percentile(flat, 1)):.2f} / {float(np.percentile(flat, 99)):.2f}",
        ]
        # Simple SNR estimate: mean / std of a central ROI
        try:
            cx, cy, cz = [s//2 for s in gray.shape[:3]]
            r = max(3, min(20, min(gray.shape[:3])//8))
            roi = gray[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r]
            snr = float(roi.mean() / (roi.std() + 1e-9))
            lines.append(f"  SNR (central ROI): {snr:.1f}  (higher = less noise)")
        except Exception:
            pass

    if hu_vol is not None:
        lines += [
            "",
            "══ HU CALIBRATION ══════════════════════════════════",
            f"  HU range         : {float(hu_vol.min()):.1f}  to  {float(hu_vol.max()):.1f}",
            f"  HU mean          : {float(hu_vol.mean()):.2f}",
        ]

    if labels is not None:
        lines += [
            "",
            "══ PHASE SEGMENTATION ══════════════════════════════",
        ]
        total = labels.size
        for pid, name in [(0,"Void/Air"), (1,"Matrix/Paste"), (2,"Aggregate")]:
            count = int((labels == pid).sum())
            if count:
                lines.append(f"  {name:18s}: {count:>9d} vox  ({100*count/total:.2f}%)")

    lines += [
        "",
        "══ AFFINE MATRIX ═══════════════════════════════════",
    ]
    for row in affine:
        lines.append("  " + "  ".join(f"{v:10.4f}" for v in row))
    lines += [
        "",
        "══ NIFTI HEADER ════════════════════════════════════",
    ]
    for k in ('sizeof_hdr','dim_info','dim','pixdim','vox_offset',
              'scl_slope','scl_inter','xyzt_units','qform_code','sform_code'):
        try:
            lines.append(f"  {k:20s}: {header[k]}")
        except Exception:
            pass

    return "\n".join(lines)


# ── Histogram ─────────────────────────────────────────────────────────────────

def compute_histogram(gray, n_bins=256):
    flat = gray.ravel()
    if flat.size > MAX_HIST_VOXELS:
        step = flat.size // MAX_HIST_VOXELS
        flat = flat[::step]
    mn, mx = float(flat.min()), float(flat.max())
    counts, edges = np.histogram(flat, bins=n_bins, range=(mn, mx))
    nz   = gray.shape[2]
    zidx = list(range(0, nz, max(1, nz // 80)))
    means = [float(gray[:, :, z].mean()) for z in zidx]
    return zidx, means, counts, edges, mn, mx


# ── Reorientation ─────────────────────────────────────────────────────────────

def run_reorientation(img, target_orientation):
    target_orientation = target_orientation.upper()
    cur  = nio.io_orientation(img.affine)
    tgt  = nio.axcodes2ornt(tuple(target_orientation))
    xfm  = nio.ornt_transform(cur, tgt)
    reor = img.as_reoriented(xfm)
    aff  = reor.affine
    hdr  = reor.header.copy()
    q_code = int(img.header.get_qform(coded=True)[1])
    s_code = int(img.header.get_sform(coded=True)[1])
    hdr.set_qform(aff, code=q_code)
    hdr.set_sform(aff, code=s_code)
    return nib.Nifti1Image(np.asanyarray(reor.dataobj), aff, hdr)


# ── Angle rotation ────────────────────────────────────────────────────────────

def _rot_matrix(axis, angle_deg):
    a = np.deg2rad(angle_deg); c, s = np.cos(a), np.sin(a)
    if axis == 'x': return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == 'y': return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def run_angle_rotation(img, axis, angle_deg):
    data = _raw(img)
    axis_plane = {'x':(1,2), 'y':(0,2), 'z':(0,1)}
    kw = dict(angle=angle_deg, axes=axis_plane[axis],
              reshape=False, order=3, mode='constant', cval=0.0)
    if data.dtype.names:
        channels = {}
        for ch in data.dtype.names:
            rot = ndimage.rotate(data[ch].astype(np.float32), **kw)
            channels[ch] = np.clip(rot, 0, 255).astype(data.dtype[ch])
        rot_data = np.zeros(channels[data.dtype.names[0]].shape, dtype=data.dtype)
        for ch in data.dtype.names:
            rot_data[ch] = channels[ch]
    else:
        rot_data = ndimage.rotate(data, **kw)
    center  = np.array(data.shape[:3]) / 2.0
    R       = _rot_matrix(axis, angle_deg)
    m_      = img.affine[:3,:3]
    cw      = m_ @ center + img.affine[:3,3]
    new_m   = R @ m_
    new_t   = cw - new_m @ center
    new_aff = img.affine.copy()
    new_aff[:3,:3] = new_m; new_aff[:3,3] = new_t
    hdr = img.header.copy(); hdr.set_data_shape(rot_data.shape)
    q_code = int(img.header.get_qform(coded=True)[1])
    s_code = int(img.header.get_sform(coded=True)[1])
    hdr.set_qform(new_aff, code=q_code)
    hdr.set_sform(new_aff, code=s_code)
    return nib.Nifti1Image(rot_data, new_aff, hdr)


# ── Cropper ───────────────────────────────────────────────────────────────────

def run_cropper(img, x_range, y_range, z_range):
    shape = img.shape
    for (s, e), dim, ax in zip([x_range, y_range, z_range], shape, 'XYZ'):
        if s < 0 or e > dim or s >= e:
            raise ValueError(f"Invalid {ax} range [{s}:{e}] for size {dim}")
    x0,x1 = x_range; y0,y1 = y_range; z0,z1 = z_range
    cropped = np.asanyarray(img.dataobj[x0:x1, y0:y1, z0:z1])
    off     = np.array([x0, y0, z0])
    new_t   = img.affine[:3,:3] @ off + img.affine[:3,3]
    new_aff = img.affine.copy(); new_aff[:3,3] = new_t
    hdr     = img.header.copy(); hdr.set_data_shape(cropped.shape)
    return nib.Nifti1Image(cropped, new_aff, hdr)


# ── Gunzip ────────────────────────────────────────────────────────────────────

def run_gunzip(gz_path):
    gz_path  = Path(gz_path)
    if gz_path.suffix != '.gz':
        raise ValueError("File does not end in .gz")
    out_path = gz_path.with_suffix('')
    with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


# ── ct.hpp snippet generator ──────────────────────────────────────────────────

def generate_ct_hpp_snippet(gray, hu_cal, E_map, img, model, params, void_thresh,
                             hu_air_int, hu_ref_int):
    """Generate a ct.hpp-compatible C++ snippet for FCM pipeline."""
    spacing = img.header.get_zooms()
    vox_str = ", ".join(f"{float(s):.4f}" for s in spacing[:3])
    shape   = img.shape

    if model == "linear":
        map_body = (
            f"    // Linear: E [MPa] = {params['a']} * HU + {params['b']}\n"
            f"    double hu = pixel_intensity * {hu_cal['m']:.6f} + {hu_cal['c']:.6f};\n"
            f"    if (hu < {void_thresh:.1f}) return {params.get('E_void', 0.001):.4f};\n"
            f"    return std::max({params.get('E_void',0.001):.4f}, {params['a']} * hu + {params['b']});\n"
        )
    elif model == "power":
        map_body = (
            f"    // Power-law: E [MPa] = {params['a']} * HU^{params['b']}\n"
            f"    double hu = pixel_intensity * {hu_cal['m']:.6f} + {hu_cal['c']:.6f};\n"
            f"    if (hu < {void_thresh:.1f}) return {params.get('E_void',0.001):.4f};\n"
            f"    double hu_clamp = std::max(hu, {params.get('hu_min_clamp',1.0):.2f});\n"
            f"    return {params['a']} * std::pow(hu_clamp, {params['b']});\n"
        )
    elif model == "bilinear":
        map_body = (
            f"    // Bilinear: void < {params['hu_thresh']} HU\n"
            f"    double hu = pixel_intensity * {hu_cal['m']:.6f} + {hu_cal['c']:.6f};\n"
            f"    if (hu < {params['hu_thresh']:.1f}) return {params['E_void']:.4f};\n"
            f"    return {params['E_solid']:.4f};\n"
        )
    else:
        map_body = "    // Custom table mapping — implement interpolation\n    return 1.0;\n"

    snippet = f"""// ── ct.hpp snippet generated by NIfTI Tool Suite v5 ────────────────────────
// Material model : {model}
// Void threshold : {void_thresh:.1f} HU
// HU calibration : m = {hu_cal['m']:.6f},  c = {hu_cal['c']:.6f}
//   (HU = m * raw_intensity + c)
// Voxel spacing  : {vox_str} mm
// Volume shape   : {shape[0]} x {shape[1]} x {shape[2]}
// E units        : MPa

#include <cmath>
#include <algorithm>

// Call this for each voxel's raw pixel intensity.
// Returns Young's modulus in MPa.
inline double youngsModulusFromPixel(double pixel_intensity) {{
{map_body}}}

// Convenience: apply calibration only
inline double pixelToHU(double pixel_intensity) {{
    return pixel_intensity * {hu_cal['m']:.6f} + {hu_cal['c']:.6f};
}}

// E stats from last run:
//   E min  = {float(E_map.min()):.2f} MPa
//   E max  = {float(E_map.max()):.2f} MPa
//   E mean = {float(E_map.mean()):.2f} MPa
"""
    return snippet


# =============================================================================
# REUSABLE UI WIDGETS
# =============================================================================

def styled_btn(parent, text, cmd, accent=False, danger=False, small=False,
               teal=False, **kw):
    if teal:
        bg = TEAL; fg = "#FFFFFF"
    elif accent:
        bg = ACCENT; fg = "#FFFFFF"
    elif danger:
        bg = ERR; fg = "#FFFFFF"
    else:
        bg = BTN_BG; fg = TEXT
    font = FONT_SMALL if small else FONT_UI
    b    = tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg=fg, activebackground=BTN_HOV,
                     activeforeground=TEXT, relief='flat', bd=0,
                     padx=10, pady=4 if small else 6,
                     cursor='hand2', font=font, **kw)
    hover_bg = "#C96A1A" if accent else (ERR if danger else (TEAL if teal else BTN_HOV))
    b.bind('<Enter>', lambda e: b.config(bg=hover_bg))
    b.bind('<Leave>', lambda e: b.config(bg=bg))
    return b

def styled_entry(parent, textvariable=None, width=12):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg=ENTRY_BG, fg=TEXT, insertbackground=ACCENT,
                    relief='flat', bd=0, highlightthickness=1,
                    highlightbackground=BORDER, highlightcolor=ACCENT,
                    font=FONT_MONO)

def lbl(parent, text, dim=False, head=False, **kw):
    fg   = TEXT_DIM if dim else TEXT
    font = FONT_HEAD if head else FONT_UI
    return tk.Label(parent, text=text, bg=PANEL, fg=fg, font=font, **kw)


class CollapsibleSection(tk.Frame):
    def __init__(self, parent, title, start_open=True, **kw):
        super().__init__(parent, bg=BG, pady=3, **kw)
        self._open  = tk.BooleanVar(value=start_open)
        self._title = title
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill='x')
        self._toggle_btn = tk.Label(
            hdr, text=self._header_text(), bg=BG, fg=ACCENT,
            font=FONT_HEAD, cursor='hand2', anchor='w')
        self._toggle_btn.pack(fill='x')
        self._toggle_btn.bind('<Button-1>', lambda e: self.toggle())
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x', pady=(0, 3))
        self.content = tk.Frame(self, bg=PANEL, bd=0,
                                highlightthickness=1, highlightbackground=BORDER)
        if start_open:
            self.content.pack(fill='x')

    def _header_text(self):
        return f" {'▼' if self._open.get() else '►'} {self._title}"

    def toggle(self):
        self._open.set(not self._open.get())
        self._toggle_btn.config(text=self._header_text())
        if self._open.get():
            self.content.pack(fill='x')
        else:
            self.content.forget()

    def open(self):
        if not self._open.get():
            self.toggle()


class IntRangeRow(tk.Frame):
    def __init__(self, parent, axis_label, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.start_var = tk.StringVar(value="0")
        self.end_var   = tk.StringVar(value="0")
        tk.Label(self, text=axis_label, bg=PANEL, fg=ACCENT,
                 font=("Consolas", 10, "bold"), width=3).pack(side='left', padx=(8,4))
        tk.Label(self, text="start", bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side='left')
        styled_entry(self, textvariable=self.start_var, width=6).pack(side='left', padx=(2,8))
        tk.Label(self, text="end", bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side='left')
        styled_entry(self, textvariable=self.end_var, width=6).pack(side='left', padx=(2,4))


class StageProgressBar(tk.Frame):
    STAGES = [
        ("Opening file",      0.10),
        ("Decoding voxels",   0.40),
        ("Greyscale convert", 0.65),
        ("Updating sliders",  0.80),
        ("Computing stats",   0.92),
        ("Ready",             1.00),
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=PANEL2, **kw)
        self._stage_idx = -1
        self._mode      = 'hidden'
        self._spin_pos  = 0.0
        self._label = tk.Label(self, text="", bg=PANEL2, fg=TEXT_DIM,
                               font=FONT_SMALL, width=18, anchor='w')
        self._label.pack(side='left', padx=(6, 2))
        self._canvas = tk.Canvas(self, height=14, width=160, bg=PANEL2,
                                 bd=0, highlightthickness=0)
        self._canvas.pack(side='left', padx=(0, 6))
        self._canvas.bind('<Configure>', self._redraw)

    def begin_staged(self):
        self._stage_idx = 0; self._mode = 'stage'
        self.pack(side='right', padx=4); self._draw_stage()

    def advance(self):
        if self._stage_idx < len(self.STAGES) - 1:
            self._stage_idx += 1
        self._draw_stage()

    def done(self):
        self._stage_idx = len(self.STAGES) - 1
        self._draw_stage()
        self.after(900, self._hide)

    def start(self, interval=12):
        self._mode = 'spinner'; self._spin_pos = 0.0
        self.pack(side='right', padx=4); self._tick_spinner()

    def stop(self):
        self._hide()

    def _hide(self):
        self._mode = 'hidden'; self._label.config(text=""); self.pack_forget()

    def _draw_stage(self):
        if self._stage_idx < 0 or self._stage_idx >= len(self.STAGES):
            return
        name, frac = self.STAGES[self._stage_idx]
        self._label.config(text=name); self._draw_bar(frac)

    def _draw_bar(self, frac):
        c = self._canvas; w = c.winfo_width() or 160; h = 14
        c.delete('all'); r = 5
        c.create_rectangle(r, 2, w-r, h-2, fill=BORDER, outline='', width=0)
        fill_w = max(0, int(frac * w))
        if fill_w > r:
            c.create_rectangle(r, 2, fill_w, h-2, fill=ACCENT, outline='', width=0)
        c.create_text(w//2, h//2, text=f"{int(frac*100)}%",
                      fill=TEXT if frac > 0.5 else TEXT_DIM, font=FONT_SMALL)

    def _redraw(self, event=None):
        if self._mode == 'stage':   self._draw_stage()
        elif self._mode == 'spinner': self._draw_bar(self._spin_pos)

    def _tick_spinner(self):
        if self._mode != 'spinner': return
        self._spin_pos = (self._spin_pos + 0.05) % 1.05
        self._draw_bar(min(self._spin_pos, 1.0))
        self.after(60, self._tick_spinner)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class NiftiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NIfTI Tool Suite  v6  —  SimReady")
        self.configure(bg=BG)
        self.geometry("1600x1000")
        self.minsize(1200, 750)

        # Core state
        self._img         = None
        self._gray        = None
        self._path        = None
        self._busy        = False
        self._axis_labels = {'x':('X','-'),'y':('Y','-'),'z':('Z','-')}
        self._ww          = None
        self._wc          = None

        # Material mapping state
        self._hu_vol      = None   # calibrated HU volume
        self._E_map       = None   # Young's modulus volume  [MPa]
        self._labels      = None   # phase label volume
        self._hu_cal      = {}     # {'m': slope, 'c': intercept}
        self._E_stats     = {}     # phase statistics dict
        self._porosity    = 0.0

        self._build_ui()
        self._check_deps()
        self._setup_dnd()
        self.bind_all('<Control-o>', lambda e: self._open_file())
        self._start_ram_monitor()

    # ── drag-and-drop ─────────────────────────────────────────────────────────

    def _setup_dnd(self):
        try:
            import tkinterdnd2
            self.drop_target_register('DND_Files')
            self.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        path = event.data.strip().strip('{}')
        if path:
            self._load_path(path)

    # ── dependency check ──────────────────────────────────────────────────────

    def _check_deps(self):
        missing = []
        if not HAS_NIBABEL: missing.append("nibabel / numpy / scipy")
        if not HAS_MPL:     missing.append("matplotlib")
        if not HAS_PSUTIL:  missing.append("psutil  (optional)")
        if not HAS_3D:      missing.append("mpl_toolkits  (optional)")
        if not HAS_SKIMAGE: missing.append("scikit-image  (optional — for 3-D surface)")
        if missing:
            self._append_log(
                f"  Missing packages: {', '.join(missing)}\n"
                f"  Install: pip install nibabel numpy scipy matplotlib psutil scikit-image\n", 'warn')

    # ── RAM monitor ───────────────────────────────────────────────────────────

    def _start_ram_monitor(self):
        self._update_ram()

    def _update_ram(self):
        mb = _process_rss_mb()
        if mb is not None:
            self._ram_var.set(f"RAM  {mb:.0f} MiB")
        else:
            self._ram_var.set("RAM  — (pip install psutil)")
        self.after(2000, self._update_ram)

    # =========================================================================
    # UI SKELETON
    # =========================================================================

    def _build_ui(self):
        # ── top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG, pady=8)
        top.pack(fill='x', padx=14)

        tk.Label(top, text="NIfTI Tool Suite", bg=BG, fg=TEXT,
                 font=FONT_TITLE).pack(side='left')
        tk.Label(top, text="v6  SimReady  |  Ctrl+O to open",
                 bg=BG, fg=TEXT_DIM, font=FONT_SMALL).pack(side='left', padx=12, pady=2)

        btn_row = tk.Frame(top, bg=BG)
        btn_row.pack(side='right')
        styled_btn(btn_row, "Open NIfTI...", self._open_file, accent=True).pack(side='left', padx=4)
        styled_btn(btn_row, "Gunzip .gz...", self._do_gunzip).pack(side='left')

        # ── file info bar ─────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x')
        self._info_var = tk.StringVar(value="No file loaded  ─  open a .nii or .nii.gz file")
        inf = tk.Frame(self, bg=PANEL2, pady=5)
        inf.pack(fill='x')
        tk.Label(inf, textvariable=self._info_var, bg=PANEL2, fg=TEXT_DIM,
                 font=FONT_MONO, anchor='w').pack(side='left', padx=10)

        # ── main paned ────────────────────────────────────────────────────────
        paned = tk.PanedWindow(self, orient='horizontal', bg=BG,
                               sashwidth=6, sashrelief='flat', sashpad=2)
        paned.pack(fill='both', expand=True)

        ctrl_outer = tk.Frame(paned, bg=BG, width=380)
        paned.add(ctrl_outer, minsize=320)

        ctrl_canvas = tk.Canvas(ctrl_outer, bg=BG, bd=0, highlightthickness=0)
        ctrl_scroll = ttk.Scrollbar(ctrl_outer, orient='vertical',
                                    command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_scroll.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        self._ctrl_frame = tk.Frame(ctrl_canvas, bg=BG)
        self._ctrl_win_id = ctrl_canvas.create_window(
            (0,0), window=self._ctrl_frame, anchor='nw')
        self._ctrl_frame.bind('<Configure>',
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind('<Configure>',
            lambda e: ctrl_canvas.itemconfig(self._ctrl_win_id, width=e.width))
        ctrl_canvas.bind('<MouseWheel>',
            lambda e: ctrl_canvas.yview_scroll(-1*(e.delta//120), 'units'))
        self._ctrl_canvas = ctrl_canvas

        self._build_controls(self._ctrl_frame)

        # ── right notebook ────────────────────────────────────────────────────
        right = tk.Frame(paned, bg=BG)
        paned.add(right, minsize=700)

        style = ttk.Style()
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab',
                        background=PANEL2, foreground=TEXT_DIM,
                        padding=[10, 4], font=FONT_SMALL)
        style.map('TNotebook.Tab',
                  background=[('selected', PANEL), ('active', BORDER)],
                  foreground=[('selected', ACCENT)])

        nb = ttk.Notebook(right)
        nb.pack(fill='both', expand=True)

        self._triplanar_tab  = tk.Frame(nb, bg=BG)
        self._viewer3d_tab   = tk.Frame(nb, bg=BG)
        self._histogram_tab  = tk.Frame(nb, bg=BG)
        self._emap_tab       = tk.Frame(nb, bg=BG)
        self._export_tab     = tk.Frame(nb, bg=BG)
        self._log_tab        = tk.Frame(nb, bg=BG)

        nb.add(self._triplanar_tab,  text="  Tri-Planar  ")
        nb.add(self._viewer3d_tab,   text="  3-D View  ")
        nb.add(self._histogram_tab,  text="  Histogram  ")
        nb.add(self._emap_tab,       text="  E-Map Viewer  ")
        nb.add(self._export_tab,     text="  Sim Export  ")
        nb.add(self._log_tab,        text="  Log  ")

        self._nb = nb

        self._build_triplanar(self._triplanar_tab)
        self._build_3d_view(self._viewer3d_tab)
        self._build_histogram_tab(self._histogram_tab)
        self._build_emap_tab(self._emap_tab)
        self._build_export_tab(self._export_tab)
        self._build_log(self._log_tab)

        # ── status bar ────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x', side='bottom')
        status_bar = tk.Frame(self, bg=PANEL2, pady=4)
        status_bar.pack(fill='x', side='bottom')
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=PANEL2, fg=TEXT_DIM, font=FONT_SMALL, anchor='w').pack(
                     side='left', padx=10)
        self._probe_var = tk.StringVar(value="")
        tk.Label(status_bar, textvariable=self._probe_var,
                 bg=PANEL2, fg=ACCENT, font=FONT_MONO, anchor='e').pack(
                     side='left', padx=14)
        self._ram_var = tk.StringVar(value="RAM  --")
        tk.Label(status_bar, textvariable=self._ram_var,
                 bg=PANEL2, fg=TEXT_DIM, font=FONT_SMALL, anchor='e').pack(
                     side='right', padx=16)
        tk.Frame(status_bar, bg=BORDER, width=1).pack(side='right', fill='y', pady=2)
        self._prog = StageProgressBar(status_bar)

    # =========================================================================
    # CONTROLS PANEL
    # =========================================================================

    def _build_controls(self, parent):

        # ── File info ─────────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "📋 Metadata / File Info")
        sec.pack(fill='x', padx=10, pady=3)
        styled_btn(sec.content, "Read Full Metadata", self._do_metadata,
                   small=True).pack(padx=10, pady=8, anchor='w')

        # ── Intensity viewer ──────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "🎨 Viewer Controls")
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)

        for label, var_name, default, values in [
            ("Colormap", "_cmap_var", "gray", CMAPS),
            ("CT Preset", "_preset_var", "-- preset --", list(CT_PRESETS.keys())),
        ]:
            row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=10, anchor='w').pack(side='left')
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            cb = ttk.Combobox(row, textvariable=var, values=values,
                              state='readonly', width=14)
            cb.pack(side='left')
            if var_name == "_cmap_var":
                cb.bind('<<ComboboxSelected>>', lambda e: self._refresh_triplanar())
            else:
                cb.bind('<<ComboboxSelected>>', self._apply_preset)

        for label, var_name in [("Window W", "_ww_var"), ("Window C", "_wc_var")]:
            row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=10, anchor='w').pack(side='left')
            var = tk.StringVar(value="auto")
            setattr(self, var_name, var)
            styled_entry(inner if False else row, textvariable=var, width=8).pack(side='left')

        btn_row = tk.Frame(inner, bg=PANEL); btn_row.pack(fill='x', pady=4)
        styled_btn(btn_row, "Apply W/C", self._apply_window_manual,
                   small=True).pack(side='left', padx=(0,4))
        styled_btn(btn_row, "Reset Auto", self._reset_window,
                   small=True).pack(side='left')

        # ── Intensity statistics ──────────────────────────────────────────────
        sec = CollapsibleSection(parent, "📊 Intensity Statistics")
        sec.pack(fill='x', padx=10, pady=3)
        inner2 = tk.Frame(sec.content, bg=PANEL); inner2.pack(fill='x', padx=10, pady=4)
        styled_btn(inner2, "Compute Histogram", self._do_histogram,
                   small=True).pack(anchor='w', pady=(0,4))
        self._stats_frame = tk.Frame(inner2, bg=PANEL)
        self._stats_frame.pack(fill='x')
        self._stat_labels = {}
        for key in ("Min", "Max", "Mean", "Std", "p1", "p99"):
            row = tk.Frame(self._stats_frame, bg=PANEL); row.pack(fill='x', pady=1)
            tk.Label(row, text=f"{key:>4}:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=5, anchor='e').pack(side='left')
            lv = tk.StringVar(value="--")
            tk.Label(row, textvariable=lv, bg=PANEL, fg=ACCENT,
                     font=FONT_MONO, anchor='w').pack(side='left', padx=4)
            self._stat_labels[key] = lv

        # ── HU Calibration ────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "🔬 HU Calibration")
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)

        tk.Label(inner, text="Two-point linear: HU = m·raw + c",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w', pady=(0,6))

        def _cal_row(parent, label, var_name, default, tip=""):
            r = tk.Frame(parent, bg=PANEL); r.pack(fill='x', pady=2)
            tk.Label(r, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=16, anchor='w').pack(side='left')
            v = tk.StringVar(value=str(default))
            setattr(self, var_name, v)
            styled_entry(r, textvariable=v, width=9).pack(side='left')
            if tip:
                tk.Label(r, text=tip, bg=PANEL, fg=TEXT_DIM,
                         font=FONT_SMALL).pack(side='left', padx=4)

        _cal_row(inner, "Air intensity",  "_cal_air_int",  "0",    "raw px = air")
        _cal_row(inner, "Ref intensity",  "_cal_ref_int",  "199",  "raw px = ref material")
        _cal_row(inner, "Air HU",         "_cal_air_hu",   "-1000","HU of air")
        _cal_row(inner, "Ref HU",         "_cal_ref_hu",   "0",    "HU of ref (water=0)")

        tk.Label(inner, text="(Use histogram peak for air; known material for ref)",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL, wraplength=300,
                 justify='left').pack(anchor='w', pady=(4,2))

        self._cal_result_var = tk.StringVar(value="Not calibrated")
        tk.Label(inner, textvariable=self._cal_result_var,
                 bg=PANEL, fg=TEAL, font=FONT_MONO).pack(anchor='w', pady=2)
        styled_btn(inner, "Apply Calibration", self._do_calibrate,
                   teal=True, small=True).pack(anchor='w', pady=4)

        # ── Material mapping ──────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "⚙️  Material Mapping  →  E [MPa]")
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)

        # Preset selector
        pr_row = tk.Frame(inner, bg=PANEL); pr_row.pack(fill='x', pady=2)
        tk.Label(pr_row, text="Preset", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=10, anchor='w').pack(side='left')
        self._mat_preset_var = tk.StringVar(value="AM Concrete (default)")
        pr_cb = ttk.Combobox(pr_row, textvariable=self._mat_preset_var,
                             values=list(MATERIAL_PRESETS.keys()),
                             state='readonly', width=24)
        pr_cb.pack(side='left')
        pr_cb.bind('<<ComboboxSelected>>', self._apply_material_preset)

        # Model selector
        mod_row = tk.Frame(inner, bg=PANEL); mod_row.pack(fill='x', pady=2)
        tk.Label(mod_row, text="Model", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=10, anchor='w').pack(side='left')
        self._model_var = tk.StringVar(value="bilinear")
        for m in ("linear", "power", "bilinear", "table"):
            tk.Radiobutton(mod_row, text=m, variable=self._model_var, value=m,
                           bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
                           activebackground=PANEL, font=FONT_SMALL,
                           command=self._update_model_panel).pack(side='left', padx=3)

        # Dynamic model params frame
        self._model_params_frame = tk.Frame(inner, bg=PANEL)
        self._model_params_frame.pack(fill='x', pady=4)
        self._model_param_vars = {}
        self._update_model_panel()

        # Void threshold
        vt_row = tk.Frame(inner, bg=PANEL); vt_row.pack(fill='x', pady=2)
        tk.Label(vt_row, text="Void thresh", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=12, anchor='w').pack(side='left')
        self._void_thresh_var = tk.StringVar(value="-500")
        styled_entry(vt_row, textvariable=self._void_thresh_var, width=8).pack(side='left')
        tk.Label(vt_row, text="HU", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left', padx=4)

        # Aggregate threshold (for 3-phase)
        ag_row = tk.Frame(inner, bg=PANEL); ag_row.pack(fill='x', pady=2)
        tk.Label(ag_row, text="Aggreg. thresh", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=12, anchor='w').pack(side='left')
        self._agg_thresh_var = tk.StringVar(value="")
        styled_entry(ag_row, textvariable=self._agg_thresh_var, width=8).pack(side='left')
        tk.Label(ag_row, text="HU  (blank = 2-phase)", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left', padx=4)

        # Preset notes
        self._mat_note_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._mat_note_var,
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL,
                 wraplength=300, justify='left').pack(anchor='w', pady=(4,2))

        styled_btn(inner, "▶  Compute E-Map", self._do_compute_emap,
                   accent=True, small=True).pack(anchor='w', pady=6)

        # E stats display
        self._e_stats_frame = tk.Frame(inner, bg=PANEL)
        self._e_stats_frame.pack(fill='x', pady=2)
        self._e_stat_labels = {}
        for key in ("E_min", "E_max", "E_mean", "E_std", "porosity"):
            row = tk.Frame(self._e_stats_frame, bg=PANEL); row.pack(fill='x', pady=1)
            disp = key.replace("_", " ")
            tk.Label(row, text=f"{disp:>10}:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=12, anchor='e').pack(side='left')
            lv = tk.StringVar(value="--")
            tk.Label(row, textvariable=lv, bg=PANEL, fg=TEAL,
                     font=FONT_MONO, anchor='w').pack(side='left', padx=4)
            self._e_stat_labels[key] = lv

        # Load external preset JSON
        styled_btn(inner, "Load Preset JSON...", self._load_preset_json,
                   small=True).pack(anchor='w', pady=(8,0))

        # ── Apply first preset
        self._apply_material_preset()

        # ── Reorientation ─────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "🔄 Reorientation  (no resample)", start_open=False)
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)
        row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
        tk.Label(row, text="Target", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        self._reorient_target = tk.StringVar(value="RAS")
        for preset in ("RAS", "LPS", "LAS"):
            styled_btn(row, preset, lambda p=preset: self._reorient_target.set(p),
                       small=True).pack(side='left', padx=2)
        row2 = tk.Frame(inner, bg=PANEL); row2.pack(fill='x', pady=2)
        tk.Label(row2, text="Custom", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        styled_entry(row2, textvariable=self._reorient_target, width=8).pack(side='left')
        tk.Label(row2, text="e.g. RAS", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left', padx=6)
        self._reorient_save = tk.BooleanVar(value=True)
        tk.Checkbutton(inner, text="Choose save path", variable=self._reorient_save,
                       bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
                       activebackground=PANEL, font=FONT_SMALL).pack(anchor='w')
        styled_btn(sec.content, "Run Reorientation", self._do_reorient,
                   small=True).pack(padx=10, pady=(2,8), anchor='w')

        # ── Angle Rotation ────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "↩  Angle Rotation  (resample)", start_open=False)
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)
        row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
        tk.Label(row, text="Axis", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        self._rot_axis = tk.StringVar(value="z")
        for ax in ('x', 'y', 'z'):
            tk.Radiobutton(row, text=ax.upper(), variable=self._rot_axis, value=ax,
                           bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
                           activebackground=PANEL, font=FONT_UI).pack(side='left', padx=4)
        row2 = tk.Frame(inner, bg=PANEL); row2.pack(fill='x', pady=2)
        tk.Label(row2, text="Angle °", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        self._rot_angle = tk.StringVar(value="45")
        styled_entry(row2, textvariable=self._rot_angle, width=7).pack(side='left', padx=4)
        for deg in (90, 180, 270):
            styled_btn(row2, str(deg), lambda d=deg: self._rot_angle.set(str(d)),
                       small=True).pack(side='left', padx=2)
        self._rot_save = tk.BooleanVar(value=True)
        tk.Checkbutton(inner, text="Choose save path", variable=self._rot_save,
                       bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
                       activebackground=PANEL, font=FONT_SMALL).pack(anchor='w')
        styled_btn(sec.content, "Run Rotation", self._do_rotate,
                   small=True).pack(padx=10, pady=(2,8), anchor='w')

        # ── Crop ─────────────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "✂️  Crop  (ROI selection)", start_open=False)
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)
        self._crop_rows: dict = {}
        for axis in ('X', 'Y', 'Z'):
            r = IntRangeRow(inner, axis); r.pack(fill='x', pady=2)
            self._crop_rows[axis] = r
        btn_row2 = tk.Frame(inner, bg=PANEL); btn_row2.pack(fill='x', pady=4)
        styled_btn(btn_row2, "Auto-fill from shape", self._autofill_crop,
                   small=True).pack(side='left')
        self._crop_save = tk.BooleanVar(value=True)
        tk.Checkbutton(inner, text="Choose save path", variable=self._crop_save,
                       bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
                       activebackground=PANEL, font=FONT_SMALL).pack(anchor='w', pady=2)
        styled_btn(sec.content, "Run Crop", self._do_crop,
                   small=True).pack(padx=10, pady=(2,8), anchor='w')

        tk.Frame(parent, bg=BG, height=20).pack()

    # ── Model params panel ─────────────────────────────────────────────────────

    def _update_model_panel(self):
        for w in self._model_params_frame.winfo_children():
            w.destroy()
        self._model_param_vars.clear()
        model = self._model_var.get()
        f = self._model_params_frame

        def param_row(lbl_text, key, default, unit=""):
            r = tk.Frame(f, bg=PANEL); r.pack(fill='x', pady=2)
            tk.Label(r, text=lbl_text, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=14, anchor='w').pack(side='left')
            v = tk.StringVar(value=str(default))
            self._model_param_vars[key] = v
            styled_entry(r, textvariable=v, width=10).pack(side='left')
            if unit:
                tk.Label(r, text=unit, bg=PANEL, fg=TEXT_DIM,
                         font=FONT_SMALL).pack(side='left', padx=4)

        if model == "linear":
            tk.Label(f, text="E = a·HU + b  [MPa]", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(anchor='w')
            param_row("a  (slope)", "a", "20.0", "MPa/HU")
            param_row("b  (intercept)", "b", "0.0", "MPa")
            param_row("E_void", "E_void", "0.001", "MPa")

        elif model == "power":
            tk.Label(f, text="E = a · HU^b  [MPa]", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(anchor='w')
            param_row("a  (scale)", "a", "0.09", "MPa")
            param_row("b  (exponent)", "b", "1.92")
            param_row("HU min clamp", "hu_min_clamp", "1.0", "HU")
            param_row("E_void", "E_void", "0.001", "MPa")

        elif model == "bilinear":
            tk.Label(f, text="Piecewise: void | solid", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(anchor='w')
            param_row("HU threshold", "hu_thresh", "-200", "HU")
            param_row("E_void", "E_void", "0.001", "MPa")
            param_row("E_solid", "E_solid", "30000.0", "MPa")

        elif model == "table":
            tk.Label(f, text="Piecewise-linear table\nEnter HU values:",
                     bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w')
            self._table_text = scrolledtext.ScrolledText(
                f, bg=ENTRY_BG, fg=TEXT, font=FONT_CODE,
                width=32, height=6, relief='flat')
            self._table_text.pack(fill='x', pady=4)
            self._table_text.insert('end',
                "# HU_value  E_MPa\n"
                "-1000       0.001\n"
                " -500       0.001\n"
                "    0    5000.0\n"
                "  500   25000.0\n"
                " 1000   40000.0\n"
            )
            tk.Label(f, text="(Lines starting with # are comments)",
                     bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w')

    def _get_model_params(self):
        model = self._model_var.get()
        vars_ = self._model_param_vars
        try:
            if model == "linear":
                return {"a": float(vars_["a"].get()),
                        "b": float(vars_["b"].get()),
                        "E_void": float(vars_["E_void"].get())}
            elif model == "power":
                return {"a": float(vars_["a"].get()),
                        "b": float(vars_["b"].get()),
                        "hu_min_clamp": float(vars_["hu_min_clamp"].get()),
                        "E_void": float(vars_["E_void"].get())}
            elif model == "bilinear":
                return {"hu_thresh": float(vars_["hu_thresh"].get()),
                        "E_void": float(vars_["E_void"].get()),
                        "E_solid": float(vars_["E_solid"].get())}
            elif model == "table":
                lines = self._table_text.get('1.0', 'end').strip().splitlines()
                hu_vals = []; E_vals = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    hu_vals.append(float(parts[0]))
                    E_vals.append(float(parts[1]))
                return {"hu": hu_vals, "E": E_vals}
        except Exception as ex:
            raise ValueError(f"Bad parameter input: {ex}")

    def _apply_material_preset(self, event=None):
        name = self._mat_preset_var.get()
        preset = MATERIAL_PRESETS.get(name)
        if not preset:
            return
        self._model_var.set(preset.get("model", "linear"))
        self._void_thresh_var.set(str(preset.get("void_thresh", "-500")))
        self._mat_note_var.set(preset.get("notes", ""))
        self._update_model_panel()
        # Apply param values
        params = preset.get("params", {})
        for k, v in params.items():
            if k in self._model_param_vars:
                self._model_param_vars[k].set(str(v))
        # Apply calibration
        self._cal_air_int.set(str(preset.get("hu_air", "0")))
        self._cal_ref_int.set(str(preset.get("hu_ref", "199")))

    def _load_preset_json(self):
        path = filedialog.askopenfilename(
            title="Load material preset JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            MATERIAL_PRESETS.update(data)
            self._append_log(f"  Loaded {len(data)} preset(s) from {Path(path).name}", 'ok')
        except Exception as ex:
            messagebox.showerror("JSON Error", str(ex))

    # =========================================================================
    # TRI-PLANAR VIEWER
    # =========================================================================

    def _build_triplanar(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="matplotlib not installed",
                     bg=BG, fg=ERR, font=FONT_UI).pack(expand=True)
            return
        self._tri_fig = Figure(facecolor=BG)
        axes = self._tri_fig.subplots(1, 3)
        self._ax_sag, self._ax_cor, self._ax_axi = axes
        for ax in axes:
            ax.set_facecolor(PANEL2); ax.axis('off')
        self._tri_fig.tight_layout(pad=0.4)
        self._tri_canvas = FigureCanvasTkAgg(self._tri_fig, master=parent)
        self._tri_canvas.get_tk_widget().pack(fill='both', expand=True)
        self._tri_canvas.mpl_connect('button_press_event', self._on_tri_click)
        tb_frame = tk.Frame(parent, bg=BG); tb_frame.pack(fill='x')
        NavigationToolbar2Tk(self._tri_canvas, tb_frame)
        slider_bar = tk.Frame(parent, bg=PANEL2, pady=4)
        slider_bar.pack(fill='x')
        self._tri_sliders  = {}
        self._tri_idx_vars = {}
        self._tri_idx      = {'X': 0, 'Y': 0, 'Z': 0}
        axis_info = [('X', "Sagittal (X)"), ('Y', "Coronal  (Y)"), ('Z', "Axial    (Z)")]
        for col, (ax, label) in enumerate(axis_info):
            col_frame = tk.Frame(slider_bar, bg=PANEL2)
            col_frame.grid(row=0, column=col, sticky='ew', padx=8, pady=2)
            slider_bar.columnconfigure(col, weight=1)
            tk.Label(col_frame, text=label, bg=PANEL2, fg=ACCENT,
                     font=FONT_SMALL).pack(anchor='w')
            row_f = tk.Frame(col_frame, bg=PANEL2); row_f.pack(fill='x')
            idx_var = tk.StringVar(value="--"); self._tri_idx_vars[ax] = idx_var
            sl = tk.Scale(row_f, from_=0, to=100, orient='horizontal',
                          bg=PANEL2, fg=TEXT, troughcolor=BORDER,
                          highlightthickness=0, bd=0, showvalue=False,
                          command=lambda val, a=ax: self._on_tri_drag(a, val))
            sl.pack(side='left', fill='x', expand=True)
            tk.Label(row_f, textvariable=idx_var, bg=PANEL2, fg=ACCENT,
                     font=FONT_MONO, width=5).pack(side='left')
            self._tri_sliders[ax] = sl
        export_bar = tk.Frame(parent, bg=BG, pady=2); export_bar.pack(fill='x')
        for axis in ('Z', 'Y', 'X'):
            name = {'Z':'Axial','Y':'Coronal','X':'Sagittal'}[axis]
            styled_btn(export_bar, f"Export {name} PNG",
                       lambda a=axis: self._export_slice(a),
                       small=True).pack(side='left', padx=4)

    def _on_tri_drag(self, axis, val):
        idx = int(float(val)); self._tri_idx[axis] = idx
        self._tri_idx_vars[axis].set(str(idx))
        if self._gray is not None:
            self._refresh_triplanar()

    def _on_tri_click(self, event):
        if self._gray is None or event.inaxes is None: return
        ax_map = {self._ax_sag: 'X', self._ax_cor: 'Y', self._ax_axi: 'Z'}
        if event.inaxes not in ax_map: return
        axis = ax_map[event.inaxes]
        ex, ey = int(event.xdata or 0), int(event.ydata or 0)
        g  = self._gray
        xi, yi, zi = self._tri_idx['X'], self._tri_idx['Y'], self._tri_idx['Z']
        try:
            if axis == 'Z':   ix, iy, iz = ex, ey, zi
            elif axis == 'Y': ix, iy, iz = ex, yi, ey
            else:             ix, iy, iz = xi, ex, ey
            ix = max(0, min(ix, g.shape[0]-1))
            iy = max(0, min(iy, g.shape[1]-1))
            iz = max(0, min(iz, g.shape[2]-1))
            raw_val = float(g[ix, iy, iz])
            probe = f"voxel ({ix},{iy},{iz})  raw={raw_val:.1f}"
            if self._hu_vol is not None:
                hu = float(self._hu_vol[ix, iy, iz])
                probe += f"  HU={hu:.1f}"
            if self._E_map is not None:
                E = float(self._E_map[ix, iy, iz])
                probe += f"  E={E:.1f} MPa"
            self._probe_var.set(probe)
        except Exception:
            pass

    def _refresh_triplanar(self):
        if not HAS_MPL or self._gray is None: return
        g = self._gray; cmap = self._cmap_var.get()
        xi = max(0, min(self._tri_idx['X'], g.shape[0]-1))
        yi = max(0, min(self._tri_idx['Y'], g.shape[1]-1))
        zi = max(0, min(self._tri_idx['Z'], g.shape[2]-1))
        ww, wc = self._ww, self._wc
        if ww is None or wc is None:
            mn, mx = float(g.min()), float(g.max())
            ww = mx - mn; wc = (mn + mx) / 2.0

        def _w(sl):
            return _apply_window(sl.astype(np.float32), ww, wc)

        slices = {'X': g[xi,:,:].T, 'Y': g[:,yi,:].T, 'Z': g[:,:,zi].T}
        panels = [
            (self._ax_sag, 'X', xi, f"Sagittal  X={xi}"),
            (self._ax_cor, 'Y', yi, f"Coronal   Y={yi}"),
            (self._ax_axi, 'Z', zi, f"Axial     Z={zi}"),
        ]
        al = self._axis_labels
        compass = {
            'X': (al['y'][0], al['y'][1], al['z'][0], al['z'][1]),
            'Y': (al['x'][0], al['x'][1], al['z'][0], al['z'][1]),
            'Z': (al['x'][0], al['x'][1], al['y'][0], al['y'][1]),
        }
        # Crosshair positions: for each view, show where the OTHER two slices intersect
        # Sagittal (X slice):  horizontal axis = Y,  vertical axis = Z  → show yi, zi
        # Coronal  (Y slice):  horizontal axis = X,  vertical axis = Z  → show xi, zi
        # Axial    (Z slice):  horizontal axis = X,  vertical axis = Y  → show xi, yi
        crosshair_pos = {
            'X': (yi, zi),   # (hline_pos, vline_pos) in image coords
            'Y': (xi, zi),
            'Z': (xi, yi),
        }
        for ax_obj, axis, idx, title in panels:
            ax_obj.clear()
            sl_win = _w(slices[axis])
            ax_obj.imshow(sl_win, cmap=cmap, origin='lower', vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')
            ax_obj.set_title(title, color=TEXT_DIM, fontsize=8, pad=3)
            ax_obj.axis('off')
            lft, rgt, bot, top_ = compass[axis]
            for txt, kw_ in [(lft, dict(x=0.02, ha='left')),
                              (rgt, dict(x=0.98, ha='right'))]:
                ax_obj.text(kw_['x'], 0.5, txt, transform=ax_obj.transAxes,
                            color=ACCENT, fontsize=7, va='center',
                            ha=kw_['ha'], fontweight='bold')
            for txt, kw_ in [(bot, dict(y=0.02, va='bottom')),
                              (top_, dict(y=0.98, va='top'))]:
                ax_obj.text(0.5, kw_['y'], txt, transform=ax_obj.transAxes,
                            color=ACCENT, fontsize=7, ha='center',
                            va=kw_['va'], fontweight='bold')
            # Dynamic crosshairs — show the current position of the other two slices
            h_pos, v_pos = crosshair_pos[axis]
            ax_obj.axhline(h_pos, color='#00CFFF', lw=0.9, alpha=0.75, linestyle='--')
            ax_obj.axvline(v_pos, color='#FFD700', lw=0.9, alpha=0.75, linestyle='--')
        self._tri_fig.tight_layout(pad=0.4)
        self._tri_canvas.draw_idle()

    def _update_tri_sliders(self):
        if self._gray is None: return
        g = self._gray; dims = {'X': g.shape[0], 'Y': g.shape[1], 'Z': g.shape[2]}
        for ax, dim in dims.items():
            mid = dim // 2
            self._tri_sliders[ax].config(from_=0, to=dim-1)
            self._tri_sliders[ax].set(mid)
            self._tri_idx[ax] = mid; self._tri_idx_vars[ax].set(str(mid))

    def _export_slice(self, axis):
        if self._gray is None:
            messagebox.showwarning("No data", "Load a file first."); return
        path = filedialog.asksaveasfilename(
            title="Save slice as PNG", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if not path: return
        g = self._gray; idx = self._tri_idx[axis]; cmap = self._cmap_var.get()
        ww, wc = self._ww, self._wc
        if ww is None or wc is None:
            mn, mx = float(g.min()), float(g.max()); ww = mx-mn; wc = (mn+mx)/2.0
        if axis == 'X':   sl = g[idx,:,:].T
        elif axis == 'Y': sl = g[:,idx,:].T
        else:             sl = g[:,:,idx].T
        sl_win = _apply_window(sl.astype(np.float32), ww, wc)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,6), facecolor='white')
        ax.imshow(sl_win, cmap=cmap, origin='lower', vmin=0, vmax=1)
        ax.axis('off'); fig.tight_layout(pad=0); fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self._append_log(f"  Exported {axis} slice → {Path(path).name}", 'ok')

    # =========================================================================
    # 3-D VIEWER
    # =========================================================================

    def _build_3d_view(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="matplotlib not installed",
                     bg=BG, fg=ERR, font=FONT_UI).pack(expand=True); return

        ctrl = tk.Frame(parent, bg=BG); ctrl.pack(fill='x', padx=8, pady=4)

        # Threshold control
        r0 = tk.Frame(ctrl, bg=BG); r0.pack(side='left', padx=6)
        tk.Label(r0, text="Threshold percentile", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(anchor='w')
        self._3d_pct_var = tk.StringVar(value="50")
        styled_entry(r0, textvariable=self._3d_pct_var, width=7).pack(anchor='w')

        # Downsample
        r1 = tk.Frame(ctrl, bg=BG); r1.pack(side='left', padx=6)
        tk.Label(r1, text="Downsample (1=full)", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(anchor='w')
        self._3d_ds_var = tk.StringVar(value="2")
        styled_entry(r1, textvariable=self._3d_ds_var, width=7).pack(anchor='w')

        # Surface alpha
        r2 = tk.Frame(ctrl, bg=BG); r2.pack(side='left', padx=6)
        tk.Label(r2, text="Surface alpha", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(anchor='w')
        self._3d_alpha_var = tk.StringVar(value="0.85")
        styled_entry(r2, textvariable=self._3d_alpha_var, width=7).pack(anchor='w')

        # Color by E toggle
        r3 = tk.Frame(ctrl, bg=BG); r3.pack(side='left', padx=6)
        tk.Label(r3, text=" ", bg=BG, font=FONT_SMALL).pack(anchor='w')
        self._3d_show_emap = tk.BooleanVar(value=False)
        tk.Checkbutton(r3, text="Color by E-map", variable=self._3d_show_emap,
                       bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
                       activebackground=BG, font=FONT_SMALL).pack(anchor='w')

        styled_btn(ctrl, "▶  Render Surface", self._do_3d_render,
                   accent=True, small=True).pack(side='left', padx=10, pady=(14, 0))

        if not HAS_3D:
            tk.Label(parent, text="mpl_toolkits.mplot3d not available — pip install matplotlib",
                     bg=BG, fg=WARN, font=FONT_UI).pack(); return

        if not HAS_SKIMAGE:
            tk.Label(parent,
                     text="scikit-image not installed — pip install scikit-image\n"
                          "(needed for marching-cubes surface rendering)",
                     bg=BG, fg=WARN, font=FONT_UI, justify='center').pack(pady=4)

        self._3d_fig = Figure(facecolor=BG)
        self._3d_ax  = self._3d_fig.add_subplot(111, projection='3d')
        self._3d_ax.set_facecolor(PANEL2)
        self._3d_canvas = FigureCanvasTkAgg(self._3d_fig, master=parent)
        self._3d_canvas.get_tk_widget().pack(fill='both', expand=True)
        tb2 = tk.Frame(parent, bg=BG); tb2.pack(fill='x')
        NavigationToolbar2Tk(self._3d_canvas, tb2)

    def _do_3d_render(self):
        if not self._require_img(): return
        if not HAS_SKIMAGE:
            messagebox.showerror("Missing package",
                                 "scikit-image is required for surface rendering.\n"
                                 "Install with:  pip install scikit-image")
            return
        try:
            pct    = float(self._3d_pct_var.get())
            ds     = max(1, int(self._3d_ds_var.get()))
            alpha  = float(self._3d_alpha_var.get())
        except ValueError:
            messagebox.showerror("Bad input", "Check numeric fields."); return

        def _run():
            try:
                self._set_status("Building surface mesh...", busy=True)
                g = self._get_gray()

                # Downsample to keep marching cubes fast
                g_ds = g[::ds, ::ds, ::ds]

                thresh = float(np.percentile(g_ds, pct))
                self._append_log(
                    f"  3-D surface: pct={pct}%, threshold={thresh:.1f}, "
                    f"downsample=1/{ds}, shape={g_ds.shape}", 'teal')

                verts, faces, normals, _ = marching_cubes(g_ds, level=thresh)

                show_E = self._3d_show_emap.get() and self._E_map is not None

                def _draw():
                    if not HAS_3D: return
                    ax = self._3d_ax
                    ax.clear()
                    ax.set_facecolor(PANEL2)

                    if show_E:
                        # Sample E-map value at each vertex (scale back to original coords)
                        vs_orig = (verts * ds).astype(int)
                        vs_orig = np.clip(vs_orig, 0,
                                          np.array(self._E_map.shape) - 1)
                        E_verts = self._E_map[vs_orig[:,0], vs_orig[:,1], vs_orig[:,2]]
                        E_min, E_max = E_verts.min(), E_verts.max()
                        norm_E = (E_verts - E_min) / (E_max - E_min + 1e-9)
                        cmap_  = matplotlib.cm.get_cmap('viridis')
                        face_colors = cmap_(norm_E[faces].mean(axis=1))
                        mesh = Poly3DCollection(verts[faces],
                                               facecolors=face_colors,
                                               linewidth=0, alpha=alpha)
                        sm = matplotlib.cm.ScalarMappable(
                            cmap='viridis',
                            norm=mcolors.Normalize(vmin=E_min, vmax=E_max))
                        sm.set_array([])
                        self._3d_fig.colorbar(sm, ax=ax, shrink=0.5, label='E [MPa]')
                    else:
                        mesh = Poly3DCollection(verts[faces],
                                               facecolors=ACCENT,
                                               linewidth=0, alpha=alpha)

                    mesh.set_edgecolor('none')
                    ax.add_collection3d(mesh)

                    # Set axis limits to verts bounding box
                    ax.set_xlim(verts[:,0].min(), verts[:,0].max())
                    ax.set_ylim(verts[:,1].min(), verts[:,1].max())
                    ax.set_zlim(verts[:,2].min(), verts[:,2].max())

                    ax.set_xlabel('X', color=TEXT_DIM, fontsize=7)
                    ax.set_ylabel('Y', color=TEXT_DIM, fontsize=7)
                    ax.set_zlabel('Z', color=TEXT_DIM, fontsize=7)
                    ax.set_title(
                        f"Surface  |  pct>{pct}%  |  {len(faces):,} faces  |  ds=1/{ds}",
                        color=TEXT_DIM, fontsize=8)
                    ax.tick_params(colors=TEXT_DIM, labelsize=6)

                    self._3d_canvas.draw()

                self.after(0, _draw)
                self._set_status("3-D surface render complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  3-D surface error: {ex}", 'err')
                self._set_status("3-D render error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================================
    # HISTOGRAM TAB
    # =========================================================================

    def _build_histogram_tab(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="matplotlib not installed",
                     bg=BG, fg=ERR, font=FONT_UI).pack(expand=True); return
        self._hist_fig = Figure(facecolor=BG)
        self._hist_ax1, self._hist_ax2 = self._hist_fig.subplots(1, 2)
        for ax in (self._hist_ax1, self._hist_ax2):
            ax.set_facecolor(PANEL2)
        self._hist_canvas = FigureCanvasTkAgg(self._hist_fig, master=parent)
        self._hist_canvas.get_tk_widget().pack(fill='both', expand=True)
        tb3 = tk.Frame(parent, bg=BG); tb3.pack(fill='x')
        NavigationToolbar2Tk(self._hist_canvas, tb3)
        opts = tk.Frame(parent, bg=BG, pady=4); opts.pack(fill='x', padx=8)
        tk.Label(opts, text="Bins:", bg=BG, fg=TEXT_DIM, font=FONT_SMALL).pack(side='left')
        self._hist_bins_var = tk.StringVar(value="256")
        styled_entry(opts, textvariable=self._hist_bins_var, width=5).pack(side='left', padx=4)
        self._hist_show_hu = tk.BooleanVar(value=True)
        tk.Checkbutton(opts, text="Show HU axis", variable=self._hist_show_hu,
                       bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
                       activebackground=BG, font=FONT_SMALL).pack(side='left', padx=8)
        self._hist_shade_phases = tk.BooleanVar(value=True)
        tk.Checkbutton(opts, text="Shade phases", variable=self._hist_shade_phases,
                       bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
                       activebackground=BG, font=FONT_SMALL).pack(side='left', padx=4)
        styled_btn(opts, "Re-run", self._do_histogram, small=True).pack(side='left', padx=8)

    def _draw_histogram(self, counts, edges, zidx, means, mn, mx):
        if not HAS_MPL: return
        ax1 = self._hist_ax1; ax2 = self._hist_ax2
        ax1.clear(); ax2.clear()
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL2)
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_color(BORDER)

        centers = (edges[:-1] + edges[1:]) / 2
        ax1.bar(centers, counts, width=(edges[1]-edges[0]),
                color=ACCENT, alpha=0.8, linewidth=0)

        # Phase shading
        if self._hist_shade_phases.get():
            try:
                vt = float(self._void_thresh_var.get())
                ax1.axvspan(mn, vt, alpha=0.12, color='blue', label='Void')
                ax1.axvline(vt, color='blue', lw=1, linestyle='--', alpha=0.7)
                agg_str = self._agg_thresh_var.get().strip()
                if agg_str:
                    at = float(agg_str)
                    ax1.axvspan(vt, at, alpha=0.10, color='green', label='Matrix')
                    ax1.axvspan(at, mx, alpha=0.10, color='orange', label='Aggregate')
                    ax1.axvline(at, color='orange', lw=1, linestyle='--', alpha=0.7)
                ax1.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER)
            except Exception:
                pass

        ax1.set_xlabel("Intensity (raw)", color=TEXT_DIM, fontsize=8)
        ax1.set_ylabel("Voxel count",    color=TEXT_DIM, fontsize=8)
        ax1.set_title("Intensity Histogram", color=TEXT, fontsize=9)

        # Dual HU axis
        if self._hist_show_hu.get() and self._hu_cal:
            m_ = self._hu_cal.get('m', 1.0); c_ = self._hu_cal.get('c', 0.0)
            ax1_top = ax1.twiny()
            ax1_top.set_xlim(ax1.get_xlim())
            raw_ticks = ax1.get_xticks()
            hu_ticks  = [m_*r + c_ for r in raw_ticks]
            ax1_top.set_xticks(raw_ticks)
            ax1_top.set_xticklabels([f"{h:.0f}" for h in hu_ticks],
                                     fontsize=7, color=TEAL)
            ax1_top.set_xlabel("HU (calibrated)", color=TEAL, fontsize=8)

        # Slice mean intensity
        ax2.plot(zidx, means, color=ACCENT, lw=1.5)
        ax2.fill_between(zidx, means,
                          alpha=0.2, color=ACCENT)
        ax2.set_xlabel("Z slice index",   color=TEXT_DIM, fontsize=8)
        ax2.set_ylabel("Mean intensity",  color=TEXT_DIM, fontsize=8)
        ax2.set_title("Mean Intensity per Z Slice", color=TEXT, fontsize=9)

        self._hist_fig.tight_layout(pad=1.2)
        self._hist_canvas.draw()

    # =========================================================================
    # E-MAP VIEWER TAB
    # =========================================================================

    def _build_emap_tab(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="matplotlib not installed",
                     bg=BG, fg=ERR, font=FONT_UI).pack(expand=True); return

        ctrl = tk.Frame(parent, bg=BG, pady=4); ctrl.pack(fill='x', padx=8)
        tk.Label(ctrl, text="E colormap:", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left')
        self._ecmap_var = tk.StringVar(value="viridis")
        ecm_cb = ttk.Combobox(ctrl, textvariable=self._ecmap_var,
                              values=E_CMAPS, state='readonly', width=12)
        ecm_cb.pack(side='left', padx=4)
        ecm_cb.bind('<<ComboboxSelected>>', lambda e: self._refresh_emap_viewer())

        self._emap_overlay = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Overlay on CT", variable=self._emap_overlay,
                       bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
                       activebackground=BG, font=FONT_SMALL,
                       command=self._refresh_emap_viewer).pack(side='left', padx=8)
        tk.Label(ctrl, text="Alpha:", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left')
        self._emap_alpha_var = tk.StringVar(value="0.6")
        styled_entry(ctrl, textvariable=self._emap_alpha_var, width=5).pack(side='left', padx=4)
        styled_btn(ctrl, "Refresh", self._refresh_emap_viewer, small=True).pack(side='left', padx=8)

        self._emap_fig = Figure(facecolor=BG)
        self._emap_axes = self._emap_fig.subplots(1, 3)
        for ax in self._emap_axes:
            ax.set_facecolor(PANEL2); ax.axis('off')
        self._emap_fig.tight_layout(pad=0.4)
        self._emap_canvas = FigureCanvasTkAgg(self._emap_fig, master=parent)
        self._emap_canvas.get_tk_widget().pack(fill='both', expand=True)
        tb4 = tk.Frame(parent, bg=BG); tb4.pack(fill='x')
        NavigationToolbar2Tk(self._emap_canvas, tb4)

        # Slider bar — share with triplanar indices
        slider_bar = tk.Frame(parent, bg=PANEL2, pady=4); slider_bar.pack(fill='x')
        self._emap_sliders  = {}
        self._emap_idx_vars = {}
        self._emap_idx      = {'X': 0, 'Y': 0, 'Z': 0}
        axis_info = [('X', "Sagittal (X)"), ('Y', "Coronal  (Y)"), ('Z', "Axial    (Z)")]
        for col, (ax, label) in enumerate(axis_info):
            col_frame = tk.Frame(slider_bar, bg=PANEL2)
            col_frame.grid(row=0, column=col, sticky='ew', padx=8, pady=2)
            slider_bar.columnconfigure(col, weight=1)
            tk.Label(col_frame, text=label, bg=PANEL2, fg=TEAL,
                     font=FONT_SMALL).pack(anchor='w')
            row_f = tk.Frame(col_frame, bg=PANEL2); row_f.pack(fill='x')
            idx_var = tk.StringVar(value="--"); self._emap_idx_vars[ax] = idx_var
            sl = tk.Scale(row_f, from_=0, to=100, orient='horizontal',
                          bg=PANEL2, fg=TEXT, troughcolor=BORDER,
                          highlightthickness=0, bd=0, showvalue=False,
                          command=lambda val, a=ax: self._on_emap_drag(a, val))
            sl.pack(side='left', fill='x', expand=True)
            tk.Label(row_f, textvariable=idx_var, bg=PANEL2, fg=TEAL,
                     font=FONT_MONO, width=5).pack(side='left')
            self._emap_sliders[ax] = sl

        # Per-slice E histogram at bottom
        self._emap_slice_fig = Figure(figsize=(4, 1.8), facecolor=BG)
        self._emap_slice_ax  = self._emap_slice_fig.add_subplot(111)
        self._emap_slice_ax.set_facecolor(PANEL2)
        self._emap_slice_canvas = FigureCanvasTkAgg(self._emap_slice_fig, master=parent)
        self._emap_slice_canvas.get_tk_widget().pack(fill='x', padx=8, pady=4)

    def _on_emap_drag(self, axis, val):
        idx = int(float(val)); self._emap_idx[axis] = idx
        self._emap_idx_vars[axis].set(str(idx))
        self._refresh_emap_viewer()

    def _update_emap_sliders(self):
        if self._E_map is None: return
        for ax, dim in zip(('X','Y','Z'), self._E_map.shape[:3]):
            mid = dim // 2
            self._emap_sliders[ax].config(from_=0, to=dim-1)
            self._emap_sliders[ax].set(mid)
            self._emap_idx[ax] = mid; self._emap_idx_vars[ax].set(str(mid))

    def _refresh_emap_viewer(self):
        if not HAS_MPL or self._E_map is None: return
        E  = self._E_map
        cmap = self._ecmap_var.get()
        xi = max(0, min(self._emap_idx['X'], E.shape[0]-1))
        yi = max(0, min(self._emap_idx['Y'], E.shape[1]-1))
        zi = max(0, min(self._emap_idx['Z'], E.shape[2]-1))
        E_min, E_max = float(E.min()), float(E.max())
        norm = mcolors.Normalize(vmin=E_min, vmax=E_max)

        slices_E = {'X': E[xi,:,:].T, 'Y': E[:,yi,:].T, 'Z': E[:,:,zi].T}
        titles   = {'X': f"Sagittal X={xi}", 'Y': f"Coronal  Y={yi}", 'Z': f"Axial    Z={zi}"}

        overlay = self._emap_overlay.get()
        try:
            alpha = float(self._emap_alpha_var.get())
        except Exception:
            alpha = 0.6

        self._emap_fig.clear()
        axes = self._emap_fig.subplots(1, 3)

        for ax_obj, axis in zip(axes, ('X','Y','Z')):
            ax_obj.set_facecolor(PANEL2); ax_obj.axis('off')
            sl_E = slices_E[axis]
            if overlay and self._gray is not None:
                g = self._gray
                if axis == 'X':   sl_g = g[xi,:,:].T
                elif axis == 'Y': sl_g = g[:,yi,:].T
                else:             sl_g = g[:,:,zi].T
                mn_g, mx_g = float(sl_g.min()), float(sl_g.max())
                ww = mx_g - mn_g; wc = (mn_g + mx_g) / 2.0
                sl_g_w = _apply_window(sl_g.astype(np.float32), ww, wc)
                ax_obj.imshow(sl_g_w, cmap='gray', origin='lower', vmin=0, vmax=1,
                              aspect='equal', interpolation='nearest')
                ax_obj.imshow(sl_E, cmap=cmap, norm=norm, origin='lower',
                              aspect='equal', interpolation='nearest', alpha=alpha)
            else:
                im = ax_obj.imshow(sl_E, cmap=cmap, norm=norm, origin='lower',
                                   aspect='equal', interpolation='nearest')
                self._emap_fig.colorbar(im, ax=ax_obj, shrink=0.7,
                                         label='E [MPa]', fraction=0.046, pad=0.04)
            ax_obj.set_title(titles[axis], color=TEXT_DIM, fontsize=8, pad=3)

        self._emap_fig.tight_layout(pad=0.4)
        self._emap_canvas.draw_idle()

        # Per-slice E histogram (axial)
        self._emap_slice_ax.clear()
        self._emap_slice_ax.set_facecolor(PANEL2)
        sl_flat = E[:,:,zi].ravel()
        self._emap_slice_ax.hist(sl_flat, bins=64, color=TEAL, alpha=0.8, linewidth=0)
        self._emap_slice_ax.set_xlabel(f"E [MPa]  axial Z={zi}", color=TEXT_DIM, fontsize=8)
        self._emap_slice_ax.set_ylabel("count", color=TEXT_DIM, fontsize=8)
        self._emap_slice_ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in self._emap_slice_ax.spines.values(): sp.set_color(BORDER)
        self._emap_slice_fig.tight_layout(pad=0.4)
        self._emap_slice_canvas.draw_idle()

    # =========================================================================
    # SIM EXPORT TAB
    # =========================================================================

    def _build_export_tab(self, parent):
        outer = tk.Frame(parent, bg=BG); outer.pack(fill='both', expand=True, padx=16, pady=12)

        tk.Label(outer, text="Simulation Export", bg=BG, fg=TEXT,
                 font=FONT_HEAD).pack(anchor='w', pady=(0,8))
        tk.Label(outer, text="Run  ▶ Compute E-Map  first in the controls panel.",
                 bg=BG, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w')

        sep = tk.Frame(outer, bg=BORDER, height=1); sep.pack(fill='x', pady=8)

        # ── Export buttons ────────────────────────────────────────────────────
        for label, tip, cmd in [
            ("Export E-Map as NIfTI (.nii.gz)",
             "Float32 NIfTI — direct input to FCM / FEM pipeline",
             self._export_emap_nifti),
            ("Export E-Map as raw float32 binary",
             "C-order binary + JSON header sidecar (e.g. for custom solvers)",
             self._export_emap_raw),
            ("Export Phase Labels as NIfTI",
             "uint8 label map: 0=void, 1=matrix, 2=aggregate",
             self._export_labels_nifti),
            ("Export Phase Statistics CSV",
             "Per-phase volume fraction, HU stats, E stats",
             self._export_stats_csv),
            ("Generate ct.hpp snippet",
             "Copy-paste C++ for AMfracture / Finite Cell Method pipeline",
             self._export_ct_hpp),
        ]:
            btn_f = tk.Frame(outer, bg=PANEL, pady=8, padx=12,
                             highlightthickness=1, highlightbackground=BORDER)
            btn_f.pack(fill='x', pady=4)
            styled_btn(btn_f, label, cmd, teal=True).pack(side='left')
            tk.Label(btn_f, text=tip, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(side='left', padx=12)

        sep2 = tk.Frame(outer, bg=BORDER, height=1); sep2.pack(fill='x', pady=8)

        # ── ct.hpp preview ────────────────────────────────────────────────────
        tk.Label(outer, text="ct.hpp snippet preview:", bg=BG, fg=TEXT,
                 font=FONT_HEAD).pack(anchor='w', pady=(0,4))
        self._cpp_text = scrolledtext.ScrolledText(
            outer, bg="#1E1E1E", fg="#D4D4D4", insertbackground=ACCENT,
            font=("Consolas", 9), relief='flat', wrap='none', height=18)
        self._cpp_text.pack(fill='both', expand=True)
        self._cpp_text.insert('end', "// Compute the E-Map first, then click 'Generate ct.hpp snippet'.\n")

        btn_row = tk.Frame(outer, bg=BG, pady=4); btn_row.pack(fill='x')
        styled_btn(btn_row, "Copy to clipboard", self._copy_cpp, small=True).pack(side='left')
        styled_btn(btn_row, "Save as ct.hpp...", self._save_cpp, small=True).pack(side='left', padx=6)

    # =========================================================================
    # LOG PANE
    # =========================================================================

    def _build_log(self, parent):
        top_row = tk.Frame(parent, bg=BG); top_row.pack(fill='x', padx=6, pady=(6,0))
        tk.Label(top_row, text="Output / Log", bg=BG, fg=TEXT,
                 font=FONT_HEAD).pack(side='left')
        styled_btn(top_row, "Clear", self._clear_log, small=True).pack(side='right')
        self._log_widget = scrolledtext.ScrolledText(
            parent, bg=ENTRY_BG, fg=TEXT, insertbackground=ACCENT,
            font=FONT_MONO, relief='flat', wrap='word', state='disabled')
        self._log_widget.pack(fill='both', expand=True, padx=6, pady=6)
        self._log_widget.tag_config('accent', foreground=ACCENT)
        self._log_widget.tag_config('ok',     foreground=ACCENT2)
        self._log_widget.tag_config('warn',   foreground=WARN)
        self._log_widget.tag_config('err',    foreground=ERR)
        self._log_widget.tag_config('dim',    foreground=TEXT_DIM)
        self._log_widget.tag_config('teal',   foreground=TEAL)

    def _clear_log(self):
        self._log_widget.config(state='normal')
        self._log_widget.delete('1.0', 'end')
        self._log_widget.config(state='disabled')

    def _append_log(self, text: str, tag: str = ''):
        self.after(0, self._do_append_log, text, tag)

    def _do_append_log(self, text: str, tag: str):
        self._log_widget.config(state='normal')
        self._log_widget.insert('end', text + "\n", tag)
        self._log_widget.see('end')
        self._log_widget.config(state='disabled')

    def _log_sep(self, title: str = ''):
        line = f"── {title} {'─'*max(0, 50-len(title))}"
        self._append_log("", '')
        self._append_log(line, 'accent')

    def _set_status(self, msg: str, busy: bool = False):
        self.after(0, self._do_set_status, msg, busy)

    def _do_set_status(self, msg: str, busy: bool):
        self._status_var.set(msg)
        if busy: self._prog.start()
        else:    self._prog.stop()

    # =========================================================================
    # WINDOWING HELPERS
    # =========================================================================

    def _apply_preset(self, event=None):
        name = self._preset_var.get(); val = CT_PRESETS.get(name)
        if val is None: return
        ww, wc = val
        self._ww_var.set(str(ww)); self._wc_var.set(str(wc))
        self._ww = float(ww);      self._wc = float(wc)
        self._refresh_triplanar()

    def _apply_window_manual(self):
        try:
            self._ww = float(self._ww_var.get()); self._wc = float(self._wc_var.get())
        except ValueError:
            self._ww = self._wc = None; self._ww_var.set("auto"); self._wc_var.set("auto")
        self._refresh_triplanar()

    def _reset_window(self):
        self._ww = self._wc = None; self._ww_var.set("auto"); self._wc_var.set("auto")
        self._preset_var.set("-- preset --"); self._refresh_triplanar()

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open NIfTI file",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")])
        if path: self._load_path(path)

    def _load_path(self, path):
        if not HAS_NIBABEL:
            messagebox.showerror("Missing dependency", "nibabel is not installed."); return
        for attr in ('_gray','_img','_hu_vol','_E_map','_labels'):
            if getattr(self, attr) is not None:
                delattr(self, attr); setattr(self, attr, None)
        gc.collect()
        self._hu_cal = {}; self._cal_result_var.set("Not calibrated")
        self._log_sep(f"Loading: {Path(path).name}")
        self._status_var.set(f"Opening {Path(path).name}...")
        self._prog.begin_staged()
        threading.Thread(target=self._staged_load, args=(path,), daemon=True).start()

    def _staged_load(self, path):
        try:
            img = nib.load(str(path)); self.after(0, self._prog.advance)
            raw = _raw(img);          self.after(0, self._prog.advance)
            gray = _to_gray(raw); del raw; gc.collect()
            self.after(0, self._prog.advance)
            self._img = img; self._path = Path(path)
            self._gray = gray; self._ww = None; self._wc = None
            self._axis_labels = _get_axis_labels(img.affine)
            shape_str = str(img.shape); dtype_str = str(img.get_data_dtype())
            spacing   = img.header.get_zooms()
            phys = tuple(round(float(img.shape[i])*float(spacing[i]),2)
                         for i in range(min(3,len(img.shape))))
            self.after(0, lambda: self._info_var.set(
                f"{Path(path).name}   |   {shape_str}   |   {dtype_str}   |   {phys} mm"))
            self.after(0, lambda: self._ww_var.set("auto"))
            self.after(0, lambda: self._wc_var.set("auto"))
            self._append_log(f"  Loaded: {shape_str}  dtype={dtype_str}", 'ok')
            self._append_log(f"  Physical size: {phys} mm", 'dim')
            self.after(0, self._autofill_crop)
            self.after(0, self._update_tri_sliders)
            self.after(50, self._refresh_triplanar)
            self.after(50, self._prog.advance)
            self.after(200, self._update_stats)
            self.after(250, self._prog.advance)
            self.after(600, self._prog.done)
            self.after(600, lambda: self._status_var.set(f"Loaded: {Path(path).name}"))
        except Exception as ex:
            self._append_log(f"  Error loading: {ex}", 'err')
            self.after(0, lambda: self._status_var.set("Error loading file."))
            self.after(0, self._prog.stop)

    def _get_gray(self):
        if self._gray is None and self._img is not None:
            self._gray = _to_gray(_raw(self._img))
            self.after(0, self._update_tri_sliders)
        return self._gray

    def _require_img(self) -> bool:
        if self._img is None:
            messagebox.showwarning("No file", "Please open a NIfTI file first."); return False
        return True

    def _require_emap(self) -> bool:
        if self._E_map is None:
            messagebox.showwarning("No E-Map", "Please compute the E-Map first."); return False
        return True

    def _ask_save_path(self, default_name: str, ext=".nii.gz",
                        ftypes=None):
        ftypes = ftypes or [("NIfTI gz", "*.nii.gz"), ("NIfTI", "*.nii"), ("All", "*.*")]
        return filedialog.asksaveasfilename(
            title="Save output as...", initialfile=default_name,
            defaultextension=ext, filetypes=ftypes) or None

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def _update_stats(self):
        if self._gray is None: return
        g = self._gray; flat = g.ravel()
        vals = {
            "Min":  f"{float(flat.min()):.2f}",
            "Max":  f"{float(flat.max()):.2f}",
            "Mean": f"{float(flat.mean()):.4f}",
            "Std":  f"{float(flat.std()):.4f}",
            "p1":   f"{float(np.percentile(flat,  1)):.2f}",
            "p99":  f"{float(np.percentile(flat, 99)):.2f}",
        }
        for k, v in vals.items():
            self._stat_labels[k].set(v)

    # =========================================================================
    # ACTIONS — STANDARD
    # =========================================================================

    def _do_gunzip(self):
        path = filedialog.askopenfilename(
            title="Select .gz file",
            filetypes=[("GZipped files", "*.gz"), ("All files", "*.*")])
        if not path: return
        def _run():
            try:
                self._log_sep(f"Gunzip: {Path(path).name}")
                self._set_status(f"Decompressing...", busy=True)
                out = run_gunzip(path)
                self._append_log(f"  → {out}", 'ok')
                self._set_status("Gunzip complete.", busy=False)
                if messagebox.askyesno("Done", f"Decompressed to:\n{out}\n\nLoad now?"):
                    self._load_path(str(out))
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Gunzip failed.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _do_metadata(self):
        if not self._require_img(): return
        def _run():
            try:
                self._log_sep("Metadata")
                self._set_status("Reading metadata...", busy=True)
                txt = read_metadata(self._img, self._gray, self._hu_vol, self._labels)
                self._append_log(txt, 'dim')
                self._set_status("Metadata read.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Metadata error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _do_histogram(self):
        if not self._require_img(): return
        def _run():
            try:
                self._log_sep("Histogram & Brightness Analysis")
                self._set_status("Computing histogram...", busy=True)
                gray = self._get_gray()
                try:
                    n_bins = int(self._hist_bins_var.get())
                except Exception:
                    n_bins = 256
                zidx, means, counts, edges, mn, mx = compute_histogram(gray, n_bins)
                self._append_log(f"  Shape: {gray.shape}  range: {mn:.2f}–{mx:.2f}", 'dim')
                self._append_log(f"  Mean: {gray.mean():.4f}  Std: {gray.std():.4f}", 'dim')
                self.after(0, self._draw_histogram, counts, edges, zidx, means, mn, mx)
                self.after(0, self._update_stats)
                self._set_status("Histogram complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Histogram error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _do_reorient(self):
        if not self._require_img(): return
        target = self._reorient_target.get().strip().upper()
        out_path = None
        if self._reorient_save.get():
            stem = self._path.name.replace('.nii.gz','').replace('.nii','')
            out_path = self._ask_save_path(f"{stem}_reoriented_{target}.nii.gz")
            if not out_path: return
        def _run():
            try:
                if len(target) != 3:
                    self._append_log("Target must be 3 letters (e.g. RAS).", 'warn'); return
                self._log_sep(f"Reorientation → {target}")
                self._set_status(f"Reorienting to {target}...", busy=True)
                result = run_reorientation(self._img, target)
                self._append_log(f"  New shape: {result.shape}", 'ok')
                if out_path:
                    nib.save(result, str(out_path))
                    self._append_log(f"  Saved → {Path(out_path).name}", 'ok')
                    self._set_status("Reorientation saved.", busy=False)
                    if messagebox.askyesno("Saved", "Load reoriented file now?"):
                        self._load_path(str(out_path))
                else:
                    self._img = result; self._gray = None
                    self._axis_labels = _get_axis_labels(result.affine)
                    self._set_status("Reorientation done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Reorientation error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _do_rotate(self):
        if not self._require_img(): return
        try:
            axis = self._rot_axis.get(); angle = float(self._rot_angle.get())
        except ValueError:
            messagebox.showerror("Bad input", "Angle must be a number."); return
        out_path = None
        if self._rot_save.get():
            stem = self._path.name.replace('.nii.gz','').replace('.nii','')
            out_path = self._ask_save_path(f"{stem}_rot_{axis}_{int(angle)}deg.nii.gz")
            if not out_path: return
        def _run():
            try:
                self._log_sep(f"Rotation  axis={axis}  angle={angle}°")
                self._set_status(f"Rotating {angle}° around {axis.upper()}...", busy=True)
                result = run_angle_rotation(self._img, axis, angle)
                self._append_log(f"  New shape: {result.shape}", 'ok')
                if out_path:
                    nib.save(result, str(out_path)); self._set_status("Rotation saved.", busy=False)
                    if messagebox.askyesno("Saved", "Load rotated file now?"):
                        self._load_path(str(out_path))
                else:
                    self._img = result; self._gray = None; self._set_status("Rotation done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Rotation error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _autofill_crop(self):
        if self._img is None: return
        for axis, dim in zip(('X','Y','Z'), self._img.shape[:3]):
            self._crop_rows[axis].start_var.set("0")
            self._crop_rows[axis].end_var.set(str(dim))

    def _do_crop(self):
        if not self._require_img(): return
        try:
            ranges = {ax: (int(self._crop_rows[ax].start_var.get()),
                           int(self._crop_rows[ax].end_var.get()))
                      for ax in ('X','Y','Z')}
        except ValueError:
            messagebox.showerror("Bad input", "Crop start/end must be integers."); return
        out_path = None
        if self._crop_save.get():
            stem = self._path.name.replace('.nii.gz','').replace('.nii','')
            out_path = self._ask_save_path(f"{stem}_cropped.nii.gz")
            if not out_path: return
        def _run():
            try:
                self._log_sep(f"Crop  X{ranges['X']}  Y{ranges['Y']}  Z{ranges['Z']}")
                self._set_status("Cropping...", busy=True)
                result = run_cropper(self._img, ranges['X'], ranges['Y'], ranges['Z'])
                self._append_log(f"  Cropped shape: {result.shape}", 'ok')
                if out_path:
                    nib.save(result, str(out_path)); self._set_status("Crop saved.", busy=False)
                    if messagebox.askyesno("Saved", "Load cropped file now?"):
                        self._load_path(str(out_path))
                else:
                    self._img = result; self._gray = None; self._set_status("Crop done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Crop error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================================
    # ACTIONS — MATERIAL MAPPING
    # =========================================================================

    def _do_calibrate(self):
        if not self._require_img(): return
        try:
            air_int = float(self._cal_air_int.get())
            ref_int = float(self._cal_ref_int.get())
            air_hu  = float(self._cal_air_hu.get())
            ref_hu  = float(self._cal_ref_hu.get())
        except ValueError:
            messagebox.showerror("Bad input", "All calibration fields must be numbers."); return

        def _run():
            try:
                self._log_sep("HU Calibration")
                self._set_status("Calibrating to HU...", busy=True)
                gray = self._get_gray()
                hu_vol, m, c = calibrate_to_hu(gray, air_int, ref_int, air_hu, ref_hu)
                self._hu_vol  = hu_vol
                self._hu_cal  = {'m': m, 'c': c,
                                  'air_int': air_int, 'ref_int': ref_int}
                result_text = f"HU = {m:.4f}·raw + {c:.2f}"
                self.after(0, lambda: self._cal_result_var.set(result_text))
                self._append_log(f"  Calibration: {result_text}", 'teal')
                self._append_log(f"  HU range: {float(hu_vol.min()):.1f}  to  {float(hu_vol.max()):.1f}", 'dim')
                self._set_status("HU calibration complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  Calibration error: {ex}", 'err')
                self._set_status("Calibration error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _do_compute_emap(self):
        if not self._require_img(): return
        if self._hu_vol is None:
            if not messagebox.askyesno(
                    "No HU calibration",
                    "HU calibration has not been applied.\n"
                    "The raw intensity will be used directly as HU.\n\n"
                    "Continue anyway?"):
                return

        try:
            model  = self._model_var.get()
            params = self._get_model_params()
            void_thresh = float(self._void_thresh_var.get())
            agg_str = self._agg_thresh_var.get().strip()
            agg_thresh = float(agg_str) if agg_str else None
        except Exception as ex:
            messagebox.showerror("Bad parameters", str(ex)); return

        def _run():
            try:
                self._log_sep(f"E-Map Computation  model={model}")
                self._set_status("Computing E-Map...", busy=True)
                gray = self._get_gray()
                hu_vol = self._hu_vol if self._hu_vol is not None else gray

                E_map  = compute_E_map(hu_vol, model, params, void_thresh)
                labels = segment_phases(hu_vol, void_thresh, agg_thresh)
                stats, porosity = phase_statistics(hu_vol, E_map, labels)

                self._E_map   = E_map
                self._labels  = labels
                self._E_stats = stats
                self._porosity = porosity

                self._append_log(f"  E-Map shape : {E_map.shape}", 'teal')
                self._append_log(f"  E range     : {float(E_map.min()):.2f}  –  {float(E_map.max()):.2f} MPa", 'teal')
                self._append_log(f"  E mean ± std: {float(E_map.mean()):.2f} ± {float(E_map.std()):.2f} MPa", 'teal')
                self._append_log(f"  Porosity    : {porosity*100:.2f}%", 'teal')

                for phase_name, s in stats.items():
                    self._append_log(
                        f"  {phase_name:20s}  vol={s['vol_frac']*100:.2f}%  "
                        f"E={s['E_mean_MPa']:.1f} MPa", 'dim')

                # Update E stat labels
                def _upd():
                    self._e_stat_labels["E_min"].set(f"{float(E_map.min()):.2f} MPa")
                    self._e_stat_labels["E_max"].set(f"{float(E_map.max()):.2f} MPa")
                    self._e_stat_labels["E_mean"].set(f"{float(E_map.mean()):.2f} MPa")
                    self._e_stat_labels["E_std"].set(f"{float(E_map.std()):.2f} MPa")
                    self._e_stat_labels["porosity"].set(f"{porosity*100:.2f}%")
                self.after(0, _upd)
                self.after(0, self._update_emap_sliders)
                self.after(50, self._refresh_emap_viewer)
                self._set_status("E-Map complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  E-Map error: {ex}", 'err')
                self._set_status("E-Map error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================================
    # EXPORT ACTIONS
    # =========================================================================

    def _export_emap_nifti(self):
        if not self._require_emap(): return
        stem = self._path.name.replace('.nii.gz','').replace('.nii','')
        out = self._ask_save_path(f"{stem}_Emap.nii.gz")
        if not out: return
        def _run():
            try:
                self._log_sep("Export E-Map NIfTI")
                self._set_status("Saving E-Map NIfTI...", busy=True)
                hdr = self._img.header.copy()
                hdr.set_data_dtype(np.float32)
                emap_img = nib.Nifti1Image(self._E_map, self._img.affine, hdr)
                nib.save(emap_img, str(out))
                self._append_log(f"  Saved → {Path(out).name}", 'ok')
                self._set_status("E-Map NIfTI saved.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Export error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _export_emap_raw(self):
        if not self._require_emap(): return
        stem = self._path.name.replace('.nii.gz','').replace('.nii','')
        out = self._ask_save_path(f"{stem}_Emap.raw",
                                   ext=".raw",
                                   ftypes=[("Raw binary", "*.raw"), ("All","*.*")])
        if not out: return
        def _run():
            try:
                self._log_sep("Export E-Map raw float32")
                self._set_status("Saving raw binary...", busy=True)
                E_c = np.ascontiguousarray(self._E_map, dtype=np.float32)
                with open(out, 'wb') as f:
                    f.write(E_c.tobytes())
                # JSON sidecar
                spacing = self._img.header.get_zooms()
                meta = {
                    "shape": list(self._E_map.shape),
                    "dtype": "float32",
                    "order": "C",
                    "voxel_spacing_mm": [float(s) for s in spacing[:3]],
                    "E_min_MPa": float(self._E_map.min()),
                    "E_max_MPa": float(self._E_map.max()),
                    "hu_calibration": self._hu_cal,
                    "source_file": str(self._path),
                }
                json_out = str(out).replace('.raw', '_header.json')
                with open(json_out, 'w') as f:
                    json.dump(meta, f, indent=2)
                self._append_log(f"  Binary  → {Path(out).name}", 'ok')
                self._append_log(f"  Header  → {Path(json_out).name}", 'ok')
                self._set_status("Raw export saved.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Export error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _export_labels_nifti(self):
        if self._labels is None:
            messagebox.showwarning("No labels", "Compute E-Map first."); return
        stem = self._path.name.replace('.nii.gz','').replace('.nii','')
        out = self._ask_save_path(f"{stem}_phases.nii.gz")
        if not out: return
        def _run():
            try:
                self._log_sep("Export Phase Labels NIfTI")
                self._set_status("Saving labels...", busy=True)
                hdr = self._img.header.copy(); hdr.set_data_dtype(np.uint8)
                lbl_img = nib.Nifti1Image(self._labels, self._img.affine, hdr)
                nib.save(lbl_img, str(out))
                self._append_log(f"  Saved → {Path(out).name}  (0=void, 1=matrix, 2=aggregate)", 'ok')
                self._set_status("Labels saved.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err'); self._set_status("Export error.", busy=False)
        threading.Thread(target=_run, daemon=True).start()

    def _export_stats_csv(self):
        if not self._E_stats:
            messagebox.showwarning("No stats", "Compute E-Map first."); return
        stem = self._path.name.replace('.nii.gz','').replace('.nii','')
        out = self._ask_save_path(f"{stem}_phase_stats.csv", ext=".csv",
                                   ftypes=[("CSV","*.csv"),("All","*.*")])
        if not out: return
        try:
            lines = ["phase,voxels,vol_frac,HU_mean,HU_std,E_mean_MPa,E_min_MPa,E_max_MPa,E_std_MPa"]
            for phase, s in self._E_stats.items():
                lines.append(
                    f"{phase},{s['voxels']},{s['vol_frac']:.6f},"
                    f"{s['HU_mean']:.4f},{s['HU_std']:.4f},"
                    f"{s['E_mean_MPa']:.4f},{s['E_min_MPa']:.4f},"
                    f"{s['E_max_MPa']:.4f},{s['E_std_MPa']:.4f}"
                )
            lines.append(f"porosity,,,{self._porosity:.6f},,,,")
            with open(out, 'w') as f:
                f.write("\n".join(lines))
            self._append_log(f"  Stats CSV → {Path(out).name}", 'ok')
        except Exception as ex:
            self._append_log(f"  {ex}", 'err')

    def _export_ct_hpp(self):
        if not self._require_emap(): return
        try:
            params = self._get_model_params()
            model  = self._model_var.get()
            void_thresh = float(self._void_thresh_var.get())
        except Exception as ex:
            messagebox.showerror("Parameter error", str(ex)); return

        snippet = generate_ct_hpp_snippet(
            self._gray, self._hu_cal if self._hu_cal else {'m':1.0,'c':0.0},
            self._E_map, self._img, model, params, void_thresh,
            self._hu_cal.get('air_int', 0), self._hu_cal.get('ref_int', 199)
        )
        self._cpp_text.config(state='normal')
        self._cpp_text.delete('1.0', 'end')
        self._cpp_text.insert('end', snippet)
        self._cpp_text.config(state='disabled')
        self._nb.select(self._export_tab)
        self._append_log("  ct.hpp snippet generated — see Sim Export tab.", 'teal')

    def _copy_cpp(self):
        txt = self._cpp_text.get('1.0', 'end')
        self.clipboard_clear(); self.clipboard_append(txt)
        self._append_log("  Copied ct.hpp snippet to clipboard.", 'ok')

    def _save_cpp(self):
        out = self._ask_save_path("ct_snippet.hpp", ext=".hpp",
                                   ftypes=[("C++ header","*.hpp *.h"),("All","*.*")])
        if not out: return
        txt = self._cpp_text.get('1.0', 'end')
        with open(out, 'w') as f: f.write(txt)
        self._append_log(f"  Saved → {Path(out).name}", 'ok')


# =============================================================================

if __name__ == "__main__":
    app = NiftiApp()
    app.mainloop()
