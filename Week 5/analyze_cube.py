"""
CT Cube Analysis Script — Fast Subsampled Version
Amit Yadav | AMfracture Project | Bauhaus-Universität Weimar

Usage:
    python analyze_cube.py --file cube_wc03.nii.gz --wc 0.3 --step 3
    python analyze_cube.py --file cube_wc04.nii.gz --wc 0.4 --step 3
    python analyze_cube.py --file cube_wc05.nii.gz --wc 0.5 --step 3

    --step 3  means every 3rd voxel → 1/27th of voxels, ~27x faster
    --step 2  means every 2nd voxel → 1/8th of voxels, ~8x faster
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="Path to cropped NIfTI cube")
parser.add_argument("--wc",   required=True, help="w/c ratio label e.g. 0.3")
parser.add_argument("--step", type=int, default=3, help="Subsampling step (default 3)")
parser.add_argument("--raw",  action="store_true", help="Data is in raw units (not HU)")
args = parser.parse_args()

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
LABEL = f"wc{args.wc.replace('.', '')}"

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading {args.file} ...")
img  = nib.load(args.file)
data = img.get_fdata()
print(f"      Full volume shape : {data.shape}")
print(f"      Full voxel count  : {data.size:,}")

# ─── SUBSAMPLE ────────────────────────────────────────────────────────────────
s = args.step
sub = data[::s, ::s, ::s]
print(f"\n[2/5] Subsampled (step={s}) shape : {sub.shape}")
print(f"      Subsampled voxel count       : {sub.size:,}  ({100*sub.size/data.size:.1f}% of full)")

# ─── HU CONVERSION ────────────────────────────────────────────────────────────
# If your scanner outputs raw 16-bit values, convert to HU first.
# Common linear mapping: HU = raw * slope + intercept
# Check your NIfTI header for scl_slope / scl_inter — or use --raw flag
# and we'll apply a default rescale.

header = img.header
slope  = float(header.get_slope_inter()[0] or 1.0)
inter  = float(header.get_slope_inter()[1] or 0.0)

if slope != 1.0 or inter != 0.0:
    print(f"\n      NIfTI header rescale: HU = raw × {slope:.4f} + {inter:.1f}")
    hu_sub  = sub  * slope + inter
    hu_full = data * slope + inter   # for z-profile (full z, subsampled xy)
else:
    print(f"\n      No rescale in header — using values as-is.")
    print(f"      If x-axis shows 0–27000, pass --raw to apply default offset.")
    if args.raw:
        # Typical GE phoenix 16-bit: midpoint ~32768 offset
        # Adjust RAW_OFFSET if needed after checking your scanner docs
        RAW_OFFSET = 0
        hu_sub  = sub  - RAW_OFFSET
        hu_full = data - RAW_OFFSET
        print(f"      Applied raw offset: {RAW_OFFSET}")
    else:
        hu_sub  = sub
        hu_full = data

# ─── STATISTICS ───────────────────────────────────────────────────────────────
print(f"\n[3/5] Computing statistics ...")

# Exclude air voxels (below -500 HU or equivalent) from cement stats
VOID_THRESHOLD = np.percentile(hu_sub, 5)   # bottom 5% = pores/air
mask = hu_sub > VOID_THRESHOLD

hu_flat  = hu_sub[mask].ravel()
hu_all   = hu_sub.ravel()

stats = {
    "mean_HU"    : np.mean(hu_flat),
    "median_HU"  : np.median(hu_flat),
    "std_HU"     : np.std(hu_flat),
    "skewness"   : float(np.mean(((hu_flat - np.mean(hu_flat))/np.std(hu_flat))**3)),
    "cv"         : np.std(hu_flat) / np.mean(hu_flat),          # coefficient of variation
    "porosity_pct": 100 * np.sum(hu_sub <= VOID_THRESHOLD) / hu_sub.size,
    "min_HU"     : np.min(hu_all),
    "max_HU"     : np.max(hu_all),
}

# ─── HU → DENSITY → E ────────────────────────────────────────────────────────
# Step 1: HU → density
# rho [g/cm³] = (HU + 1000) / 1000   (water=1.0, air=0.0)
# Note: if your data is in raw units this will be wrong until HU is calibrated
# For now we compute assuming HU axis is correct after header rescale

def hu_to_density(hu):
    return np.clip((hu + 1000) / 1000, 0, None)

def density_to_E(rho, E0=33.0, rho0=2.0, n=2.0):
    """Garboczi power law. E0 in GPa."""
    return np.clip(E0 * (rho / rho0) ** n, 0, None)

rho_field = hu_to_density(hu_sub)
E_field   = density_to_E(rho_field)

rho_mask  = rho_field[mask]
E_mask    = E_field[mask]

stats["mean_rho"] = np.mean(rho_mask)
stats["mean_E"]   = np.mean(E_mask)
stats["std_E"]    = np.std(E_mask)

# ─── PRINT TABLE ──────────────────────────────────────────────────────────────
print(f"\n{'─'*45}")
print(f"  Results for w/c = {args.wc}")
print(f"{'─'*45}")
print(f"  Mean HU          : {stats['mean_HU']:>10.1f}")
print(f"  Median HU        : {stats['median_HU']:>10.1f}")
print(f"  Std HU           : {stats['std_HU']:>10.1f}")
print(f"  Skewness         : {stats['skewness']:>10.3f}")
print(f"  CV (std/mean)    : {stats['cv']:>10.4f}")
print(f"  Porosity %       : {stats['porosity_pct']:>10.2f} %")
print(f"  Mean density     : {stats['mean_rho']:>10.3f} g/cm³")
print(f"  Mean E           : {stats['mean_E']:>10.2f} GPa")
print(f"  Std E            : {stats['std_E']:>10.2f} GPa")
print(f"{'─'*45}\n")

# ─── PLOTS ────────────────────────────────────────────────────────────────────
print(f"[4/5] Plotting ...")

DPI     = 300
STYLE   = {"fontsize": 13}
TITLE_S = 15

# ── Plot 1: HU Histogram ──────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 6))
ax.hist(hu_all, bins=256, color="#2196F3", alpha=0.85, edgecolor="none")
ax.axvline(VOID_THRESHOLD, color="#E53935", linestyle="--", linewidth=2,
           label=f"Void threshold ({VOID_THRESHOLD:.0f})")
ax.axvline(stats["mean_HU"], color="#FF8F00", linestyle="-", linewidth=2,
           label=f"Matrix mean ({stats['mean_HU']:.0f})")
ax.set_xlabel("HU / Intensity", **STYLE)
ax.set_ylabel("Voxel count", **STYLE)
ax.set_title(f"HU Intensity Histogram — w/c = {args.wc}", fontsize=TITLE_S, fontweight="bold")
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
fig1.tight_layout()
out1 = OUT_DIR / f"plot1_histogram_{LABEL}.png"
fig1.savefig(out1, dpi=DPI, bbox_inches="tight")
print(f"      Saved → {out1}")
plt.close(fig1)

# ── Plot 2: E Distribution PDF ────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 6))
E_flat = E_mask.ravel()
ax.hist(E_flat, bins=200, color="#43A047", alpha=0.85, edgecolor="none", density=True)
ax.axvline(stats["mean_E"], color="#E53935", linestyle="--", linewidth=2,
           label=f"Mean E = {stats['mean_E']:.2f} GPa")
ax.axvline(stats["mean_E"] - stats["std_E"], color="#FF8F00", linestyle=":",
           linewidth=1.5, label=f"±1 std  ({stats['std_E']:.2f} GPa)")
ax.axvline(stats["mean_E"] + stats["std_E"], color="#FF8F00", linestyle=":",
           linewidth=1.5)
ax.set_xlabel("Young's Modulus E (GPa)", **STYLE)
ax.set_ylabel("Probability density", **STYLE)
ax.set_title(f"E Distribution (Simulation Input) — w/c = {args.wc}", fontsize=TITLE_S, fontweight="bold")
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
fig2.tight_layout()
out2 = OUT_DIR / f"plot2_E_distribution_{LABEL}.png"
fig2.savefig(out2, dpi=DPI, bbox_inches="tight")
print(f"      Saved → {out2}")
plt.close(fig2)

# ── Plot 3: Z-slice Mean HU Profile ──────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(12, 5))
z_means = []
for z in range(data.shape[2]):
    slc = hu_full[::s, ::s, z]
    z_means.append(np.mean(slc))
z_means = np.array(z_means)
ax.plot(z_means, color="#7B1FA2", linewidth=1.5, label="Per-slice mean HU")
ax.axhline(stats["mean_HU"], color="#FF8F00", linestyle="--", linewidth=1.5,
           label=f"Bulk mean ({stats['mean_HU']:.0f})")
ax.fill_between(range(len(z_means)),
                stats["mean_HU"] - stats["std_HU"],
                stats["mean_HU"] + stats["std_HU"],
                alpha=0.15, color="#FF8F00", label="±1 std band")
ax.set_xlabel("Z slice index", **STYLE)
ax.set_ylabel("Mean HU / Intensity", **STYLE)
ax.set_title(f"Homogeneity Along Z Axis — w/c = {args.wc}", fontsize=TITLE_S, fontweight="bold")
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
fig3.tight_layout()
out3 = OUT_DIR / f"plot3_z_profile_{LABEL}.png"
fig3.savefig(out3, dpi=DPI, bbox_inches="tight")
print(f"      Saved → {out3}")
plt.close(fig3)

# ─── SAVE STATS ───────────────────────────────────────────────────────────────
print(f"[5/5] Saving stats ...")
out_csv = OUT_DIR / f"stats_{LABEL}.txt"
with open(out_csv, "w") as f:
    f.write(f"w/c ratio : {args.wc}\n")
    for k, v in stats.items():
        f.write(f"{k:20s}: {v:.4f}\n")
print(f"      Saved → {out_csv}")
print(f"\nDone.\n")