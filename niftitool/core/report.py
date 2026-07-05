"""One-click PDF analysis report.

Bundles everything the GUI has computed for the loaded volume into a
multi-page, print-friendly PDF (white background, A4 landscape):

1. Title page — file info, calibration, material model, key results.
2. Orthogonal mid-slices with the current window applied.
3. Intensity histogram + per-slice mean curve.
4. Phase segmentation overlay + per-phase statistics table.
5. Void analytics (size distribution, cumulative volume, top-10 table)
   — included when a :class:`~.porosity.PorosityResult` is supplied.
6. E-Map mid-slice + statistics — included when an E-map is supplied.

The function is GUI-free: it takes numpy arrays and plain dicts, builds
matplotlib ``Figure`` objects directly on the Agg backend (safe from a
worker thread — no Qt canvas involved), and writes with ``PdfPages``.
"""

from __future__ import annotations

import datetime

from ..deps import np, HAS_MPL

# Print palette (independent of the app theme — reports must print well).
_INK      = "#1A1A1A"
_DIM      = "#666666"
_ACCENT   = "#E07B2A"
_GREEN    = "#6A9E3F"
_TEAL     = "#1A7A8A"
_GRID     = "#DDDDDD"
_PAGE     = (11.69, 8.27)   # A4 landscape, inches

_PHASE_COLORS = ["#3B82F6", "#6A9E3F", "#E07B2A"]  # void, matrix, aggregate


def _new_page():
    from matplotlib.figure import Figure
    fig = Figure(figsize=_PAGE, facecolor="white")
    return fig

def _style_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors=_DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_GRID)
    ax.grid(True, color=_GRID, lw=0.5, alpha=0.6)

def _footer(fig, page_no):
    fig.text(0.985, 0.015, f"NIfTI Tool Suite — page {page_no}",
             ha="right", va="bottom", fontsize=7, color=_DIM)

def _window(img2d, ww, wc):
    """Apply CT windowing; fall back to p1–p99 stretch."""
    a = np.asarray(img2d, dtype=np.float32)
    if ww and wc is not None:
        lo, hi = wc - ww / 2.0, wc + ww / 2.0
    else:
        lo, hi = np.percentile(a, [1, 99])
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max() or 1.0)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

def _mid_slices(vol):
    """(sagittal, coronal, axial) middle slices, display-oriented."""
    sx, sy, sz = (s // 2 for s in vol.shape[:3])
    return (
        ("Sagittal  x=%d" % sx, np.rot90(vol[sx, :, :])),
        ("Coronal   y=%d" % sy, np.rot90(vol[:, sy, :])),
        ("Axial     z=%d" % sz, np.rot90(vol[:, :, sz])),
    )


def generate_pdf_report(
    out_path,
    *,
    file_name: str = "volume",
    shape=None,
    spacing=None,
    dtype: str = "",
    gray=None,
    ww=None, wc=None,
    hu_cal: dict | None = None,
    hu_vol=None,
    labels=None,
    phase_stats: dict | None = None,
    porosity: float | None = None,
    void_result=None,
    E_map=None,
    model_name: str = "",
    model_params: dict | None = None,
    void_thresh=None,
    progress=None,           # fn(str) — stage callback for the UI log
) -> int:
    """Write the PDF; returns the number of pages."""
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for PDF reports")
    from matplotlib.backends.backend_pdf import PdfPages

    def _stage(msg):
        if progress is not None:
            progress(msg)

    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    page = 0

    with PdfPages(str(out_path)) as pdf:

        # ── page 1 : title ────────────────────────────────────────────
        _stage("title page")
        page += 1
        fig = _new_page()
        fig.text(0.06, 0.86, "CT Analysis Report", fontsize=26,
                 fontweight="bold", color=_INK)
        fig.text(0.06, 0.80, file_name, fontsize=13, color=_ACCENT,
                 family="monospace")
        fig.text(0.06, 0.76, f"Generated {now}  ·  NIfTI Tool Suite",
                 fontsize=9, color=_DIM)

        rows = []
        if shape is not None:
            rows.append(("Volume shape", " × ".join(str(s) for s in shape[:3]) + "  voxels"))
        if spacing is not None:
            rows.append(("Voxel spacing", "  ×  ".join(f"{float(s):.4f}" for s in spacing[:3]) + "  mm"))
            if shape is not None:
                dims = [s * float(sp) for s, sp in zip(shape[:3], spacing[:3])]
                rows.append(("Physical size", "  ×  ".join(f"{d:.2f}" for d in dims) + "  mm"))
        if dtype:
            rows.append(("Data type", dtype))
        if hu_cal and "m" in hu_cal:
            rows.append(("HU calibration", f"HU = {hu_cal['m']:.6g} · I  +  {hu_cal['c']:.6g}"))
        else:
            rows.append(("HU calibration", "not applied (raw intensities)"))
        if model_name:
            p = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                          for k, v in (model_params or {}).items())
            rows.append(("Material model", f"{model_name}   ({p})" if p else model_name))
        if void_thresh is not None:
            rows.append(("Void threshold", f"{float(void_thresh):g}  HU"))

        y = 0.66
        import textwrap
        for k, v in rows:
            fig.text(0.08, y, k, fontsize=10, color=_DIM)
            # Wrap long values so they never run into the key-results column.
            for j, chunk in enumerate(textwrap.wrap(str(v), 34) or [""]):
                fig.text(0.30, y, chunk, fontsize=10, color=_INK,
                         family="monospace")
                y -= 0.038
            y -= 0.007

        # key results box
        key = []
        if porosity is not None:
            key.append(f"Porosity (voxel fraction) : {porosity * 100:.3f} %")
        if void_result is not None and void_result.voids:
            key.append(f"Closed voids detected     : {void_result.n_voids}")
            key.append(f"Median void eq. diameter  : {void_result.d_median_mm:.4f} mm")
        if E_map is not None:
            key.append(f"E-map range               : "
                       f"{float(E_map.min()):.1f} – {float(E_map.max()):.1f} MPa")
            key.append(f"E-map mean ± std          : "
                       f"{float(E_map.mean()):.1f} ± {float(E_map.std()):.1f} MPa")
        if key:
            fig.text(0.60, 0.66, "Key results", fontsize=12,
                     fontweight="bold", color=_INK)
            yy = 0.61
            for line in key:
                fig.text(0.60, yy, line, fontsize=10, color=_TEAL,
                         family="monospace")
                yy -= 0.042
        _footer(fig, page)
        pdf.savefig(fig)

        # ── page 2 : orthogonal slices ────────────────────────────────
        if gray is not None:
            _stage("slice snapshots")
            page += 1
            fig = _new_page()
            fig.suptitle("Orthogonal Mid-Slices", fontsize=14,
                         fontweight="bold", color=_INK)
            axes = fig.subplots(1, 3)
            for ax, (title, sl) in zip(axes, _mid_slices(gray)):
                ax.imshow(_window(sl, ww, wc), cmap="gray", aspect="equal")
                ax.set_title(title, fontsize=9, color=_DIM, family="monospace")
                ax.axis("off")
            if ww and wc is not None:
                fig.text(0.5, 0.06, f"Window  WW={ww:g}  WC={wc:g}",
                         ha="center", fontsize=8, color=_DIM)
            _footer(fig, page)
            pdf.savefig(fig)

        # ── page 3 : histogram ────────────────────────────────────────
        if gray is not None:
            _stage("histogram")
            page += 1
            flat = np.asarray(gray, dtype=np.float32).ravel()
            if flat.size > 8_000_000:
                flat = flat[:: flat.size // 8_000_000]
            fig = _new_page()
            fig.suptitle("Intensity Distribution", fontsize=14,
                         fontweight="bold", color=_INK)
            ax1, ax2 = fig.subplots(1, 2)
            for ax in (ax1, ax2):
                _style_ax(ax)
            counts, edges = np.histogram(flat, bins=256)
            centers = (edges[:-1] + edges[1:]) / 2
            ax1.bar(centers, counts, width=edges[1] - edges[0],
                    color=_ACCENT, alpha=0.85, linewidth=0)
            if void_thresh is not None and hu_vol is not None:
                pass  # threshold is in HU domain; raw histogram shown as-is
            ax1.set_xlabel("Intensity (raw)", fontsize=9, color=_DIM)
            ax1.set_ylabel("Voxel count", fontsize=9, color=_DIM)

            means = [float(gray[:, :, z].mean())
                     for z in range(0, gray.shape[2],
                                    max(1, gray.shape[2] // 400))]
            ax2.plot(np.linspace(0, gray.shape[2] - 1, len(means)), means,
                     color=_TEAL, lw=1.5)
            ax2.set_xlabel("Z slice index", fontsize=9, color=_DIM)
            ax2.set_ylabel("Mean intensity", fontsize=9, color=_DIM)
            _footer(fig, page)
            pdf.savefig(fig)

        # ── page 4 : phases ───────────────────────────────────────────
        if labels is not None and phase_stats:
            _stage("phase statistics")
            page += 1
            fig = _new_page()
            fig.suptitle("Phase Segmentation", fontsize=14,
                         fontweight="bold", color=_INK)
            gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1],
                                  left=0.05, right=0.96, top=0.86, bottom=0.10)

            ax = fig.add_subplot(gs[0])
            z = labels.shape[2] // 2
            from matplotlib.colors import ListedColormap
            ax.imshow(np.rot90(labels[:, :, z]),
                      cmap=ListedColormap(_PHASE_COLORS),
                      vmin=0, vmax=2, interpolation="nearest", aspect="equal")
            ax.set_title(f"Axial mid-slice  z={z}   "
                         f"(blue=void, green=matrix, orange=aggregate)",
                         fontsize=8, color=_DIM)
            ax.axis("off")

            axt = fig.add_subplot(gs[1])
            axt.axis("off")
            cols = ["Phase", "Voxels", "Vol %", "HU mean±std", "E mean [MPa]"]
            cell_rows = []
            for name, s in phase_stats.items():
                cell_rows.append([
                    name, f"{s['voxels']:,}",
                    f"{s['vol_frac'] * 100:.2f}",
                    f"{s['HU_mean']:.0f} ± {s['HU_std']:.0f}",
                    f"{s['E_mean_MPa']:.1f}",
                ])
            tbl = axt.table(cellText=cell_rows, colLabels=cols,
                            loc="upper center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1, 1.6)
            for (r, _c), cell in tbl.get_celld().items():
                cell.set_edgecolor(_GRID)
                if r == 0:
                    cell.set_text_props(fontweight="bold", color="white")
                    cell.set_facecolor(_ACCENT)
            if porosity is not None:
                axt.text(0.5, 0.35, f"Porosity: {porosity * 100:.3f} %",
                         ha="center", fontsize=13, color=_TEAL,
                         fontweight="bold", transform=axt.transAxes)
            _footer(fig, page)
            pdf.savefig(fig)

        # ── page 5 : void analytics ───────────────────────────────────
        if void_result is not None and void_result.voids:
            _stage("void analytics")
            page += 1
            r = void_result
            d = r.diameters_mm()
            v = r.volumes_mm3()
            fig = _new_page()
            fig.suptitle("Void / Pore Analytics", fontsize=14,
                         fontweight="bold", color=_INK)
            gs = fig.add_gridspec(2, 2, left=0.07, right=0.96,
                                  top=0.86, bottom=0.09, hspace=0.45,
                                  wspace=0.28, height_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0]); _style_ax(ax1)
            bins = min(60, max(10, int(np.sqrt(d.size))))
            ax1.hist(d, bins=bins, color=_ACCENT, alpha=0.85, linewidth=0)
            ax1.axvline(r.d_median_mm, color=_TEAL, lw=1.2, ls="--")
            ax1.set_xlabel("Equivalent diameter (mm)", fontsize=8, color=_DIM)
            ax1.set_ylabel("Void count", fontsize=8, color=_DIM)
            ax1.set_title("Size distribution", fontsize=9, color=_INK)

            ax2 = fig.add_subplot(gs[0, 1]); _style_ax(ax2)
            order = np.argsort(d)
            ax2.plot(d[order], np.cumsum(v[order]) / v.sum() * 100.0,
                     color=_GREEN, lw=1.6)
            ax2.set_xlabel("Equivalent diameter (mm)", fontsize=8, color=_DIM)
            ax2.set_ylabel("Cumulative void volume (%)", fontsize=8, color=_DIM)
            ax2.set_title("Cumulative volume vs size", fontsize=9, color=_INK)
            ax2.set_ylim(0, 105)

            # summary text
            ax3 = fig.add_subplot(gs[1, 0]); ax3.axis("off")
            from .porosity import summary_lines
            ax3.text(0.0, 0.95, "\n".join(summary_lines(r)),
                     fontsize=9, color=_INK, family="monospace",
                     va="top", transform=ax3.transAxes)

            # top-10 table
            ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
            cols = ["#", "Voxels", "Vol mm³", "Eq.d mm", "Sph.", "Elong."]
            cell_rows = []
            for i, void in enumerate(r.voids[:10]):
                sph = "—" if np.isnan(void.sphericity) else f"{void.sphericity:.3f}"
                cell_rows.append([
                    str(i + 1), f"{void.voxels:,}", f"{void.volume_mm3:.4g}",
                    f"{void.eq_diam_mm:.4f}", sph, f"{void.elongation:.2f}",
                ])
            tbl = ax4.table(cellText=cell_rows, colLabels=cols,
                            loc="upper center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.35)
            for (row, _c), cell in tbl.get_celld().items():
                cell.set_edgecolor(_GRID)
                if row == 0:
                    cell.set_text_props(fontweight="bold", color="white")
                    cell.set_facecolor(_TEAL)
            ax4.set_title("Ten largest voids", fontsize=9, color=_INK)
            _footer(fig, page)
            pdf.savefig(fig)

        # ── page 6 : E-map ────────────────────────────────────────────
        if E_map is not None:
            _stage("E-map")
            page += 1
            fig = _new_page()
            fig.suptitle("Young's Modulus Map", fontsize=14,
                         fontweight="bold", color=_INK)
            axes = fig.subplots(1, 3)
            ims = None
            for ax, (title, sl) in zip(axes, _mid_slices(E_map)):
                ims = ax.imshow(sl, cmap="viridis", aspect="equal")
                ax.set_title(title, fontsize=9, color=_DIM, family="monospace")
                ax.axis("off")
            if ims is not None:
                cb = fig.colorbar(ims, ax=list(axes), shrink=0.7,
                                  pad=0.02, aspect=30)
                cb.set_label("E  (MPa)", fontsize=9, color=_DIM)
                cb.ax.tick_params(labelsize=8, colors=_DIM)
            fig.text(0.5, 0.06,
                     f"E: min {float(E_map.min()):.1f}   "
                     f"mean {float(E_map.mean()):.1f}   "
                     f"max {float(E_map.max()):.1f}   "
                     f"std {float(E_map.std()):.1f}   MPa",
                     ha="center", fontsize=9, color=_TEAL,
                     family="monospace")
            _footer(fig, page)
            pdf.savefig(fig)

        info = pdf.infodict()
        info["Title"] = f"CT Analysis Report — {file_name}"
        info["Creator"] = "NIfTI Tool Suite"

    return page
