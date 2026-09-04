"""One-click PDF analysis report.

Collects what the GUI has computed for the loaded volume into a
print-friendly PDF (white background, A4 landscape):

1. Title page with file info and key results.
2. Orthogonal mid-slices with the current window applied.
3. Intensity histogram and per-slice mean curve.
4. Void analytics: size distribution, cumulative volume, largest voids.
5. Pore threshold and the sensitivity of the porosity to it.
6. Beam-hardening cupping profile, fit and correction.
7. Binarisation histogram with the cut, and the indicator field.

Every section after the first is included only when the corresponding
analysis has been run.

Takes numpy arrays and dicts, builds matplotlib ``Figure`` objects on the
Agg backend so it can be called from a worker thread, and writes with
``PdfPages``.
"""

from __future__ import annotations

import datetime

from ..deps import np, HAS_MPL

# Print palette, independent of the app theme.
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
    fig.text(0.985, 0.015, f"NIfTI Tool Suite, page {page_no}",
             ha="right", va="bottom", fontsize=7, color=_DIM)

def _window(img2d, ww, wc):
    """Apply CT windowing; fall back to p1-p99 stretch."""
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
    void_result=None,
    void_thresh=None,
    bin_result=None,          # niftitool.core.binarize.BinarizeResult
    bh_fit=None,             # niftitool.core.beam_hardening.CuppingFit
    pore_threshold=None,     # niftitool.core.pore_threshold.ThresholdResult
    pore_sensitivity: dict | None = None,
    progress=None,           # fn(str), stage callback for the UI log
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

        # Page 1: title
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
        if void_thresh is not None:
            rows.append(("Void threshold", f"{float(void_thresh):g}  grey value"))

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

        key = []
        if void_result is not None:
            basis = "of specimen" if void_result.specimen_voxels else "of array"
            key.append(f"Porosity {basis:<16}: "
                       f"{void_result.porosity_pct:.3f} %")
            if void_result.voids:
                key.append(f"Closed voids detected     : {void_result.n_voids}")
                key.append(f"Median void eq. diameter  : "
                           f"{void_result.d_median_mm:.4f} mm")
        if pore_threshold is not None:
            key.append(f"Pore threshold            : "
                       f"{pore_threshold.T:.1f} ({pore_threshold.method_used})")
        if bh_fit is not None:
            key.append(f"Cupping index             : "
                       f"{bh_fit.cupping_index_before_pct:+.2f} % → "
                       f"{bh_fit.cupping_index_after_pct:+.2f} %")
        if bin_result is not None:
            key.append(f"Material fraction of box  : "
                       f"{bin_result.material_fraction_of_box * 100:.2f} %")
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

        # Page 2: orthogonal slices
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

        # Page 3: histogram
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
            if void_thresh is not None:
                ax1.axvline(float(void_thresh), color="#C0392B", lw=1.6,
                            label=f"threshold {float(void_thresh):.0f}")
                ax1.legend(fontsize=8)
            ax1.set_xlabel("Grey value", fontsize=9, color=_DIM)
            ax1.set_ylabel("Voxel count", fontsize=9, color=_DIM)

            means = [float(gray[:, :, z].mean())
                     for z in range(0, gray.shape[2],
                                    max(1, gray.shape[2] // 400))]
            ax2.plot(np.linspace(0, gray.shape[2] - 1, len(means)), means,
                     color=_TEAL, lw=1.5)
            ax2.set_xlabel("Z slice index", fontsize=9, color=_DIM)
            ax2.set_ylabel("Mean grey value", fontsize=9, color=_DIM)
            _footer(fig, page)
            pdf.savefig(fig)

        # Page 4: void analytics
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

            ax3 = fig.add_subplot(gs[1, 0]); ax3.axis("off")
            from .porosity import summary_lines
            ax3.text(0.0, 0.95, "\n".join(summary_lines(r)),
                     fontsize=9, color=_INK, family="monospace",
                     va="top", transform=ax3.transAxes)

            ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
            cols = ["#", "Voxels", "Vol mm³", "Eq.d mm", "Sph.", "Elong."]
            cell_rows = []
            for i, void in enumerate(r.voids[:10]):
                sph = "n/a" if np.isnan(void.sphericity) else f"{void.sphericity:.3f}"
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

        # Page: automatic pore threshold and its sensitivity
        if pore_threshold is not None:
            _stage("pore threshold")
            page += 1
            fig = _new_page()
            fig.suptitle("Pore Threshold and Its Sensitivity", fontsize=14,
                         fontweight="bold", color=_INK)
            axes = fig.subplots(1, 2)
            ax0, ax1 = axes
            _style_ax(ax0); _style_ax(ax1)

            t = pore_threshold
            xs = np.linspace(t.solid_mode - 6 * t.solid_sigma,
                             t.solid_mode + 4 * t.solid_sigma, 500)
            sig = max(t.solid_sigma, 1e-9)
            ax0.plot(xs, np.exp(-0.5 * ((xs - t.solid_mode) / sig) ** 2),
                     color=_DIM, lw=1.2, ls="-.",
                     label=f"solid peak {t.solid_mode:.0f} +/- {sig:.0f}")
            ax0.axvline(t.t_sigma, color=_ACCENT, ls="--", lw=1.3,
                        label=f"{t.k_sigma:g}-sigma = {t.t_sigma:.0f}")
            ax0.axvline(t.t_otsu, color=_INK, ls=":", lw=1.3,
                        label=f"Otsu = {t.t_otsu:.0f}")
            if t.t_valley is not None:
                ax0.axvline(t.t_valley, color=_TEAL, ls="--", lw=1.3,
                            label=f"valley = {t.t_valley:.0f}")
            ax0.axvline(t.T, color="#C0392B", lw=2.2,
                        label=f"chosen = {t.T:.0f} ({t.method_used})")
            ax0.set_xlabel("Grey value", fontsize=9, color=_DIM)
            ax0.set_ylabel("Solid-peak model", fontsize=9, color=_DIM)
            ax0.set_title("Threshold candidates", fontsize=9, color=_INK)
            ax0.legend(fontsize=7)

            if pore_sensitivity:
                keys = [k for k in pore_sensitivity if k != "chosen"]
                vals = [pore_sensitivity[k] for k in keys]
                ax1.bar(range(len(keys)), vals, color=_TEAL)
                ax1.set_xticks(range(len(keys)))
                ax1.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
                ax1.set_ylabel("Pore fraction of the interior sample (%)",
                               fontsize=9, color=_DIM)
                ax1.set_title("How much the answer depends on the threshold",
                              fontsize=9, color=_INK)
                for i, v in enumerate(vals):
                    ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom",
                             fontsize=7, color=_INK)
            else:
                ax1.text(0.5, 0.5, "no sensitivity sweep recorded",
                         transform=ax1.transAxes, ha="center", color=_DIM,
                         fontsize=9)

            fig.text(0.06, 0.06,
                     "A porosity figure is only meaningful together with the "
                     "threshold it was measured at. The bars show what the "
                     "same volume would report under each candidate "
                     "threshold.",
                     fontsize=8, color=_DIM, wrap=True)
            _footer(fig, page)
            pdf.savefig(fig)

        # Page: beam-hardening correction
        if bh_fit is not None and getattr(bh_fit, "depth_mm", None) is not None:
            _stage("beam hardening")
            page += 1
            fig = _new_page()
            fig.suptitle("Beam-Hardening (Cupping) Correction", fontsize=14,
                         fontweight="bold", color=_INK)
            ax = fig.add_axes([0.10, 0.42, 0.82, 0.42])
            _style_ax(ax)
            d = bh_fit.depth_mm
            ax.plot(d, bh_fit.profile, color=_ACCENT, lw=1.8,
                    label="measured profile")
            ax.plot(d, bh_fit.fitted, color=_INK, lw=1.2, ls="--",
                    label=f"polynomial fit (degree {bh_fit.poly_degree})")
            ax.plot(d, bh_fit.corrected, color=_TEAL, lw=1.8,
                    label="after correction")
            ax.set_xlabel("Depth below the specimen surface (mm)",
                          fontsize=9, color=_DIM)
            ax.set_ylabel("Mean solid grey value", fontsize=9, color=_DIM)
            ax.legend(fontsize=8)

            lines = [
                f"cupping index      {bh_fit.cupping_index_before_pct:+.2f} %"
                f"  ->  {bh_fit.cupping_index_after_pct:+.2f} %",
                f"depth fitted       {bh_fit.fit_depth_mm[0]:.4f} - "
                f"{bh_fit.fit_depth_mm[1]:.4f} mm",
                f"interior reference {bh_fit.i_ref:.2f}",
                f"solid samples      {bh_fit.n_solid_samples:,} voxels "
                f"(analysis stride x{bh_fit.stride})",
                f"depth reference    "
                + ("the array border counts as specimen surface"
                   if bh_fit.border_is_surface
                   else "only real material/air interfaces"),
            ]
            fig.text(0.10, 0.30, "\n".join(lines), fontsize=9, color=_INK,
                     family="monospace", va="top")
            fig.text(0.10, 0.12,
                     "Depth is the shortest distance from each voxel to the "
                     "specimen surface, computed with a Euclidean distance\n"
                     "transform at the true voxel spacing. It is therefore "
                     "valid for any specimen shape and any anisotropic grid,\n"
                     "not only for a block cropped flush to the field of view.",
                     fontsize=8, color=_DIM, va="top")
            _footer(fig, page)
            pdf.savefig(fig)

        # Page: binarisation for the finite cell method
        if bin_result is not None:
            _stage("binarisation")
            page += 1
            fig = _new_page()
            fig.suptitle("Binarisation \u2014 FCM Indicator Field",
                         fontsize=14, fontweight="bold", color=_INK)
            ax0, ax1 = fig.subplots(1, 2)
            _style_ax(ax0)

            sample = getattr(bin_result, "sample", None)
            if sample is not None and np.size(sample):
                values = np.asarray(sample).ravel()
                values = values[np.isfinite(values)]
                ax0.hist(values, bins=400, color=_DIM, log=True)
                ax0.axvline(bin_result.threshold, color="#C0392B", lw=1.8,
                            label=f"{bin_result.method} = "
                                  f"{bin_result.threshold:.1f}")
                ax0.legend(loc="upper left", fontsize=8)
            ax0.set_xlabel("Grey value", fontsize=9, color=_DIM)
            ax0.set_ylabel("Voxel count (log)", fontsize=9, color=_DIM)
            ax0.set_title("Histogram and the cut", fontsize=9, color=_INK)

            preview = getattr(bin_result, "preview_mask", None)
            if preview is not None:
                ax1.imshow(np.rot90(preview), cmap="gray",
                           interpolation="nearest", aspect="equal")
                ax1.set_title("Indicator field (white = material)",
                              fontsize=9, color=_INK)
            ax1.set_xticks([]); ax1.set_yticks([])

            lines = [
                f"method              {bin_result.method}",
                f"threshold           {bin_result.threshold:.2f}",
                f"material fraction   "
                f"{bin_result.material_fraction_of_box * 100:.2f} % of the box",
            ]
            if bin_result.porosity_inside_specimen is not None:
                lines.append(
                    f"porosity            "
                    f"{bin_result.porosity_inside_specimen * 100:.2f} % "
                    f"inside the specimen envelope")
            if bin_result.material_components:
                lines.append(
                    f"material bodies     {bin_result.material_components}"
                    + (f" (kept the largest, removed "
                       f"{bin_result.removed_voxels:,} island voxels)"
                       if bin_result.removed_voxels else ""))
            fig.text(0.06, 0.10, "\n".join(lines), fontsize=9, color=_INK,
                     family="monospace", va="bottom")
            if bin_result.separability_warning:
                fig.text(0.06, 0.03,
                         "WARNING: " + bin_result.separability_warning,
                         fontsize=8, color="#C0392B", va="bottom", wrap=True)
            _footer(fig, page)
            pdf.savefig(fig)

        info = pdf.infodict()
        info["Title"] = f"CT Analysis Report: {file_name}"
        info["Creator"] = "NIfTI Tool Suite"

    return page
