"""Beam-hardening tab: measure the cupping profile and correct for it.

The tab shows three things:

1. mean solid grey value against depth below the specimen surface, with
   the fitted polynomial and the corrected profile on top;
2. the voxel count per depth bin, so a fit driven by few deep voxels is
   visible;
3. the same slice before and after correction, on identical grey limits.

The cupping index (how much brighter the surface layer is than the core) is
reported before and after, which shows whether the correction worked and
whether it overshot.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget,
)

from ..config import ACCENT, ACCENT2, BG, BORDER, ERR, PANEL2, TEAL, TEXT, TEXT_DIM
from ..deps import HAS_MPL, np
from .figure_viewer import click_hint, enable_click_to_enlarge
from .widgets import styled_btn

#: A solid/pore threshold discarding more than this share of the specimen is
#: rejected. Pores are a minority phase, so such a cut removes material.
MAX_PORE_FRACTION = 0.25


class BeamHardeningTabMixin:
    """Adds the Beam Hardening tab to :class:`NiftiApp`."""

    def _build_beam_hardening_tab(self, parent):
        self._bh_fit = None
        self._bh_fit_info = None
        self._bh_coarse_mask = None
        self._bh_stride = 1

        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QWidget(parent)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(8, 6, 8, 6)
        bar_lay.setSpacing(6)
        bar_lay.addWidget(styled_btn(
            bar, "▶ Measure Cupping", self._do_beam_hardening_fit, accent=True,
        ))
        bar_lay.addWidget(styled_btn(
            bar, "Apply Correction && Save...",
            self._do_beam_hardening_apply, teal=True,
        ))
        bar_lay.addWidget(styled_btn(
            bar, "Save fit as JSON...", self._export_beam_hardening_json,
            small=True,
        ))
        bar_lay.addStretch(1)
        root.addWidget(bar)

        self._bh_summary = QLabel(
            "No measurement yet. Set the options in the Beam Hardening tool "
            "panel on the left, then click  ▶ Measure Cupping.", parent,
        )
        self._bh_summary.setFont(QFont("Consolas", 9))
        self._bh_summary.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2}; padding: 6px 10px;"
        )
        self._bh_summary.setWordWrap(True)
        root.addWidget(self._bh_summary)

        if HAS_MPL:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            holder = QWidget(parent)
            h_lay = QVBoxLayout(holder)
            h_lay.setContentsMargins(0, 0, 0, 0)
            self._bh_fig = Figure(facecolor=BG)
            gs = self._bh_fig.add_gridspec(2, 2, height_ratios=[3, 2])
            self._bh_ax_curve = self._bh_fig.add_subplot(gs[0, :])
            self._bh_ax_before = self._bh_fig.add_subplot(gs[1, 0])
            self._bh_ax_after = self._bh_fig.add_subplot(gs[1, 1])
            for ax in (self._bh_ax_curve, self._bh_ax_before, self._bh_ax_after):
                ax.set_facecolor(PANEL2)
            self._bh_canvas = FigureCanvasQTAgg(self._bh_fig)
            h_lay.addWidget(self._bh_canvas)

            h_lay.addWidget(click_hint(holder))
            enable_click_to_enlarge(self._bh_canvas, self._bh_fig, self)
            root.addWidget(holder, 1)
            self._clear_bh_axes()
            self._bh_canvas.draw()
        else:
            lbl = QLabel("matplotlib not installed", parent)
            lbl.setStyleSheet(f"color: {ERR}; background-color: {BG};")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(lbl, 1)

    # --- options -------------------------------------------------------

    def _bh_options(self) -> dict:
        """Read the Beam Hardening tool panel."""
        def _int(widget, default):
            try:
                return int(float(widget.text()))
            except Exception:
                return default

        def _float(widget, default):
            try:
                return float(widget.text())
            except Exception:
                return default

        return {
            "poly_degree": max(1, min(8, _int(self._bh_poly_deg, 4))),
            "bin_mm": max(1e-4, _float(self._bh_bin_mm, 0.25)),
            "stride": max(0, _int(self._bh_stride_widget, 0)),
            "border_is_surface": bool(self._bh_border_check.isChecked()),
            "remove_background": bool(self._bh_removebg_check.isChecked()),
            "out_dtype": self._bh_dtype_combo.currentText().strip(),
        }

    # --- measure -------------------------------------------------------

    def _do_beam_hardening_fit(self):
        """Measure the depth profile and fit the cupping curve."""
        if not self._require_img():
            return
        opts = self._bh_options()

        def _run():
            try:
                from ..core import beam_hardening as bh
                from ..core import pore_threshold as pt
                from ..utils import available_ram_mb, OperationCancelled

                self._log_sep("Beam Hardening: measure cupping")
                self._set_status("Reading the analysis grid...", busy=True)

                vol = self._get_gray_lazy()
                spacing = tuple(
                    float(s) for s in self._img.header.get_zooms()[:3]
                )
                stride = opts["stride"] or bh.choose_stride(
                    vol.shape, available_ram_mb())

                def _read_prog(done, total):
                    if self.cancel_requested():
                        raise OperationCancelled()
                    self._set_status(
                        f"Reading analysis grid...  slice {done}/{total}",
                        busy=True)

                coarse = bh.coarse_volume(vol, stride, progress=_read_prog)
                if stride > 1:
                    self._append_log(
                        f"  Analysis grid: every {stride}. voxel "
                        f"({coarse.shape[0]}x{coarse.shape[1]}x"
                        f"{coarse.shape[2]}). The depth field is smooth, so "
                        f"this does not bias the profile; it only limits "
                        f"depth resolution to about "
                        f"{stride * max(spacing):.4f} mm.", 'dim')

                # Specimen vs air, using the isodata-refined Otsu split. A
                # plain Otsu argmax sits at the edge of the empty valley when
                # the air population is large.
                self._set_status("Finding the specimen...", busy=True)
                from ..core.segmentation import otsu_threshold as isodata_otsu

                flat = coarse.ravel()
                if flat.size > 4_000_000:
                    flat = flat[:: flat.size // 4_000_000]
                t_specimen = float(isodata_otsu(flat))

                # Solid vs pore, inside the specimen only. A pore is much
                # darker than solid material, so the split is taken at the
                # histogram valley when a pore population exists. If there is
                # none the threshold stays at the specimen level. Cutting at
                # "mode - k sigma" would be wrong here, because the cupping
                # under measurement is what widens the solid peak, and the
                # cut would remove the dark deep interior the profile needs.
                interior = coarse[coarse >= t_specimen]
                if interior.size > 4_000_000:
                    interior = interior[:: interior.size // 4_000_000]
                th = pt.auto_threshold(interior, k_sigma=3.0, method="auto")

                # A candidate discarding a large share of the specimen is
                # cutting into the material rather than separating pores.
                t_solid = float(t_specimen)
                why = ("no separated pore population, so all specimen "
                       "voxels are used")
                if th.t_valley is not None:
                    excluded = float((interior < th.t_valley).mean())
                    if excluded <= MAX_PORE_FRACTION:
                        t_solid = float(th.t_valley)
                        why = (f"histogram valley between the pore and solid "
                               f"peaks, excluding {excluded * 100:.1f} % of "
                               f"the specimen")
                    else:
                        why = (f"the valley at {th.t_valley:.4g} would "
                               f"exclude {excluded * 100:.0f} % of the "
                               f"specimen; that is material, not pores, so "
                               f"all specimen voxels are used")

                self._append_log(
                    f"  Specimen threshold {t_specimen:.4g}   "
                    f"solid threshold {t_solid:.4g}  ({why})", 'dim')
                self._append_log(
                    "  Pores are kept out of the profile so a "
                    "depth-dependent pore fraction cannot be read as a "
                    "grey-value gradient.", 'dim')

                self._set_status("Finding the specimen surface...", busy=True)
                mask = bh.specimen_mask(coarse, t_specimen)

                self._set_status("Measuring the depth profile...", busy=True)
                fit = bh.fit_cupping(
                    vol, spacing,
                    specimen_threshold=t_specimen,
                    solid_threshold=t_solid,
                    stride=stride,
                    poly_degree=opts["poly_degree"],
                    bin_mm=opts["bin_mm"],
                    border_is_surface=opts["border_is_surface"],
                    coarse=coarse,
                )

                # Publish the mask and stride before the fit: Apply becomes
                # usable as soon as _bh_fit is set and needs all three.
                self._bh_coarse_mask = mask
                self._bh_stride = stride
                self._bh_fit_info = fit.to_dict()
                self._bh_fit = fit

                for line in bh.summary_lines(fit):
                    self._append_log("  " + line, 'teal')

                self.after(0, self._show_bh_fit, fit)
                self.after(0, self._show_tab, 'beam_hardening')
                self._set_status("Cupping measured.", busy=False)
            except Exception as ex:
                from ..utils import OperationCancelled
                if isinstance(ex, OperationCancelled):
                    self._append_log("  Measurement stopped by user.", 'warn')
                    self._set_status("Stopped.", busy=False)
                else:
                    self._append_log(f"  Beam hardening error: {ex}", 'err')
                    self._set_status("Beam hardening error.", busy=False)

        self._run_task("Measure cupping", _run)

    # --- apply ---------------------------------------------------------

    def _do_beam_hardening_apply(self):
        """Write the corrected volume, streaming it slice by slice."""
        if not self._require_img():
            return
        if self._bh_fit is None:
            QMessageBox.warning(
                self, "No fit yet",
                "Click  ▶ Measure Cupping  first: the correction is applied "
                "from the fitted depth profile.")
            return
        opts = self._bh_options()
        stem = self._nifti_stem()
        out = self._ask_save_path(f"{stem}_bhcorrected.nii", ext=".nii",
                                  ftypes=[("NIfTI", "*.nii"), ("All", "*.*")])
        if not out:
            return

        fit = self._bh_fit
        mask = self._bh_coarse_mask
        stride = self._bh_stride

        def _run():
            try:
                from ..core import beam_hardening as bh

                self._log_sep("Beam Hardening: apply correction")
                self._set_status("Writing corrected volume...", busy=True)

                def _prog(done, total):
                    self._set_status(
                        f"Writing corrected volume...  slice {done}/{total}",
                        busy=True)

                info = bh.apply_correction(
                    self._img, self._get_gray_lazy(), out, fit,
                    coarse_mask=mask, stride=stride,
                    remove_background=opts["remove_background"],
                    out_dtype=opts["out_dtype"],
                    progress=_prog,
                    cancelled=self.cancel_requested,
                )

                self._append_log(
                    f"  Saved → {Path(out).name}   "
                    f"({info['bytes'] / 1e6:.0f} MB, {info['dtype']})", 'ok')
                self._append_log(
                    "  The source affine, orientation and voxel spacing were "
                    "copied unchanged, so the corrected volume stays "
                    "registered with the original.", 'dim')
                if opts["remove_background"]:
                    self._append_log(
                        "  Background outside the specimen was set to 0.",
                        'dim')
                if opts["out_dtype"] == "int16":
                    self._append_log(
                        "  Written as int16: values were rounded and clipped "
                        "to -32768 to 32767. Use float32 if the corrected range "
                        "may leave that window.", 'warn')

                self.after(0, self._show_bh_slices,
                           info["demo_before"], info["demo_after"],
                           info["demo_z"])
                self._set_status("Correction written.", busy=False)
                self.after(0, self._ask_load_after_save,
                           "Beam-hardening-corrected volume saved to:",
                           Path(out))
            except Exception as ex:
                from ..utils import OperationCancelled
                if isinstance(ex, OperationCancelled):
                    self._append_log("  Correction stopped by user.", 'warn')
                    self._set_status("Stopped.", busy=False)
                else:
                    self._append_log(f"  Correction error: {ex}", 'err')
                    self._set_status("Correction error.", busy=False)

        self._run_task("Apply beam-hardening correction", _run)

    def _export_beam_hardening_json(self):
        """Save the fit so the same correction can be reapplied elsewhere."""
        if self._bh_fit is None:
            QMessageBox.warning(self, "No fit yet",
                                "Measure the cupping profile first.")
            return
        import json

        stem = self._nifti_stem()
        out = self._ask_save_path(
            f"{stem}_beam_hardening_fit.json", ext=".json",
            ftypes=[("JSON", "*.json"), ("All", "*.*")])
        if not out:
            return
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self._bh_fit.to_dict(), f, indent=2)
            self._append_log(f"  Fit saved → {Path(out).name}", 'ok')
        except Exception as ex:
            self._append_log(f"  {ex}", 'err')

    # --- drawing -------------------------------------------------------

    def _clear_bh_axes(self):
        if not HAS_MPL:
            return
        for ax in (self._bh_ax_curve, self._bh_ax_before, self._bh_ax_after):
            ax.clear()
            ax.set_facecolor(PANEL2)
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
        self._bh_ax_curve.set_xlabel(
            "Depth below the specimen surface (mm)", color=TEXT_DIM, fontsize=8)
        self._bh_ax_curve.set_ylabel(
            "Mean solid grey value", color=TEXT_DIM, fontsize=8)
        for ax, title in ((self._bh_ax_before, "Before"),
                          (self._bh_ax_after, "After correction")):
            ax.set_title(title, color=TEXT_DIM, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    def _show_bh_fit(self, fit):
        before = fit.cupping_index_before_pct
        after = fit.cupping_index_after_pct
        verdict = (
            "negligible beam hardening in this scan"
            if abs(before) < 1.0 else
            f"the surface layer reads {before:+.1f} % brighter than the core"
        )
        self._bh_summary.setText(
            f"cupping index: {before:+.2f} %  →  {after:+.2f} % after "
            f"correction    ({verdict})        "
            f"depth fitted: {fit.fit_depth_mm[0]:.3f}-{fit.fit_depth_mm[1]:.3f} mm"
            f"    degree {fit.poly_degree}    I_ref {fit.i_ref:.1f}"
            f"    {fit.n_solid_samples:,} solid samples"
        )

        if not HAS_MPL:
            return
        self._clear_bh_axes()
        ax = self._bh_ax_curve
        d, p, f, c = fit.depth_mm, fit.profile, fit.fitted, fit.corrected

        ax.plot(d, p, color=ACCENT, lw=1.8, label="Measured profile")
        ax.plot(d, f, color=TEXT, lw=1.2, ls="--",
                label=f"Polynomial fit (degree {fit.poly_degree})")
        ax.plot(d, c, color=ACCENT2, lw=1.8, label="After correction")
        ax.axhline(fit.i_ref, color=TEAL, lw=1.0, ls=":",
                   label=f"Interior reference {fit.i_ref:.0f}")
        ax.set_xlabel("Depth below the specimen surface (mm)",
                      color=TEXT_DIM, fontsize=8)
        ax.set_ylabel("Mean solid grey value", color=TEXT_DIM, fontsize=8)
        ax.set_title(
            f"Beam hardening: cupping index {fit.cupping_index_before_pct:+.1f} %"
            f"  →  {fit.cupping_index_after_pct:+.1f} %",
            color=TEXT, fontsize=9,
        )
        ax.grid(alpha=0.25, color=BORDER)
        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER)

        # Bin population on a twin axis, so a fit driven by a few deep
        # voxels stays visible.
        twin = ax.twinx()
        twin.bar(d, fit.counts, width=max(1e-6, (d[1] - d[0]) if d.size > 1
                                          else 0.1),
                 color=BORDER, alpha=0.35, linewidth=0)
        twin.set_ylabel("Voxels per depth bin", color=TEXT_DIM, fontsize=8)
        twin.tick_params(colors=TEXT_DIM, labelsize=7)
        twin.set_yscale("log")

        self._bh_fig.tight_layout(pad=1.2)
        self._bh_canvas.draw()

    def _show_bh_slices(self, before, after, z):
        if not HAS_MPL or before is None or after is None:
            return
        lo, hi = np.percentile(before, [1, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        for ax, data, title in (
            (self._bh_ax_before, before, f"Before  (z = {z})"),
            (self._bh_ax_after, after, "After correction"),
        ):
            ax.clear()
            ax.set_facecolor(PANEL2)
            ax.imshow(np.rot90(data), cmap="gray", vmin=lo, vmax=hi,
                      aspect="equal")
            ax.set_title(title, color=TEXT, fontsize=9)
            ax.axis("off")
        self._bh_fig.tight_layout(pad=1.2)
        self._bh_canvas.draw()
