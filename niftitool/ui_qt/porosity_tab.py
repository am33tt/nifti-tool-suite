"""Porosity tab: per-void connected-component analytics.

Runs :func:`niftitool.core.porosity.analyze_voids` on the void phase
(below the void threshold) and shows a population summary, a size
distribution with cumulative void-volume curve, a sortable table of the
largest voids and CSV export. The result is kept on ``self._void_result``
for the PDF report.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..config import ACCENT, ACCENT2, BG, BORDER, ERR, PANEL2, TEAL, TEXT, TEXT_DIM
from ..deps import HAS_MPL, np
from .figure_viewer import click_hint, enable_click_to_enlarge
from .widgets import styled_btn, styled_entry

_TABLE_MAX_ROWS = 500

#: The plots this tab can show, in selector order, as ``(key, label)``. One
#: is visible at a time so each gets the whole panel.
PORO_PANELS = (
    ("mask", "Void mask"),
    ("threshold", "Threshold"),
    ("size", "Size distribution"),
    ("cumulative", "Cumulative volume"),
)

_PORO_TAB_QSS = (
    f"QPushButton {{ background-color: transparent; color: {TEXT_DIM};"
    f"  border: none; padding: 5px 12px; font-weight: 600;"
    f"  border-bottom: 2px solid transparent; }}"
    f"QPushButton:hover {{ color: {TEXT}; }}"
    f"QPushButton:checked {{ color: {ACCENT};"
    f"  border-bottom: 2px solid {ACCENT}; }}"
)


class PorosityTabMixin:
    """Adds the Porosity tab to :class:`NiftiApp`."""

    def _build_porosity_tab(self, parent):
        self._void_result = None

        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Options row
        opts = QWidget(parent)
        opts_lay = QHBoxLayout(opts)
        opts_lay.setContentsMargins(8, 6, 8, 6)
        opts_lay.setSpacing(6)

        def _opt_label(text):
            lbl = QLabel(text, opts)
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
            return lbl

        opts_lay.addWidget(_opt_label("Min void size (vox):"))
        self._poro_min_vox = styled_entry(opts, width=6)
        self._poro_min_vox.setText("8")
        opts_lay.addWidget(self._poro_min_vox)

        opts_lay.addWidget(_opt_label("Connectivity:"))
        self._poro_conn = QComboBox(opts)
        self._poro_conn.addItems(["6 (faces)", "18 (+edges)", "26 (+corners)"])
        opts_lay.addWidget(self._poro_conn)

        self._poro_excl_border = QCheckBox("Exclude border air", opts)
        self._poro_excl_border.setChecked(True)
        opts_lay.addWidget(self._poro_excl_border)

        self._poro_specimen_only = QCheckBox("Specimen basis", opts)
        self._poro_specimen_only.setChecked(True)
        opts_lay.addWidget(self._poro_specimen_only)

        self._poro_sphericity = QCheckBox("Sphericity (slower)", opts)
        self._poro_sphericity.setChecked(True)
        opts_lay.addWidget(self._poro_sphericity)

        opts_lay.addWidget(_opt_label("Threshold:"))
        self._poro_thresh_method = QComboBox(opts)
        self._poro_thresh_method.addItems(["auto", "sigma", "otsu", "valley"])
        opts_lay.addWidget(self._poro_thresh_method)

        opts_lay.addWidget(_opt_label("k:"))
        self._poro_k_sigma = styled_entry(opts, width=4)
        self._poro_k_sigma.setText("3.0")
        opts_lay.addWidget(self._poro_k_sigma)

        opts_lay.addWidget(styled_btn(
            opts, "▶ Auto Threshold", self._do_auto_pore_threshold,
            teal=True,
        ))
        opts_lay.addWidget(styled_btn(
            opts, "▶ Analyze Voids", self._do_porosity, accent=True,
        ))
        opts_lay.addWidget(styled_btn(
            opts, "Export voids CSV...", self._export_voids_csv, small=True,
        ))
        opts_lay.addStretch(1)
        root.addWidget(opts)

        # Summary line
        self._poro_summary = QLabel(
            "No analysis yet. Click  ▶ Auto Threshold  to measure the "
            "pore threshold from the specimen interior, then "
            "▶ Analyze Voids.", parent,
        )
        self._poro_summary.setFont(QFont("Consolas", 9))
        self._poro_summary.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2}; padding: 6px 10px;"
        )
        self._poro_summary.setWordWrap(True)
        root.addWidget(self._poro_summary)

        # Plots / table splitter
        split = QSplitter(Qt.Orientation.Vertical, parent)
        split.setHandleWidth(6)
        split.setChildrenCollapsible(False)

        if HAS_MPL:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            plot_holder = QWidget(split)
            ph_lay = QVBoxLayout(plot_holder)
            ph_lay.setContentsMargins(0, 0, 0, 0)
            ph_lay.setSpacing(0)

            # One plot at a time. Four tiled into this space leaves every
            # axis too small to read a value off.
            selector = QWidget(plot_holder)
            sel_lay = QHBoxLayout(selector)
            sel_lay.setContentsMargins(8, 4, 8, 0)
            sel_lay.setSpacing(2)

            self._poro_stack = QStackedWidget(plot_holder)
            self._poro_figs: dict = {}
            self._poro_canvases: dict = {}
            self._poro_nav_buttons: dict = {}
            group = QButtonGroup(selector)
            group.setExclusive(True)
            self._poro_nav_group = group

            for index, (key, label) in enumerate(PORO_PANELS):
                fig = Figure(facecolor=BG)
                ax = fig.add_subplot(111)
                ax.set_facecolor(PANEL2)
                canvas = FigureCanvasQTAgg(fig)
                enable_click_to_enlarge(canvas, fig, self)
                self._poro_figs[key] = fig
                self._poro_canvases[key] = canvas
                self._poro_stack.addWidget(canvas)

                btn = QPushButton(label, selector)
                btn.setCheckable(True)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setFont(QFont("Segoe UI", 9))
                btn.setStyleSheet(_PORO_TAB_QSS)
                btn.clicked.connect(
                    lambda _c, i=index: self._poro_stack.setCurrentIndex(i))
                group.addButton(btn)
                sel_lay.addWidget(btn)
                self._poro_nav_buttons[key] = btn

            sel_lay.addWidget(click_hint(selector))
            sel_lay.addStretch(1)
            ph_lay.addWidget(selector)
            ph_lay.addWidget(self._poro_stack, 1)

            # Named aliases, so the drawing code addresses a plot by name
            # rather than by grid position.
            self._poro_ax0 = self._poro_figs["mask"].axes[0]
            self._poro_ax3 = self._poro_figs["threshold"].axes[0]
            self._poro_ax1 = self._poro_figs["size"].axes[0]
            self._poro_ax2 = self._poro_figs["cumulative"].axes[0]

            self._poro_nav_buttons["mask"].setChecked(True)
            self._poro_stack.setCurrentIndex(0)
            split.addWidget(plot_holder)
        else:
            lbl = QLabel("matplotlib not installed", split)
            lbl.setStyleSheet(f"color: {ERR}; background-color: {BG};")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            split.addWidget(lbl)

        self._poro_table = QTableWidget(split)
        self._poro_table.setColumnCount(8)
        self._poro_table.setHorizontalHeaderLabels([
            "#", "Voxels", "Volume mm³", "Eq. diam mm", "Sphericity",
            "Elongation", "Centroid (vox)", "Centroid (mm)",
        ])
        self._poro_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._poro_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._poro_table.setSortingEnabled(True)
        self._poro_table.setFont(QFont("Consolas", 9))
        self._poro_table.verticalHeader().setVisible(False)
        hdr = self._poro_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._poro_table.setStyleSheet(
            f"QTableWidget {{ background-color: #FFFFFF; color: {TEXT}; "
            f"border: 1px solid {BORDER}; gridline-color: {BORDER}; }}"
            f"QTableWidget::item:selected {{ background-color: {ACCENT}; "
            f"color: #FFFFFF; }}"
        )
        split.addWidget(self._poro_table)
        split.setSizes([420, 280])
        root.addWidget(split, 1)

    # Automatic pore threshold

    def _do_auto_pore_threshold(self):
        """Measure the pore threshold from the specimen interior.

        Sampling the whole volume fails because the surrounding air is
        usually the largest population in the histogram and pulls every
        automatic threshold towards it. This samples interior voxels only, at
        full resolution, and reports each candidate threshold with the
        porosity it produces, so the sensitivity is visible before use.
        """
        if not self._require_img():
            return
        method = self._poro_thresh_method.currentText().strip()
        try:
            k_sigma = float(self._poro_k_sigma.text())
        except Exception:
            k_sigma = 3.0

        def _run():
            try:
                from ..core import pore_threshold as pt
                from ..utils import OperationCancelled

                self._log_sep("Automatic pore threshold")
                self._set_status("Sampling the specimen interior...", busy=True)

                vol = self._get_gray_lazy()
                spacing = self._img.header.get_zooms()[:3]

                already_masked, why = pt.resolve_masking(vol, "auto")
                self._append_log(
                    f"  Background mode: "
                    f"{'file is already background-masked' if already_masked else 'raw scan, specimen detected per slice'}"
                    f"  ({why})", 'dim')

                def _prog(done, total):
                    if self.cancel_requested():
                        raise OperationCancelled()
                    self._set_status(
                        f"Scanning slices...  {done}/{total}", busy=True)

                z_lo, z_hi, _zs, _areas = pt.find_valid_zrange(
                    vol, already_masked, progress=_prog)
                self._append_log(
                    f"  Valid slice range: {z_lo} to {z_hi} of "
                    f"{vol.shape[2]}  (partial or empty end slices are "
                    f"excluded so they cannot bias the threshold)", 'dim')

                sample = pt.sample_interior(
                    vol, (z_lo, z_hi), already_masked, progress=_prog)
                result = pt.auto_threshold(sample, k_sigma=k_sigma,
                                           method=method)
                sensitivity = pt.porosity_sensitivity(sample, result)

                self._pore_threshold_result = result
                self._pore_threshold_info = dict(result.to_dict())
                self._pore_threshold_info.update({
                    "valid_z_range": [int(z_lo), int(z_hi)],
                    "background_already_masked": bool(already_masked),
                    "porosity_sensitivity_pct": sensitivity,
                })

                for line in pt.summary_lines(result, sensitivity, spacing):
                    self._append_log("  " + line, 'teal')

                # Feed the chosen value into the shared void threshold so the
                # void analysis and the report agree.
                self.after(0, lambda t=result.T:
                           self._void_thresh_var.set(f"{t:.1f}"))
                self.after(0, self._show_pore_threshold, result, sample,
                           sensitivity)
                self.after(0, self._show_tab, 'porosity')
                self._set_status("Pore threshold measured.", busy=False)
            except Exception as ex:
                from ..utils import OperationCancelled
                if isinstance(ex, OperationCancelled):
                    self._append_log("  Threshold search stopped by user.",
                                     'warn')
                    self._set_status("Stopped.", busy=False)
                else:
                    self._append_log(f"  Pore threshold error: {ex}", 'err')
                    self._set_status("Pore threshold error.", busy=False)

        self._run_task("Auto pore threshold", _run)

    def _show_pore_threshold(self, result, sample, sensitivity):
        """Draw the interior histogram with every candidate threshold."""
        spread = [v for k, v in sensitivity.items()
                  if k.startswith("k=") and np.isfinite(v)]
        spread_txt = ""
        if len(spread) >= 2:
            spread_txt = (f"    porosity spans {min(spread):.3f}-"
                          f"{max(spread):.3f} % across k = 2 to 4")
        self._poro_summary.setText(
            f"threshold {result.T:.1f} ({result.method_used})    "
            f"solid peak {result.solid_mode:.1f} ± {result.solid_sigma:.1f}    "
            f"candidates: sigma {result.t_sigma:.1f}, otsu {result.t_otsu:.1f}, "
            f"valley "
            f"{'none' if result.t_valley is None else f'{result.t_valley:.1f}'}"
            f"{spread_txt}    →  now click ▶ Analyze Voids"
        )
        if not HAS_MPL:
            return

        ax = self._poro_ax3
        ax.clear()
        ax.set_facecolor(PANEL2)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BORDER)

        lo = float(np.percentile(sample, 0.05))
        hi = float(np.percentile(sample, 99.9))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 320)
        ax.hist(sample, bins=bins, histtype="step", lw=1.4, density=True,
                color=TEAL, label="Interior histogram")

        xs = np.linspace(lo, hi, 600)
        sigma = max(result.solid_sigma, 1e-9)
        gauss = (np.exp(-0.5 * ((xs - result.solid_mode) / sigma) ** 2)
                 / (sigma * np.sqrt(2.0 * np.pi)))
        ax.plot(xs, gauss, color=TEXT_DIM, lw=1.0, ls="-.",
                label=f"Solid peak {result.solid_mode:.0f} ± {sigma:.0f}")

        ax.axvline(result.t_sigma, color=ACCENT, ls="--", lw=1.2,
                   label=f"{result.k_sigma:g}-sigma = {result.t_sigma:.0f}")
        ax.axvline(result.t_otsu, color=TEXT, ls=":", lw=1.2,
                   label=f"Otsu = {result.t_otsu:.0f}")
        if result.t_valley is not None:
            ax.axvline(result.t_valley, color=ACCENT2, ls="--", lw=1.2,
                       label=f"Valley = {result.t_valley:.0f}")
        ax.axvline(result.T, color=ERR, lw=2.0, alpha=0.85,
                   label=f"Chosen = {result.T:.0f} ({result.method_used})")

        ax.set_yscale("log")
        ax.set_xlabel("Grey value", color=TEXT_DIM, fontsize=8)
        ax.set_ylabel("Density (log)", color=TEXT_DIM, fontsize=8)
        ax.set_title("Pore threshold on the interior histogram",
                     color=TEXT, fontsize=9)
        ax.legend(fontsize=6.5, facecolor=PANEL2, edgecolor=BORDER,
                  loc="upper left")

        self._poro_redraw()
        self._show_poro_panel("threshold")

    # Panels

    def _poro_redraw(self):
        """Lay out and repaint every plot panel."""
        if not HAS_MPL:
            return
        for key, fig in self._poro_figs.items():
            try:
                fig.tight_layout(pad=1.4)
            except Exception:
                pass
            self._poro_canvases[key].draw_idle()

    def _show_poro_panel(self, key: str):
        """Bring one plot to the front of the selector."""
        button = getattr(self, "_poro_nav_buttons", {}).get(key)
        if button is None:
            return
        button.setChecked(True)
        for index, (name, _label) in enumerate(PORO_PANELS):
            if name == key:
                self._poro_stack.setCurrentIndex(index)
                break

    # Action

    def _do_porosity(self):
        if not self._require_img():
            return
        void_thresh = self._resolve_void_thresh()
        if void_thresh is None:
            QMessageBox.critical(
                self, "Bad threshold",
                "Could not determine a void threshold. Open a file first.",
            )
            return
        # Catches HU-style thresholds entered against raw-intensity data.
        void_thresh = self._sane_void_thresh(void_thresh)
        try:
            min_vox = max(1, int(float(self._poro_min_vox.text())))
        except Exception:
            min_vox = 8
        connectivity = self._poro_conn.currentIndex() + 1
        excl_border = self._poro_excl_border.isChecked()
        want_sph = self._poro_sphericity.isChecked()
        specimen_basis = self._poro_specimen_only.isChecked()

        def _run():
            try:
                from ..core.porosity import (
                    analyze_voids, build_void_mask, choose_stride,
                    specimen_from_void_mask, summary_lines,
                )
                from ..utils import available_ram_mb, OperationCancelled

                self._log_sep("Porosity / Void Analysis")
                self._set_status("Analyzing voids...", busy=True)

                # Never materialise the float32 volume. The mask is streamed
                # slice by slice, downsampled if RAM demands it.
                vol = self._get_gray_lazy()
                spacing = self._img.header.get_zooms()[:3]

                stride = choose_stride(vol.shape, available_ram_mb())
                self._append_log(
                    f"  Void mask: grey value < {void_thresh:g}   "
                    f"min={min_vox} vox  conn={connectivity}  "
                    f"border-excluded={excl_border}", 'dim',
                )
                if stride > 1:
                    eff = tuple(float(s) * stride for s in spacing)
                    self._append_log(
                        f"  Volume too large for RAM at full resolution, "
                        f"analysing every {stride}. voxel "
                        f"(effective voxel {eff[0]:.4f} mm). Voids below "
                        f"~{2 * max(eff):.3f} mm won't be resolved.", 'warn',
                    )

                def _mask_prog(done, total):
                    if self.cancel_requested():
                        raise OperationCancelled()
                    self._set_status(
                        f"Building void mask...  slice {done}/{total}",
                        busy=True,
                    )

                mask = build_void_mask(vol, void_thresh, stride,
                                       progress=_mask_prog)
                spacing_eff = tuple(float(s) * stride for s in spacing)

                # Referring the porosity to the specimen rather than the
                # whole array makes it a property of the material instead of
                # the field of view.
                specimen_voxels = None
                if specimen_basis:
                    self._set_status("Finding the specimen...", busy=True)
                    specimen = specimen_from_void_mask(mask, erode=1)
                    specimen_voxels = int(specimen.sum())
                    if specimen_voxels == 0:
                        self._append_log(
                            "  No connected specimen found above the "
                            "threshold, falling back to the whole-array "
                            "basis.", 'warn')
                        specimen_voxels = None
                    else:
                        outside = int((mask & ~specimen).sum())
                        mask &= specimen
                        self._append_log(
                            f"  Specimen basis: {specimen_voxels:,} voxels "
                            f"({100.0 * specimen_voxels / mask.size:.1f} % of "
                            f"the array); {outside:,} void voxels outside the "
                            f"specimen (surrounding air and the surface "
                            f"layer) are excluded.", 'dim')
                    del specimen

                # Mid-slice preview of which voxels count as void, the check
                # to make when porosity reads 0.
                zi = mask.shape[2] // 2
                mid_mask = mask[:, :, zi].copy()
                mid_gray = np.asarray(
                    vol[:, :, min(zi * stride, vol.shape[2] - 1)]
                )[::stride, ::stride]

                def _prog(done, total):
                    if self.cancel_requested():
                        raise OperationCancelled()
                    if total and done % 1000 == 0:
                        self._set_status(
                            f"Measuring voids...  {done}/{total}", busy=True,
                        )

                self._set_status("Labelling voids...", busy=True)
                result = analyze_voids(
                    mask, spacing_eff,
                    min_voxels=min_vox, connectivity=connectivity,
                    exclude_border=excl_border,
                    specimen_voxels=specimen_voxels,
                    compute_sphericity=want_sph, progress=_prog,
                )
                del mask
                result.stride = stride
                self._void_result = result

                for ln in summary_lines(result):
                    self._append_log("  " + ln, 'teal')

                self.after(0, self._show_porosity_result, result,
                           mid_gray, mid_mask)
                self.after(0, self._show_tab, 'porosity')
                self._set_status("Void analysis complete.", busy=False)
            except Exception as ex:
                from ..utils import OperationCancelled
                if isinstance(ex, OperationCancelled):
                    self._append_log("  Void analysis stopped by user.",
                                     'warn')
                    self._set_status("Stopped.", busy=False)
                else:
                    self._append_log(f"  Void analysis error: {ex}", 'err')
                    self._set_status("Void analysis error.", busy=False)

        self._run_task("Analyse voids", _run)

    # Drawing

    def _show_porosity_result(self, r, mid_gray=None, mid_mask=None):
        raw_pct = 100.0 * r.raw_void_voxels / max(1, r.analysed_voxels)

        # Summary bar; the empty cases state why nothing was found.
        if r.voids:
            basis = "of specimen" if r.specimen_voxels else "of whole array"
            self._poro_summary.setText(
                f"voids: {r.n_voids}    closed porosity: "
                f"{r.porosity_pct:.3f} % {basis}"
                f"    raw void fraction: {raw_pct:.3f} %"
                f"    eq.diam mean/median/max: "
                f"{r.d_mean_mm:.4f} / {r.d_median_mm:.4f} / {r.d_max_mm:.4f} mm"
                f"    NN spacing: {r.mean_nn_dist_mm:.4f} mm"
            )
        elif raw_pct < 0.001:
            self._poro_summary.setText(
                f"The threshold selects no voxels at all (raw void "
                f"fraction {raw_pct:.4f} %). The threshold does not match "
                f"the data; check the Histogram tab: it should sit "
                f"between the air peak and the material peak."
            )
        else:
            hint = ""
            if r.stride > 1:
                hint = (f"  At ×{r.stride} downsampling, voids below "
                        f"~{2 * max(r.spacing_mm):.2f} mm are invisible. "
                        f"Crop a region of interest (Crop tool) and rerun "
                        f"at full resolution.")
            self._poro_summary.setText(
                f"The mask contains {raw_pct:.3f} % void voxels, but no "
                f"connected void survived the filters (≥ {r.min_voxels} "
                f"vox, border-touching removed). Lower 'Min void size' or "
                f"untick 'Exclude border air' to inspect.{hint}"
            )

        # The mask preview is drawn even when no void survived the filters.
        if HAS_MPL:
            ax0, ax1, ax2 = self._poro_ax0, self._poro_ax1, self._poro_ax2
            for ax in (ax0, ax1, ax2):
                ax.clear()
                ax.set_facecolor(PANEL2)
                ax.tick_params(colors=TEXT_DIM, labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color(BORDER)

            if mid_gray is not None and mid_mask is not None:
                from matplotlib.colors import ListedColormap
                ax0.imshow(np.rot90(mid_gray), cmap='gray', aspect='equal')
                mm = np.rot90(mid_mask)
                if mm.any():
                    ax0.imshow(
                        np.ma.masked_where(~mm, mm),
                        cmap=ListedColormap(["#E5484D"]),
                        alpha=0.55, interpolation='nearest', aspect='equal',
                    )
                ax0.set_title(
                    f"Mid slice: red = void  ({raw_pct:.2f} % of voxels)",
                    color=TEXT, fontsize=9,
                )
                ax0.axis('off')
            else:
                ax0.set_title("No preview", color=TEXT_DIM, fontsize=9)
                ax0.axis('off')

            if not r.voids:
                for ax, msg in ((ax1, "no voids to plot"),
                                (ax2, "")):
                    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                            ha='center', color=TEXT_DIM, fontsize=9)
                self._poro_redraw()
            else:
                self._draw_poro_plots(r)

        self._show_poro_panel("size" if r.voids else "mask")
        self._fill_poro_table(r)

    def _draw_poro_plots(self, r):
        d = r.diameters_mm()
        v = r.volumes_mm3()
        ax1, ax2 = self._poro_ax1, self._poro_ax2
        if True:
            bins = min(60, max(10, int(np.sqrt(d.size))))
            ax1.hist(d, bins=bins, color=ACCENT, alpha=0.85, linewidth=0)
            ax1.set_xlabel("Equivalent diameter (mm)", color=TEXT_DIM, fontsize=8)
            ax1.set_ylabel("Void count", color=TEXT_DIM, fontsize=8)
            ax1.set_title("Void Size Distribution", color=TEXT, fontsize=9)
            ax1.axvline(r.d_median_mm, color=TEAL, lw=1.2, linestyle='--',
                        label=f"median {r.d_median_mm:.3f} mm")
            ax1.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER)

            order = np.argsort(d)
            cum = np.cumsum(v[order]) / v.sum() * 100.0
            ax2.plot(d[order], cum, color=ACCENT2, lw=1.6)
            ax2.set_xlabel("Equivalent diameter (mm)", color=TEXT_DIM, fontsize=8)
            ax2.set_ylabel("Cumulative void volume (%)", color=TEXT_DIM, fontsize=8)
            ax2.set_title("Cumulative Volume vs Size", color=TEXT, fontsize=9)
            ax2.set_ylim(0, 105)

            self._poro_redraw()

    def _fill_poro_table(self, r):
        """Populate the void table (largest first, capped)."""
        t = self._poro_table
        t.setSortingEnabled(False)
        rows = r.voids[:_TABLE_MAX_ROWS]
        t.setRowCount(len(rows))
        for i, void in enumerate(rows):
            cvox = "({:.0f}, {:.0f}, {:.0f})".format(*void.centroid_vox)
            cmm = "({:.2f}, {:.2f}, {:.2f})".format(*void.centroid_mm)
            sph = "n/a" if np.isnan(void.sphericity) else f"{void.sphericity:.3f}"
            cells = [
                (i + 1, True), (void.voxels, True),
                (f"{void.volume_mm3:.5g}", False),
                (f"{void.eq_diam_mm:.4f}", False),
                (sph, False), (f"{void.elongation:.2f}", False),
                (cvox, False), (cmm, False),
            ]
            for col, (val, numeric) in enumerate(cells):
                item = QTableWidgetItem()
                if numeric:
                    item.setData(Qt.ItemDataRole.DisplayRole, int(val))
                else:
                    item.setText(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
                t.setItem(i, col, item)
        t.setSortingEnabled(True)

    # Export

    def _export_voids_csv(self):
        if self._void_result is None or not self._void_result.voids:
            QMessageBox.warning(
                self, "No void data", "Run  ▶ Analyze Voids  first.",
            )
            return
        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '') \
            if self._path else "volume"
        out = self._ask_save_path(
            f"{stem}_voids.csv", ext=".csv", ftypes=[("CSV", "*.csv")],
        )
        if not out:
            return

        def _run():
            try:
                from ..core.porosity import voids_to_csv

                self._log_sep("Export voids CSV")
                if self._abort_if_stopped("CSV export"):
                    return
                voids_to_csv(self._void_result, out)
                self._append_log(
                    f"  {len(self._void_result.voids)} voids → {out}", 'ok',
                )
                self._set_status("Voids CSV saved.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Export error.", busy=False)

        self._run_task("Export voids CSV", _run)
