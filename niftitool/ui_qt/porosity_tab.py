"""Porosity tab: per-void connected-component analytics.

Runs :func:`niftitool.core.porosity.analyze_voids` on the void phase
(below the Material panel's void threshold), then shows:

* a population summary (count, closed porosity, diameter percentiles),
* a size-distribution histogram + cumulative void-volume curve,
* a sortable table of the largest voids,
* CSV export of every measured void.

The result object is kept on ``self._void_result`` so the PDF report
generator can embed it.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..config import ACCENT, ACCENT2, BG, BORDER, ERR, PANEL2, TEAL, TEXT, TEXT_DIM
from ..deps import HAS_MPL, np
from .widgets import styled_btn, styled_entry

_TABLE_MAX_ROWS = 500


class PorosityTabMixin:
    """Adds the Porosity tab to :class:`NiftiApp`."""

    def _build_porosity_tab(self, parent):
        self._void_result = None

        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── options row ──────────────────────────────────────────────
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
        self._poro_excl_border.setToolTip(
            "Discard components touching the volume boundary\n"
            "(the exterior air around the specimen)."
        )
        opts_lay.addWidget(self._poro_excl_border)

        self._poro_sphericity = QCheckBox("Sphericity (slower)", opts)
        self._poro_sphericity.setChecked(True)
        self._poro_sphericity.setToolTip(
            "Mesh each void with marching cubes to measure surface area.\n"
            "Disable for a much faster first pass on huge volumes."
        )
        opts_lay.addWidget(self._poro_sphericity)

        opts_lay.addWidget(styled_btn(
            opts, "▶ Analyze Voids", self._do_porosity, accent=True,
        ))
        opts_lay.addWidget(styled_btn(
            opts, "Export voids CSV...", self._export_voids_csv, small=True,
        ))
        opts_lay.addStretch(1)
        root.addWidget(opts)

        # ── summary line ─────────────────────────────────────────────
        self._poro_summary = QLabel(
            "No analysis yet — set the void threshold in the Material panel, "
            "then click  ▶ Analyze Voids.", parent,
        )
        self._poro_summary.setFont(QFont("Consolas", 9))
        self._poro_summary.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2}; padding: 6px 10px;"
        )
        self._poro_summary.setWordWrap(True)
        root.addWidget(self._poro_summary)

        # ── plots | table splitter ───────────────────────────────────
        split = QSplitter(Qt.Orientation.Vertical, parent)
        split.setHandleWidth(6)
        split.setChildrenCollapsible(False)

        if HAS_MPL:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            plot_holder = QWidget(split)
            ph_lay = QVBoxLayout(plot_holder)
            ph_lay.setContentsMargins(0, 0, 0, 0)
            self._poro_fig = Figure(facecolor=BG)
            self._poro_ax0, self._poro_ax1, self._poro_ax2 = \
                self._poro_fig.subplots(1, 3)
            for ax in (self._poro_ax0, self._poro_ax1, self._poro_ax2):
                ax.set_facecolor(PANEL2)
            self._poro_canvas = FigureCanvasQTAgg(self._poro_fig)
            ph_lay.addWidget(self._poro_canvas)
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

    # ── action ───────────────────────────────────────────────────────

    def _do_porosity(self):
        if not self._require_img():
            return
        void_thresh = self._resolve_void_thresh()
        if void_thresh is None:
            QMessageBox.critical(
                self, "Bad threshold",
                "Could not determine a void threshold — open a file first.",
            )
            return
        # Catch manually-entered HU-style thresholds applied to
        # raw-intensity data before streaming an empty mask.
        void_thresh = self._sane_void_thresh(void_thresh)
        try:
            min_vox = max(1, int(float(self._poro_min_vox.text())))
        except Exception:
            min_vox = 8
        connectivity = self._poro_conn.currentIndex() + 1
        excl_border = self._poro_excl_border.isChecked()
        want_sph = self._poro_sphericity.isChecked()

        def _run():
            try:
                from ..core.porosity import (
                    analyze_voids, build_void_mask, choose_stride,
                    summary_lines,
                )
                from ..utils import available_ram_mb, OperationCancelled

                self._begin_cancellable()
                self._log_sep("Porosity / Void Analysis")
                self._set_status("Analyzing voids...", busy=True)

                # Never materialise the float32 volume — the mask is
                # streamed slice-by-slice, downsampled if RAM demands it.
                vol = self._hu_vol if self._hu_vol is not None \
                    else self._get_gray_lazy()
                thresh_domain = "HU" if self._hu_vol is not None else "raw intensity"
                spacing = self._img.header.get_zooms()[:3]

                stride = choose_stride(vol.shape, available_ram_mb())
                self._append_log(
                    f"  Void mask: {thresh_domain} < {void_thresh:g}   "
                    f"min={min_vox} vox  conn={connectivity}  "
                    f"border-excluded={excl_border}", 'dim',
                )
                if stride > 1:
                    eff = tuple(float(s) * stride for s in spacing)
                    self._append_log(
                        f"  Volume too large for RAM at full resolution — "
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

                # Mid-slice preview: lets the user SEE what counts as
                # void — the definitive check when porosity reads 0.
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

        threading.Thread(target=_run, daemon=True).start()

    # ── drawing ──────────────────────────────────────────────────────

    def _show_porosity_result(self, r, mid_gray=None, mid_mask=None):
        raw_pct = 100.0 * r.raw_void_voxels / max(1, r.analysed_voxels)

        # summary bar — when zero, say WHY it is zero
        if r.voids:
            self._poro_summary.setText(
                f"voids: {r.n_voids}    closed porosity: {r.porosity_pct:.3f} %"
                f"    raw void fraction: {raw_pct:.3f} %"
                f"    eq.diam mean/median/max: "
                f"{r.d_mean_mm:.4f} / {r.d_median_mm:.4f} / {r.d_max_mm:.4f} mm"
                f"    NN spacing: {r.mean_nn_dist_mm:.4f} mm"
            )
        elif raw_pct < 0.001:
            self._poro_summary.setText(
                f"The threshold selects no voxels at all (raw void "
                f"fraction {raw_pct:.4f} %). The threshold does not match "
                f"the data — check the Histogram tab: it should sit "
                f"between the air peak and the material peak."
            )
        else:
            hint = ""
            if r.stride > 1:
                hint = (f"  At ×{r.stride} downsampling, voids below "
                        f"~{2 * max(r.spacing_mm):.2f} mm are invisible — "
                        f"crop a region of interest (Crop tool) and rerun "
                        f"at full resolution.")
            self._poro_summary.setText(
                f"The mask contains {raw_pct:.3f} % void voxels, but no "
                f"connected void survived the filters (≥ {r.min_voxels} "
                f"vox, border-touching removed). Lower 'Min void size' or "
                f"untick 'Exclude border air' to inspect.{hint}"
            )

        # plots — the mask preview is drawn even when nothing was found;
        # it is the definitive "what is being counted" check.
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
                    f"Mid slice — red = void  ({raw_pct:.2f} % of voxels)",
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
                self._poro_fig.tight_layout(pad=1.2)
                self._poro_canvas.draw()
            else:
                self._draw_poro_plots(r)

        # table (largest first, capped)
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

            self._poro_fig.tight_layout(pad=1.2)
            self._poro_canvas.draw()

    def _fill_poro_table(self, r):
        """Populate the void table (largest first, capped)."""
        t = self._poro_table
        t.setSortingEnabled(False)
        rows = r.voids[:_TABLE_MAX_ROWS]
        t.setRowCount(len(rows))
        for i, void in enumerate(rows):
            cvox = "({:.0f}, {:.0f}, {:.0f})".format(*void.centroid_vox)
            cmm = "({:.2f}, {:.2f}, {:.2f})".format(*void.centroid_mm)
            sph = "—" if np.isnan(void.sphericity) else f"{void.sphericity:.3f}"
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

    # ── export ───────────────────────────────────────────────────────

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
                voids_to_csv(self._void_result, out)
                self._append_log(
                    f"  {len(self._void_result.voids)} voids → {out}", 'ok',
                )
                self._set_status("Voids CSV saved.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Export error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()
