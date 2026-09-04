"""Binarise tab: the indicator field the finite cell method integrates.

Shows the histogram with the chosen cut drawn on it and the resulting
indicator field over the same slice. The threshold sweep answers what the
histogram cannot, namely whether the cut sits on a real pore population or
in the noise of the material peak.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget,
)

from ..config import ACCENT, BG, BORDER, ERR, PANEL2, TEAL, TEXT, TEXT_DIM
from ..deps import HAS_MPL, np
from .figure_viewer import click_hint, enable_click_to_enlarge
from .widgets import styled_btn


class BinarizeTabMixin:
    """Adds the Binarise tab to :class:`NiftiApp`."""

    def _build_binarize_tab(self, parent):
        self._bin_result = None

        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QWidget(parent)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(8, 6, 8, 6)
        bar_lay.setSpacing(6)
        bar_lay.addWidget(styled_btn(
            bar, "▶ Preview", self._do_binarize_preview, accent=True))
        bar_lay.addWidget(styled_btn(
            bar, "Write NIfTI...", self._do_binarize_write, teal=True))
        bar_lay.addWidget(styled_btn(
            bar, "Threshold sweep", self._do_binarize_sweep, small=True))
        bar_lay.addWidget(styled_btn(
            bar, "Save report JSON...", self._export_binarize_json,
            small=True))
        bar_lay.addStretch(1)
        root.addWidget(bar)

        self._bin_summary = QLabel(
            "No binarisation yet. Set the options in the Binarise tool "
            "panel on the left, then click  ▶ Preview.", parent,
        )
        self._bin_summary.setFont(QFont("Consolas", 9))
        self._bin_summary.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2}; padding: 6px 10px;"
        )
        self._bin_summary.setWordWrap(True)
        root.addWidget(self._bin_summary)

        if HAS_MPL:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            holder = QWidget(parent)
            h_lay = QVBoxLayout(holder)
            h_lay.setContentsMargins(0, 0, 0, 0)
            h_lay.setSpacing(2)
            self._bin_fig = Figure(facecolor=BG)
            self._bin_ax_hist, self._bin_ax_slice, self._bin_ax_mask = \
                self._bin_fig.subplots(1, 3)
            for ax in (self._bin_ax_hist, self._bin_ax_slice,
                       self._bin_ax_mask):
                ax.set_facecolor(PANEL2)
            self._bin_canvas = FigureCanvasQTAgg(self._bin_fig)
            h_lay.addWidget(self._bin_canvas)

            h_lay.addWidget(click_hint(holder))
            enable_click_to_enlarge(self._bin_canvas, self._bin_fig, self)
            root.addWidget(holder, 1)
            self._clear_bin_axes()
            self._bin_canvas.draw()
        else:
            lbl = QLabel("matplotlib not installed", parent)
            lbl.setStyleSheet(f"color: {ERR}; background-color: {BG};")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(lbl, 1)

    # --- options -------------------------------------------------------

    def _bin_options(self) -> dict:
        def _f(widget, default):
            try:
                return float(widget.text())
            except Exception:
                return default

        def _i(widget, default):
            try:
                return int(float(widget.text()))
            except Exception:
                return default

        manual = None
        text = self._bin_manual_widget.text().strip()
        if text:
            try:
                manual = float(text)
            except ValueError:
                manual = None

        return {
            "method": self._bin_method_combo.currentText().strip(),
            "sigma_k": _f(self._bin_k_widget, 3.0),
            "manual": manual,
            "smooth": max(0.0, _f(self._bin_smooth_widget, 0.0)),
            "downsample": max(1, _i(self._bin_downsample_widget, 1)),
            "min_void": max(0, _i(self._bin_minvoid_widget, 0)),
            "largest_only": bool(self._bin_largest_check.isChecked()),
            "refine_inside": bool(self._bin_refine_check.isChecked()),
        }

    def _bin_threshold(self, opts, sample):
        """Threshold for *sample*, refined inside the specimen envelope.

        Returns ``(threshold, global_threshold, envelope_fraction, smoothed,
        inside)``. ``inside`` is the specimen interior, on which the material
        peak and the separability check must be measured. On the whole volume
        the tallest histogram peak is the surrounding air, so every figure
        derived from it would describe the background.
        """
        from ..core import binarize as bz

        if opts["method"] == "manual" and opts["manual"] is None:
            raise ValueError(
                "The manual method needs a value in the 'Manual value' box.")

        smoothed = sample
        if opts["smooth"] > 0:
            from ..deps import ndimage
            if ndimage is not None:
                # Threshold the data the way it will actually be binarised.
                smoothed = ndimage.gaussian_filter(
                    sample, opts["smooth"], mode="nearest")

        global_t = bz.compute_threshold(
            smoothed, opts["method"], sigma_k=opts["sigma_k"],
            manual=opts["manual"])
        threshold = global_t
        envelope_fraction = None
        inside = smoothed

        envelope = bz.specimen_envelope(smoothed, global_t)
        if envelope is not None and envelope.any():
            envelope_fraction = float(envelope.mean())
            candidate = smoothed[envelope]
            if candidate.size >= 1000:
                inside = candidate
                if opts["refine_inside"] and opts["method"] != "manual":
                    refined = bz.compute_threshold(
                        inside, opts["method"], sigma_k=opts["sigma_k"],
                        manual=opts["manual"])
                    if np.isfinite(refined):
                        threshold = float(refined)
        return (float(threshold), float(global_t), envelope_fraction,
                smoothed, inside)

    # --- actions -------------------------------------------------------

    def _do_binarize_sweep(self):
        """Report the void fraction against threshold = mode − k·sigma."""
        if not self._require_img():
            return

        def _run():
            try:
                from ..core import binarize as bz

                self._log_sep("Binarise: threshold sweep")
                self._set_status("Sampling for the sweep...", busy=True)
                sample, stride = bz.sample_volume(self._get_gray_lazy())
                inside = sample
                provisional = bz.otsu_threshold(sample)
                envelope = bz.specimen_envelope(sample, provisional)
                if envelope is not None and envelope.any():
                    inside = sample[envelope]
                mode, sigma, rows = bz.porosity_sweep(inside)
                if not rows:
                    self._append_log(
                        "  The sample has no spread, so no sweep is "
                        "possible.", 'warn')
                    self._set_status("Sweep produced nothing.", busy=False)
                    return
                self._append_log(
                    f"  Material mode {mode:.0f}, sigma {sigma:.0f} "
                    f"(inside the specimen, stride {stride})", 'dim')
                self._append_log(
                    "       k    threshold   void fraction   drop vs previous",
                    'dim')
                for row in rows:
                    drop = ("        -" if row["drop_from_previous"] is None
                            else f"{row['drop_from_previous'] * 100:8.3f} pt")
                    self._append_log(
                        f"    {row['k']:4.1f}    {row['threshold']:9.0f}   "
                        f"{row['void_fraction'] * 100:9.3f} %   {drop}",
                        'teal')
                self._append_log(
                    "  Pick k where the drop flattens: that plateau is the "
                    "pore population, and everything steeper than it is the "
                    "material's own noise.", 'dim')
                self._set_status("Sweep complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  Sweep error: {ex}", 'err')
                self._set_status("Sweep error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    def _do_binarize_preview(self):
        """Choose the threshold and show it, without writing a volume."""
        if not self._require_img():
            return
        try:
            opts = self._bin_options()
        except Exception as ex:
            QMessageBox.warning(self, "Input error", str(ex))
            return

        def _run():
            try:
                from ..core import binarize as bz

                self._log_sep("Binarise: preview")
                self._set_status("Sampling the volume...", busy=True)
                vol = self._get_gray_lazy()
                sample, stride = bz.sample_volume(vol)
                threshold, global_t, env_frac, smoothed, inside = \
                    self._bin_threshold(opts, sample)

                warning, mode, sigma = bz.separability_note(inside, threshold)
                below = float((smoothed < threshold).mean())

                self._append_log(
                    f"  Sampled {smoothed.size:,} voxels at stride {stride}   "
                    f"range {float(smoothed.min()):.1f} to "
                    f"{float(smoothed.max()):.1f}", 'dim')
                self._append_log(
                    f"  {opts['method']} threshold = {threshold:.2f}   "
                    f"({below * 100:.1f} % of the sample below it"
                    + (", after smoothing" if opts["smooth"] > 0 else "") + ")"
                    + (f"   [before refinement: {global_t:.2f}]"
                       if threshold != global_t else ""), 'teal')
                if env_frac is not None:
                    self._append_log(
                        f"  Specimen fills {env_frac * 100:.1f} % of the box; "
                        f"the threshold was recomputed inside it so the "
                        f"surrounding air cannot drag it.", 'dim')
                if warning:
                    self._append_log(f"  WARNING: {warning}", 'warn')

                mask = (smoothed >= threshold)
                zi = smoothed.shape[2] // 2
                self._bin_preview = {
                    "sample": smoothed, "threshold": threshold,
                    "method": opts["method"],
                    "slice": smoothed[:, :, zi], "mask": mask[:, :, zi],
                    "z": zi, "stride": stride,
                    "material_fraction": float(mask.mean()),
                    "warning": warning, "mode": mode, "sigma": sigma,
                }
                self.after(0, self._show_bin_preview)
                self.after(0, self._show_tab, 'binarize')
                self._set_status("Preview ready.", busy=False)
            except Exception as ex:
                self._append_log(f"  Binarise error: {ex}", 'err')
                self._set_status("Binarise error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    def _do_binarize_write(self):
        """Stream the indicator field to a uint8 NIfTI."""
        if not self._require_img():
            return
        try:
            opts = self._bin_options()
        except Exception as ex:
            QMessageBox.warning(self, "Input error", str(ex))
            return

        stem = self._nifti_stem()
        out = self._ask_save_path(f"{stem}_binary.nii", ext=".nii",
                                  ftypes=[("NIfTI", "*.nii"), ("All", "*.*")])
        if not out:
            return

        def _run():
            try:
                from ..core import binarize as bz
                from ..utils import OperationCancelled

                self._begin_cancellable()
                self._log_sep("Binarise: write indicator field")
                self._set_status("Sampling the volume...", busy=True)
                vol = self._get_gray_lazy()
                sample, _stride = bz.sample_volume(vol)
                threshold, global_t, env_frac, smoothed, inside = \
                    self._bin_threshold(opts, sample)
                warning, mode, sigma = bz.separability_note(inside, threshold)

                def _prog(done, total):
                    self._set_status(
                        f"Binarising...  slice {done}/{total}", busy=True)

                info = bz.binarize_stream(
                    self._img, vol, threshold, out,
                    smooth=opts["smooth"], downsample=opts["downsample"],
                    progress=_prog, cancelled=self.cancel_requested,
                )
                total = int(np.prod(info["shape"]))
                result = bz.BinarizeResult(
                    method=opts["method"], threshold=threshold,
                    threshold_before_refinement=global_t,
                    specimen_fraction_of_box=env_frac,
                    separability_warning=warning,
                    material_mode=mode, material_sigma=sigma,
                    shape=info["shape"], voxel_size_mm=info["zooms"],
                    material_voxels=info["material_voxels"],
                    total_voxels=total,
                    material_fraction_of_box=info["material_voxels"] / max(1, total),
                    smooth_sigma=opts["smooth"], downsample=opts["downsample"],
                    output_path=str(out),
                    sample=smoothed,
                    preview_mask=info["preview_mask"],
                    preview_slice=info["preview_slice"],
                    preview_z=info["preview_z"],
                )

                # Component analysis and pore filling need the whole mask in
                # memory, so skip them on a large volume.
                needs_mask = (opts["largest_only"] or opts["min_void"] > 0
                              or total <= 600_000_000)
                if needs_mask and total <= 600_000_000:
                    self._set_status("Analysing the material...", busy=True)
                    written = nib_load_mask(out)
                    mask, changed = written, False
                    if opts["min_void"] > 0:
                        mask, filled = bz.remove_small_voids(
                            mask, opts["min_void"])
                        result.voids_filled_voxels = filled
                        changed = changed or filled > 0
                    mask, comp = bz.component_report(mask, opts["largest_only"])
                    result.material_components = comp.get(
                        "material_components", 0)
                    result.largest_component_fraction = comp.get(
                        "largest_component_fraction")
                    result.removed_voxels = comp.get("removed_voxels", 0)
                    changed = changed or result.removed_voxels > 0
                    if changed:
                        self._rewrite_binary_mask(out, mask, info)
                        result.material_voxels = int(mask.sum())
                        result.material_fraction_of_box = float(mask.mean())
                        zi = min(result.preview_z, mask.shape[2] - 1)
                        result.preview_mask = mask[:, :, zi]
                    pores = bz.porosity_inside_specimen(mask)
                    if pores:
                        result.porosity_inside_specimen = pores["porosity"]
                        result.specimen_voxels = pores["specimen_voxels"]
                    del mask
                elif total > 600_000_000:
                    self._append_log(
                        "  Volume too large for the component analysis. "
                        "Use a downsample factor for that step.", 'warn')

                self._bin_result = result
                for line in bz.summary_lines(result):
                    self._append_log("  " + line, 'teal')
                self._append_log(
                    f"  Saved → {Path(out).name}  "
                    f"({info['seconds']:.1f} s)", 'ok')

                self.after(0, self._show_bin_result, result)
                self.after(0, self._show_tab, 'binarize')
                self._set_status("Indicator field written.", busy=False)
                self.after(0, self._ask_load_after_save,
                           "Binary indicator field saved to:", Path(out))
            except Exception as ex:
                from ..utils import OperationCancelled
                if isinstance(ex, OperationCancelled):
                    self._append_log("  Binarisation stopped by user.", 'warn')
                    self._set_status("Stopped.", busy=False)
                else:
                    self._append_log(f"  Binarise error: {ex}", 'err')
                    self._set_status("Binarise error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    def _rewrite_binary_mask(self, path, mask, info):
        """Save a post-processed mask back over the streamed file."""
        from ..deps import nib

        img = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8),
                              info["affine"])
        img.set_data_dtype(np.uint8)
        img.header.set_zooms(tuple(float(z) for z in info["zooms"]))
        img.header['descrip'] = b'FCM indicator: 1=material 0=void'
        nib.save(img, str(path))

    def _export_binarize_json(self):
        if self._bin_result is None:
            QMessageBox.warning(
                self, "Nothing to save",
                "Write an indicator field first; the report records how it "
                "was produced.")
            return
        stem = self._nifti_stem()
        out = self._ask_save_path(
            f"{stem}_binarize_report.json", ext=".json",
            ftypes=[("JSON", "*.json"), ("All", "*.*")])
        if not out:
            return
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self._bin_result.to_dict(), f, indent=2)
            self._append_log(f"  Report → {Path(out).name}", 'ok')
        except Exception as ex:
            self._append_log(f"  {ex}", 'err')

    # --- drawing -------------------------------------------------------

    def _clear_bin_axes(self):
        if not HAS_MPL:
            return
        for ax in (self._bin_ax_hist, self._bin_ax_slice, self._bin_ax_mask):
            ax.clear()
            ax.set_facecolor(PANEL2)
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
        self._bin_ax_hist.set_xlabel("Grey value", color=TEXT_DIM, fontsize=8)
        self._bin_ax_hist.set_ylabel("Voxel count (log)", color=TEXT_DIM,
                                     fontsize=8)
        for ax, title in ((self._bin_ax_slice, "Slice"),
                          (self._bin_ax_mask, "Indicator field")):
            ax.set_title(title, color=TEXT_DIM, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    def _draw_bin(self, sample, threshold, method, sl, mask, z, extra=""):
        if not HAS_MPL:
            return
        self._clear_bin_axes()
        ax = self._bin_ax_hist
        values = np.asarray(sample).ravel()
        values = values[np.isfinite(values)]
        ax.hist(values, bins=400, color=TEAL, log=True)
        ax.axvline(threshold, color=ERR, lw=1.8,
                   label=f"{method} = {threshold:.1f}")
        ax.set_xlabel("Grey value", color=TEXT_DIM, fontsize=8)
        ax.set_ylabel("Voxel count (log)", color=TEXT_DIM, fontsize=8)
        ax.set_title("Histogram and the cut", color=TEXT, fontsize=9)
        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER)

        if sl is not None:
            lo, hi = np.percentile(sl, [1, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            self._bin_ax_slice.imshow(np.rot90(sl), cmap="gray", vmin=lo,
                                      vmax=hi, aspect="equal")
        self._bin_ax_slice.set_title(f"Slice z = {z}", color=TEXT, fontsize=9)
        self._bin_ax_slice.axis("off")

        if mask is not None:
            self._bin_ax_mask.imshow(np.rot90(np.asarray(mask, dtype=float)),
                                     cmap="gray", vmin=0, vmax=1,
                                     interpolation="nearest", aspect="equal")
        self._bin_ax_mask.set_title(
            "Indicator field (white = material)" + extra, color=TEXT,
            fontsize=9)
        self._bin_ax_mask.axis("off")

        self._bin_fig.tight_layout(pad=1.2)
        self._bin_canvas.draw()

    def _show_bin_preview(self):
        p = self._bin_preview
        note = ("   " + p["warning"]) if p["warning"] else ""
        self._bin_summary.setText(
            f"preview (stride {p['stride']})    {p['method']} threshold "
            f"{p['threshold']:.1f}    material {p['material_fraction'] * 100:.2f} % "
            f"of the sampled box    material peak {p['mode']:.0f} ± "
            f"{p['sigma']:.0f}{note}    →  Write NIfTI... to stream the full "
            f"resolution field"
        )
        self._draw_bin(p["sample"], p["threshold"], p["method"], p["slice"],
                       p["mask"], p["z"], extra="  (preview)")

    def _show_bin_result(self, r):
        bits = [
            f"{r.method} threshold {r.threshold:.1f}",
            f"material {r.material_fraction_of_box * 100:.2f} % of the box",
            f"shape {r.shape[0]}x{r.shape[1]}x{r.shape[2]}",
        ]
        if r.porosity_inside_specimen is not None:
            bits.append(
                f"porosity {r.porosity_inside_specimen * 100:.2f} % inside "
                f"the specimen")
        if r.material_components:
            bits.append(f"{r.material_components} material "
                        f"{'body' if r.material_components == 1 else 'bodies'}")
        if r.separability_warning:
            bits.append(r.separability_warning)
        self._bin_summary.setText("    ".join(bits))
        self._draw_bin(r.sample, r.threshold, r.method, r.preview_slice,
                       r.preview_mask, r.preview_z)


def nib_load_mask(path):
    """Read a written indicator volume back as a uint8 array."""
    from ..deps import nib
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.uint8)
