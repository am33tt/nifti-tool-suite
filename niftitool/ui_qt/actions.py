"""High-level user-triggered actions.

Mixin holding every ``_do_*`` and ``_export_*`` handler, connecting the
controls panel to the functions in :mod:`niftitool.core`. Heavy actions run
in a daemon thread and marshal results back onto the Qt main thread via
:meth:`NiftiApp.after`.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..config import STATS_SUBSAMPLE_VOXELS
from ..core.geometry import (
    run_angle_rotation, run_cropper, run_cropper_streaming, run_reorientation,
)
from ..core.histogram import compute_histogram
from ..core.io import (
    LazyGrayVolume, load_nifti, preview_volume, raw_array, run_gunzip, to_gray,
)
from ..core.metadata import collect_metadata, get_axis_labels, read_metadata
from ..deps import HAS_NIBABEL, nib, np
from ..utils import available_ram_mb, total_ram_mb


class ActionsMixin:
    """All user-action handlers.

    Uses the state initialised by :class:`NiftiApp`: ``_img``, ``_gray``,
    ``_path``, ``_porosity``, ``_axis_labels``, ``_ww``, ``_wc`` and
    ``_slice_cache``.
    """

    # File loading

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NIfTI file", "",
            "NIfTI files (*.nii *.nii.gz);;All files (*.*)",
        )
        if path:
            self._load_path(path)

    def _load_path(self, path):
        if not HAS_NIBABEL:
            QMessageBox.critical(self, "Missing dependency",
                                 "nibabel is not installed.")
            return
        for attr in ('_gray', '_img'):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        gc.collect()
        self._slice_cache.set_volume(None)
        self._reset_tri_artists()
        self._auto_wwwc = None      # new volume → new auto window
        # Re-arm the auto threshold and window. A stale W/C from the previous
        # file can render the new volume transparent in the 3-D view.
        try:
            self._void_thresh_var.set("auto")
            self._reset_window()
        except Exception:
            pass
        self._log_sep(f"Loading: {Path(path).name}")

        # nibabel mmaps plain .nii but must decompress .nii.gz fully into RAM.
        try:
            sz_mb = Path(path).stat().st_size / (1024 ** 2)
            if path.lower().endswith('.gz') and sz_mb > 500:
                self._append_log(
                    f"  Large gzipped file ({sz_mb:.0f} MiB). nibabel must "
                    f"decompress to RAM before any read. Consider the "
                    f"Gunzip button on the top bar first for lower memory "
                    f"use.",
                    'warn',
                )
        except OSError:
            pass

        # On RAM-constrained machines a large volume may be pre-cropped to a
        # sidecar file, which is loaded in place of the original.
        cropped_path = self._maybe_offer_precrop(path)
        if cropped_path is not None:
            path = cropped_path

        self._status_var.set(f"Opening {Path(path).name}...")
        self._prog.begin_staged()
        self._run_task("Load volume", lambda: self._staged_load(path))

    # Large-file pre-crop

    #: File size above which we start offering the pre-load crop dialog.
    _PRECROP_FILE_MB = 1500.0
    #: Only prompt when installed RAM is at or below this (GiB).
    _PRECROP_RAM_GB_CAP = 16.0

    def _maybe_offer_precrop(self, path: str) -> str | None:
        """Return the path of a cropped sidecar, or ``None`` to load *path* as-is.

        Offered only for files above ``_PRECROP_FILE_MB`` on systems with at
        most ``_PRECROP_RAM_GB_CAP`` of RAM. ``nib.load`` returns a proxy, so
        the crop reads only the needed bytes through the file mmap.
        """
        try:
            sz_mb = Path(path).stat().st_size / (1024 ** 2)
        except OSError:
            return None
        if sz_mb < self._PRECROP_FILE_MB:
            return None

        total_mb = total_ram_mb()
        if total_mb is None or total_mb > self._PRECROP_RAM_GB_CAP * 1024.0:
            return None
        total_gb = total_mb / 1024.0

        try:
            img_hdr = nib.load(str(path))
            shape = tuple(int(d) for d in img_hdr.shape[:3])
        except Exception as ex:
            self._append_log(f"  Header peek failed: {ex}", 'warn')
            return None

        reply = QMessageBox.question(
            self,
            "Large file detected",
            f"{Path(path).name} is {sz_mb:.0f} MiB and this machine has only "
            f"{total_gb:.1f} GiB of RAM.\n\n"
            f"Current shape (X, Y, Z): {shape}\n\n"
            f"Would you like to crop it to a smaller region before loading? "
            f"The cropped copy is saved alongside the original and loaded "
            f"in place of it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return None

        ranges = self._ask_precrop_ranges(shape)
        if ranges is None:
            return None
        x_range, y_range, z_range = ranges

        try:
            self._append_log(
                f"  Pre-load crop: X{x_range} Y{y_range} Z{z_range}", 'info',
            )
            self._status_var.set("Cropping large file...")
            cropped = run_cropper(img_hdr, x_range, y_range, z_range)

            src = Path(path)
            stem = src.name
            for ext in (".nii.gz", ".nii"):
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            out_path = src.with_name(f"{stem}_cropped.nii.gz")
            n = 2
            while out_path.exists():
                out_path = src.with_name(f"{stem}_cropped_{n}.nii.gz")
                n += 1
            nib.save(cropped, str(out_path))
            self._append_log(f"  Saved cropped file: {out_path.name}", 'ok')
            return str(out_path)
        except Exception as ex:
            self._append_log(f"  Pre-crop failed: {ex}", 'err')
            QMessageBox.warning(
                self, "Crop failed",
                f"Could not crop the file:\n{ex}\n\nLoading the original instead.",
            )
            return None

    def _ask_precrop_ranges(self, shape):
        """Modal with three (start, end) spin-box pairs. Returns
        ``((x0, x1), (y0, y1), (z0, z1))`` or ``None`` on Cancel."""
        from PyQt6.QtWidgets import (
            QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
            QSpinBox, QVBoxLayout, QWidget,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Pre-load crop ranges")
        root = QVBoxLayout(dlg)
        root.addWidget(QLabel(
            f"Voxel-index ranges (half-open: end is exclusive).\n"
            f"Full shape: {shape}"
        ))
        form = QFormLayout()
        root.addLayout(form)

        spin_pairs = {}
        for i, ax in enumerate(('X', 'Y', 'Z')):
            dim = shape[i]
            row = QWidget(dlg)
            rlay = QHBoxLayout(row)
            rlay.setContentsMargins(0, 0, 0, 0)
            s0 = QSpinBox(row); s0.setRange(0, dim - 1); s0.setValue(0)
            s1 = QSpinBox(row); s1.setRange(1, dim);     s1.setValue(dim)
            rlay.addWidget(s0)
            rlay.addWidget(QLabel("to", row))
            rlay.addWidget(s1)
            form.addRow(f"{ax} (0 - {dim}):", row)
            spin_pairs[ax] = (s0, s1)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        root.addWidget(bb)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        out = []
        for ax in ('X', 'Y', 'Z'):
            s0, s1 = spin_pairs[ax]
            a, b = int(s0.value()), int(s1.value())
            if b <= a:
                QMessageBox.warning(
                    self, "Invalid range",
                    f"{ax}: end ({b}) must be greater than start ({a}).",
                )
                return None
            out.append((a, b))
        return tuple(out)

    def _staged_load(self, path):
        try:
            img = load_nifti(path)
            if self.cancel_requested():
                self._append_log("  Load cancelled.", 'warn')
                self.after(0, self._prog.stop)
                self.after(0, lambda: self._status_var.set("Stopped."))
                return
            self.after(0, self._prog.advance)

            gray = LazyGrayVolume(img.dataobj)
            self.after(0, self._prog.advance)

            self._img = img
            self._path = Path(path)
            self._gray = gray
            self._ww = None; self._wc = None
            self._axis_labels = get_axis_labels(img.affine)
            self._slice_cache.set_volume(gray)
            # Start the interactive preview once the volume is bound. It
            # runs in the background: dragging works from the first frame,
            # it just gets faster when this lands.
            self.after(0, self.start_tri_preview)
            self.after(0, self._prog.advance)

            shape_str = str(img.shape)
            dtype_str = str(img.get_data_dtype())
            spacing = img.header.get_zooms()
            phys = tuple(
                round(float(img.shape[i]) * float(spacing[i]), 2)
                for i in range(min(3, len(img.shape)))
            )
            self.after(0, lambda: self._info_var.set(
                f"{Path(path).name}   |   {shape_str}   |   {dtype_str}   |   {phys} mm"
            ))
            self.after(0, lambda: self._ww_var.set("auto"))
            self.after(0, lambda: self._wc_var.set("auto"))
            self._append_log(f"  Loaded (lazy): {shape_str}  dtype={dtype_str}", 'ok')
            self._append_log(f"  Physical size: {phys} mm", 'dim')

            self.after(0, self._autofill_crop)
            self.after(0, self._update_simulation_input_defaults)
            self.after(0, self._update_tri_sliders)
            self.after(0, self._refresh_triplanar)
            self.after(0, self._prog.advance)

            self.after(80, self._update_stats)
            self.after(120, self._prog.advance)
            self.after(300, self._prog.done)
            self.after(300, lambda: self._status_var.set(f"Loaded: {Path(path).name}"))

            self._run_task("Materialise volume",
                           self._materialise_full_volume_bg)
        except Exception as ex:
            self._append_log(f"  Error loading: {ex}", 'err')
            self.after(0, lambda: self._status_var.set("Error loading file."))
            self.after(0, self._prog.stop)

    def _materialise_full_volume_bg(self):
        try:
            g = self._gray
            if not isinstance(g, LazyGrayVolume):
                return

            # Skip eager materialisation above 60% of free RAM. The lazy
            # proxy still serves slices from the file mmap and compute
            # actions materialise on demand.
            if self.cancel_requested():
                return

            est_mb = g.size * 4 / (1024 ** 2)
            free_mb = available_ram_mb()
            # free_mb is None when psutil is missing: with no RAM figure to
            # check against, materialising unconditionally would silently
            # drop the safety margin this method exists to enforce, so an
            # unknown budget is treated the same as an insufficient one.
            if free_mb is None or est_mb > 0.60 * free_mb:
                budget_str = f"{free_mb:.0f} MiB free" if free_mb is not None \
                    else "free RAM unknown (psutil not installed)"
                self._append_log(
                    f"  Volume is {est_mb:.0f} MiB float32 vs {budget_str}, "
                    f"staying on the lazy proxy. Slicing reads through the "
                    f"file mmap; compute steps will still materialise on "
                    f"demand.",
                    'warn',
                )
                return

            arr = g.to_array()

            def _swap():
                if self._gray is g:
                    self._gray = arr
                    self._slice_cache.set_volume(arr)
                    self._slice_cache.invalidate()
                    self._append_log(
                        "  Volume materialised - interactive mode.", 'dim',
                    )
                    notify = getattr(self, '_on_volume_materialised', None)
                    if callable(notify):
                        notify()
            self.after(0, _swap)
        except MemoryError:
            self._append_log(
                "  Not enough RAM to materialise full volume, staying lazy.",
                'warn',
            )
        except Exception as ex:
            self._append_log(f"  Background materialisation error: {ex}", 'err')

    def _get_gray(self):
        if self._gray is None and self._img is not None:
            self._gray = LazyGrayVolume(self._img.dataobj)
            self._slice_cache.set_volume(self._gray)
            self.after(0, self.start_tri_preview)
            self.after(0, self._update_tri_sliders)
        if isinstance(self._gray, LazyGrayVolume):
            return self._gray.to_array()
        return self._gray

    def _get_gray_lazy(self):
        """Like :meth:`_get_gray` but never materialises the volume.

        Returns an ndarray or a ``LazyGrayVolume``. Both support ``shape``
        and per-slice ``[:, :, z]`` access, which is all the streaming paths
        need.
        """
        if self._gray is None and self._img is not None:
            self._gray = LazyGrayVolume(self._img.dataobj)
            self._slice_cache.set_volume(self._gray)
            self.after(0, self.start_tri_preview)
            self.after(0, self._update_tri_sliders)
        return self._gray

    def _resolve_void_thresh(self):
        """Current void threshold as a float, using Otsu when set to ``'auto'``.

        The computed value is written back into the field and logged. Returns
        None when the field is on auto and no file is loaded.
        """
        txt = str(self._void_thresh_var.get()).strip().lower()
        if txt not in ("", "auto"):
            try:
                return float(txt)
            except ValueError:
                pass                      # unparseable → fall back to auto
        if self._img is None:
            return None
        from ..core.segmentation import otsu_threshold

        vol = self._get_gray_lazy()
        if hasattr(vol, "subsample_flat"):
            flat = vol.subsample_flat(2_000_000)
        else:
            flat = vol.ravel()
            if flat.size > 2_000_000:
                flat = flat[:: flat.size // 2_000_000]
        t = float(otsu_threshold(flat))
        self._void_thresh_var.set(f"{t:.1f}")
        self._append_log(
            f"  Void threshold: auto (Otsu) → {t:.1f}  "
            f"(sampled range {float(flat.min()):.0f} to {float(flat.max()):.0f})",
            'teal',
        )
        return t

    def _sane_void_thresh(self, thresh: float) -> float:
        """Sanity-check *thresh* against the actual intensity distribution.

        A threshold that selects almost nothing or almost everything is the
        usual symptom of a value carried over from another scan, so the Otsu
        split is offered instead. Must run on the GUI thread. Returns the
        threshold to use.
        """
        try:
            vol = self._get_gray_lazy()
            if vol is None:
                return thresh
            if hasattr(vol, "subsample_flat"):
                flat = vol.subsample_flat(2_000_000)
            else:
                flat = vol.ravel()
                if flat.size > 2_000_000:
                    flat = flat[:: flat.size // 2_000_000]
            frac = float((flat < thresh).mean())
            if 1e-4 < frac < 0.999:
                return thresh

            from ..core.segmentation import otsu_threshold
            t_auto = otsu_threshold(flat)
            mn, mx = float(flat.min()), float(flat.max())
            what = "almost nothing" if frac <= 1e-4 else "almost everything"
            reply = QMessageBox.question(
                self, "Threshold looks wrong",
                f"The void threshold {thresh:g} selects {what} "
                f"({frac * 100:.3f} % of sampled voxels).\n\n"
                f"Your volume's intensity range is about "
                f"{mn:.0f} to {mx:.0f}, so this threshold probably "
                f"doesn't match the data.\n\n"
                f"Use the automatic (Otsu) threshold  {t_auto:.1f}  "
                f"instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._void_thresh_var.set(f"{t_auto:.1f}")
                self._append_log(
                    f"  Void threshold auto-corrected: {thresh:g} → "
                    f"{t_auto:.1f}  (Otsu; volume range {mn:.0f}-{mx:.0f})",
                    'teal',
                )
                return float(t_auto)
            return thresh
        except Exception:
            return thresh

    def _require_img(self) -> bool:
        if self._img is None:
            QMessageBox.warning(self, "No file", "Please open a NIfTI file first.")
            return False
        return True

    def _ask_save_path(self, default_name: str, ext=".nii.gz", ftypes=None):
        filters = "NIfTI gz (*.nii.gz);;NIfTI (*.nii);;All (*.*)"
        if ftypes is not None:
            # Convert tk tuples like [("CSV","*.csv"),...] to Qt filter string.
            filters = ";;".join(f"{name} ({patt})" for name, patt in ftypes)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output as...", default_name, filters,
        )
        return path or None

    # Stats (live)

    def _update_stats(self):
        if self._gray is None:
            return
        g = self._gray
        if isinstance(g, LazyGrayVolume):
            flat = g.subsample_flat(STATS_SUBSAMPLE_VOXELS)
        else:
            flat = g.ravel()
            if flat.size > STATS_SUBSAMPLE_VOXELS:
                step = flat.size // STATS_SUBSAMPLE_VOXELS
                flat = flat[::step]
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

    # Standard actions

    def _do_gunzip(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select .gz file", "",
            "GZipped files (*.gz);;All files (*.*)",
        )
        if not path:
            return

        def _run():
            try:
                self._log_sep(f"Gunzip: {Path(path).name}")
                self._set_status("Decompressing...", busy=True)
                out = run_gunzip(path)
                if self._abort_if_stopped("Decompression"):
                    return
                self._append_log(f"  → {out}", 'ok')
                self._set_status("Gunzip complete.", busy=False)
                self.after(0, lambda: self._ask_and_load_gunzip(out))
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Gunzip failed.", busy=False)

        self._run_task("Decompress", _run)

    def _ask_and_load_gunzip(self, out):
        self._ask_load_after_save("File decompressed.", out)

    def _do_metadata(self):
        if not self._require_img():
            return

        def _run():
            try:
                self._log_sep("Metadata")
                self._set_status("Reading metadata...", busy=True)
                gray_lazy = self._get_gray_lazy() if self._gray is not None else None
                sections = collect_metadata(self._img, gray_lazy)
                txt = read_metadata(self._img, gray_lazy)
                if self._abort_if_stopped("Metadata"):
                    return
                self._append_log(txt, 'dim')
                self.after(0, self.show_metadata_sections, sections)
                self.after(0, self._show_tab, 'metadata')
                self._set_status("Metadata read.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Metadata error.", busy=False)

        self._run_task("Read metadata", _run)

    def _do_histogram(self):
        if not self._require_img():
            return

        def _run():
            try:
                self._log_sep("Histogram & Brightness Analysis")
                self._set_status("Computing histogram...", busy=True)
                gray = self._get_gray_lazy()
                try:
                    n_bins = int(self._hist_bins_var.get())
                except Exception:
                    n_bins = 256
                zidx, means, counts, edges, mn, mx = compute_histogram(gray, n_bins)
                if self._abort_if_stopped("Histogram"):
                    return
                self._append_log(f"  Shape: {gray.shape}  range: {mn:.2f}-{mx:.2f}", 'dim')
                self._append_log(
                    f"  Mean: {float(gray.mean()):.4f}  "
                    f"Std: {float(gray.std()):.4f}", 'dim',
                )
                self.after(0, self._draw_histogram, counts, edges, zidx, means, mn, mx)
                self.after(0, self._update_stats)
                self.after(0, self._show_tab, 'histogram')
                self._set_status("Histogram complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Histogram error.", busy=False)

        self._run_task("Histogram", _run)

    def _do_reorient(self):
        if not self._require_img():
            return
        target = self._reorient_target.get().strip().upper()
        out_path = None
        if self._reorient_save.get():
            stem = self._path.name.replace('.nii.gz', '').replace('.nii', '')
            out_path = self._ask_save_path(f"{stem}_reoriented_{target}.nii.gz")
            if not out_path:
                return

        def _run():
            try:
                if len(target) != 3:
                    self._append_log("Target must be 3 letters (e.g. RAS).", 'warn')
                    return
                self._log_sep(f"Reorientation → {target}")
                self._set_status(f"Reorienting to {target}...", busy=True)
                result = run_reorientation(self._img, target)
                if self._abort_if_stopped("Reorientation"):
                    return
                self._append_log(f"  New shape: {result.shape}", 'ok')
                self._rotation_history.append({
                    "operation": "reorient",
                    "target_axcodes": target,
                    "resampled": False,
                    "new_shape": [int(v) for v in result.shape[:3]],
                    "saved_to": str(out_path) if out_path else None,
                })
                if out_path:
                    nib.save(result, str(out_path))
                    self._append_log(f"  Saved → {Path(out_path).name}", 'ok')
                    self._set_status("Reorientation saved.", busy=False)
                    self.after(0, lambda: self._ask_load_after_save(
                        "Reoriented volume saved.", out_path))
                else:
                    self._img = result
                    self._gray = None
                    self._slice_cache.set_volume(None)
                    self._reset_tri_artists()
                    self._axis_labels = get_axis_labels(result.affine)
                    self._set_status("Reorientation done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Reorientation error.", busy=False)

        self._run_task("Reorient", _run)

    def _ask_load_after_save(self, summary: str, out_path):
        """Report a saved file and offer to load it.

        *summary* says what was done, in one sentence ("Background
        removed."); the destination is taken from *out_path* and rendered
        here. Callers must not put the path in *summary* -- an earlier
        version left one caller passing a sentence ending in "saved to:"
        with the path supplied separately and never displayed, so the
        dialog asked a question it had not finished writing.

        The file name and its folder go on separate lines, because a
        Windows path is long enough to wrap mid-directory and the name is
        the part being looked for. A minimum width is forced through the
        layout, since QMessageBox otherwise shrinks to the title and
        squeezes the path into a column a few words wide.
        """
        from PyQt6.QtWidgets import QSizePolicy, QSpacerItem
        from PyQt6.QtCore import Qt

        out_path = Path(out_path)
        while True:
            box = QMessageBox(self)
            box.setWindowTitle("Saved")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(summary)
            box.setInformativeText(
                f"{out_path.name}\nin {out_path.parent}\n\nLoad it now?")
            # The path is the reason this dialog exists; let it be selected.
            box.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)

            load = box.addButton("Load", QMessageBox.ButtonRole.AcceptRole)
            copy = box.addButton("Copy path", QMessageBox.ButtonRole.ActionRole)
            close = box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(load)
            box.setEscapeButton(close)

            layout = box.layout()
            if layout is not None:
                layout.addItem(
                    QSpacerItem(520, 0, QSizePolicy.Policy.Minimum,
                                QSizePolicy.Policy.Expanding),
                    layout.rowCount(), 0, 1, layout.columnCount(),
                )

            box.exec()
            clicked = box.clickedButton()
            if clicked is copy:
                # Copying should not dismiss the offer to load, so the
                # dialog is rebuilt rather than closed.
                self.clipboard_clear()
                self.clipboard_append(str(out_path))
                self._append_log(f"  Path copied: {out_path}", 'dim')
                continue
            if clicked is load:
                self._load_path(str(out_path))
            return

    def _do_rotate(self):
        if not self._require_img():
            return
        try:
            axis = self._rot_axis.get()
            angle = float(self._rot_angle.get())
        except ValueError:
            QMessageBox.critical(self, "Bad input", "Angle must be a number.")
            return
        out_path = None
        if self._rot_save.get():
            stem = self._path.name.replace('.nii.gz', '').replace('.nii', '')
            out_path = self._ask_save_path(f"{stem}_rot_{axis}_{int(angle)}deg.nii.gz")
            if not out_path:
                return

        def _run():
            try:
                self._log_sep(f"Rotation  axis={axis}  angle={angle}°")
                self._set_status(f"Rotating {angle}° around {axis.upper()}...", busy=True)
                result = run_angle_rotation(self._img, axis, angle)
                if self._abort_if_stopped("Rotation"):
                    return
                self._append_log(f"  New shape: {result.shape}", 'ok')
                self._rotation_history.append({
                    "operation": "rotate",
                    "axis": str(axis),
                    "angle_deg": float(angle),
                    "resampled": True,
                    "new_shape": [int(v) for v in result.shape[:3]],
                    "saved_to": str(out_path) if out_path else None,
                })
                if out_path:
                    nib.save(result, str(out_path))
                    self._set_status("Rotation saved.", busy=False)
                    self.after(0, lambda: self._ask_load_after_save(
                        "Rotated volume saved.", out_path))
                else:
                    self._img = result
                    self._gray = None
                    self._slice_cache.set_volume(None)
                    self._reset_tri_artists()
                    self._set_status("Rotation done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Rotation error.", busy=False)

        self._run_task("Rotate", _run)

    def _do_crop(self):
        if not self._require_img():
            return
        try:
            ranges = {
                ax: (int(self._crop_rows[ax].start_var.get()),
                     int(self._crop_rows[ax].end_var.get()))
                for ax in ('X', 'Y', 'Z')
            }
        except ValueError:
            QMessageBox.critical(self, "Bad input",
                                 "Crop start/end must be integers.")
            return
        out_path = None
        if self._crop_save.get():
            stem = self._path.name.replace('.nii.gz', '').replace('.nii', '')
            out_path = self._ask_save_path(f"{stem}_cropped.nii.gz")
            if not out_path:
                return

        def _run():
            try:
                self._log_sep(
                    f"Crop  X{ranges['X']}  Y{ranges['Y']}  Z{ranges['Z']}"
                )
                self._set_status("Cropping...", busy=True)
                if out_path:
                    # Streamed straight to disk: working set is one slab,
                    # not the cropped volume, so this doesn't hand a huge
                    # in-memory array to a non-streaming nib.save (the
                    # WinError 1455 / commit-limit failure on a large crop).
                    out_shape = run_cropper_streaming(
                        self._img, ranges['X'], ranges['Y'], ranges['Z'],
                        out_path,
                        progress=lambda stage, done: self._set_status(
                            f"Crop: {stage} {done:.0%}", busy=True,
                        ),
                        cancel=self.cancel_requested,
                    )
                    if self._abort_if_stopped("Crop"):
                        return
                    self._append_log(f"  Cropped shape: {out_shape}", 'ok')
                    self._set_status("Crop saved.", busy=False)
                    self.after(0, lambda: self._ask_load_after_save(
                        "Cropped volume saved.", out_path))
                else:
                    # Kept in the session rather than saved: the point is a
                    # smaller resident array to keep working with, so this
                    # path materialises (it always crops the volume down,
                    # never up).
                    result = run_cropper(
                        self._img, ranges['X'], ranges['Y'], ranges['Z'])
                    if self._abort_if_stopped("Crop"):
                        return
                    self._append_log(f"  Cropped shape: {result.shape}", 'ok')
                    self._img = result
                    self._gray = None
                    self._slice_cache.set_volume(None)
                    self._reset_tri_artists()
                    self._set_status("Crop done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Crop error.", busy=False)

        self._run_task("Crop", _run)

    # Material mapping actions

    # Background removal

    def _do_remove_background(self):
        """Detect the specimen, strip the exterior background, save NIfTI.

        The volume is streamed: it is read in slabs and the result is
        written as it is produced, so the run needs a working set of about
        a quarter byte per voxel plus one slab, whatever the file size.
        """
        if not self._require_img():
            return
        thresh = self._resolve_void_thresh()
        if thresh is None:
            QMessageBox.critical(self, "No threshold",
                                 "Could not determine a threshold.")
            return
        try:
            margin = max(0, int(float(self._bg_margin_widget.text())))
        except Exception:
            margin = 10
        crop = self._bg_crop_widget.isChecked()
        boundary = self._bg_boundary_combo.currentData()
        fill_mode = self._bg_fill_combo.currentData()
        fill_value = None
        if fill_mode == "value":
            try:
                fill_value = float(self._bg_fill_widget.text())
            except ValueError:
                QMessageBox.critical(self, "Bad input",
                                     "The fill grey level must be a number.")
                return

        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '') \
            if self._path else "volume"
        out = self._ask_save_path(f"{stem}_nobg.nii.gz")
        if not out:
            return

        def _run():
            from ..core.background import (
                estimate_memory_bytes, remove_background,
            )
            from ..utils import OperationCancelled

            try:
                self._log_sep("Background Removal")
                self._log_memory_budget(estimate_memory_bytes)
                info = remove_background(
                    self._img, thresh, out,
                    margin=margin, crop=crop, boundary=boundary,
                    fill_mode=fill_mode, fill_value=fill_value,
                    progress=lambda stage, done: self._set_status(
                        f"Background: {stage} {done:.0%}", busy=True,
                    ),
                    cancel=self.cancel_requested,
                )
                self._append_log(
                    f"  Threshold          : {info['threshold']:.1f}", 'dim')
                self._log_envelope(info.get('envelope'))
                self._append_log(
                    f"  Solid components   : {info['solid_components']} "
                    f"(largest kept)", 'dim')
                self._append_log(
                    f"  Cavities filled    : {info['cavity_voxels']:,} voxels",
                    'dim')
                self._append_log(
                    f"  Background removed : {info['background_pct']:.1f}% "
                    f"of voxels  (fill = {info['fill_value']:g}, "
                    f"{info['fill_mode']})", 'teal')
                if info['bbox'] is not None:
                    self._append_log(
                        f"  Cropped to         : {info['out_shape']}", 'teal')
                self._append_log(
                    f"  Voxels kept as     : {info['dtype']}, unchanged", 'dim')
                self._append_log(f"  Saved → {Path(out).name}", 'ok')
                self._set_status("Background removed.", busy=False)
                self.after(0, self._ask_load_after_save,
                           "Background removed.", Path(out))
            except OperationCancelled:
                if not self._abort_if_stopped("Background removal"):
                    self._append_log("  Background removal cancelled.", 'warn')
                    self._set_status("Stopped.", busy=False)
            except Exception as ex:
                self._append_log(f"  Background removal error: {ex}", 'err')
                self._set_status("Background removal error.", busy=False)

        self._run_task("Remove background", _run)

    def _log_envelope(self, envelope):
        """Report the fitted specimen boundary, and what it cut.

        The cut figure is the check worth reading: it should account for the
        sleeve or mould and nothing else. A large one means the fit has
        clipped the specimen, and the run should be repeated with the
        largest-solid-body boundary or a wider margin.
        """
        if not envelope:
            return
        radius = envelope['radius_px']
        self._append_log(
            f"  Envelope radius    : {radius['median']:.1f} px "
            f"(p5 {radius['p5']:.1f}, p95 {radius['p95']:.1f}); "
            f"mean diameter {envelope['diameter_px']['median']:.1f} px",
            'teal',
        )
        drift = envelope['axis_drift_px']
        self._append_log(
            f"  Axis drift         : {drift[0]:.1f} x {drift[1]:.1f} px "
            f"over the volume; {envelope['rays_replaced']:,} of "
            f"{envelope['rays_total']:,} rays crossed a contact and were "
            f"fitted", 'dim',
        )
        self._append_log(
            f"  Envelope removed   : {envelope['cut_solid_voxels']:,} solid "
            f"voxels ({envelope['cut_pct_of_solid']:.1f}% of the material "
            f"found) — check this is the sleeve and not the specimen",
            'teal',
        )

    def _log_memory_budget(self, estimator):
        """Log what the streaming run will need against what is free."""
        itemsize = 2
        try:
            itemsize = np.dtype(self._img.get_data_dtype()).itemsize
        except Exception:
            pass
        needed_gb = estimator(self._img.shape, itemsize) / 1024 ** 3
        free_mb = available_ram_mb()
        free = "unknown" if free_mb is None else f"{free_mb / 1024:.1f} GiB"
        self._append_log(
            f"  Working set        : ~{needed_gb:.2f} GiB "
            f"({free} free); the volume is streamed, not loaded.", 'dim',
        )

    # PDF report

    def _do_generate_report(self):
        """Bundle everything computed so far into a multi-page PDF.

        Only the loaded volume is required; the other sections are included
        when their data exists.
        """
        if not self._require_img():
            return
        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '') \
            if self._path else "volume"
        out = self._ask_save_path(
            f"{stem}_report.pdf", ext=".pdf",
            ftypes=[("PDF", "*.pdf"), ("All", "*.*")],
        )
        if not out:
            return

        # Snapshot GUI-owned state on the main thread.
        void_thresh = None
        try:
            void_thresh = float(self._void_thresh_var.get())
        except Exception:
            pass

        def _run():
            try:
                from ..core.report import generate_pdf_report
                from ..utils import OperationCancelled

                self._log_sep("PDF Report")
                self._set_status("Generating PDF report...", busy=True)

                def _stage(st):
                    if self.cancel_requested():
                        raise OperationCancelled()
                    self._set_status(f"Report: {st}...", busy=True)
                # Figures never need full resolution; stream a strided preview.
                max_prev = 150_000_000
                avail = available_ram_mb()
                if avail is not None:
                    max_prev = min(max_prev,
                                   int(avail * 0.15 * 1024 ** 2 / 4))
                gray, pstride = preview_volume(
                    self._get_gray_lazy(), max_voxels=max(max_prev, 1_000_000),
                )
                if pstride > 1:
                    self._append_log(
                        f"  Volume read at stride ×{pstride} for the "
                        f"report figures.", 'dim',
                    )

                skipped = []
                if getattr(self, '_void_result', None) is None:
                    skipped.append("voids")
                if getattr(self, '_bh_fit', None) is None:
                    skipped.append("beam hardening")
                if getattr(self, '_pore_threshold_result', None) is None:
                    skipped.append("pore threshold")
                if skipped:
                    self._append_log(
                        f"  Not computed, skipping section(s): "
                        f"{', '.join(skipped)}", 'warn',
                    )

                n_pages = generate_pdf_report(
                    out,
                    file_name=self._path.name if self._path else "volume",
                    shape=self._img.shape[:3],
                    spacing=self._img.header.get_zooms()[:3],
                    dtype=str(self._img.get_data_dtype()),
                    gray=gray,
                    ww=self._ww, wc=self._wc,
                    void_result=getattr(self, '_void_result', None),
                    void_thresh=void_thresh,
                    bin_result=getattr(self, '_bin_result', None),
                    bh_fit=getattr(self, '_bh_fit', None),
                    pore_threshold=getattr(self, '_pore_threshold_result',
                                           None),
                    pore_sensitivity=(
                        (getattr(self, '_pore_threshold_info', None) or {})
                        .get('porosity_sensitivity_pct')
                    ),
                    progress=_stage,
                )
                self._append_log(
                    f"  {n_pages} pages → {Path(out).name}", 'ok',
                )
                self._set_status("PDF report saved.", busy=False)
            except Exception as ex:
                from ..utils import OperationCancelled
                if isinstance(ex, OperationCancelled):
                    self._append_log("  Report stopped by user.", 'warn')
                    self._set_status("Stopped.", busy=False)
                else:
                    self._append_log(f"  Report error: {ex}", 'err')
                    self._set_status("Report error.", busy=False)

        self._run_task("PDF report", _run)
