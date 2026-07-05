"""High-level user-triggered actions.

This mixin contains every ``_do_*`` and ``_export_*`` handler - the
glue between the controls panel and the pure :mod:`niftitool.core`
functions.  Actions that do heavy lifting run in a daemon thread and
marshal results back onto the Qt main thread via :meth:`NiftiApp.after`
(which is a tk-compat shim around :class:`QTimer` / a signal).
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..config import STATS_SUBSAMPLE_VOXELS
from ..core.geometry import run_angle_rotation, run_cropper, run_reorientation
from ..core.histogram import compute_histogram
from ..core.io import (
    LazyGrayVolume, load_nifti, preview_volume, raw_array, run_gunzip, to_gray,
)
from ..core.mapping import compute_E_map
from ..core.metadata import collect_metadata, get_axis_labels, read_metadata
from ..core.segmentation import phase_statistics, segment_phases
from ..deps import HAS_NIBABEL, nib, np
from ..utils import available_ram_mb, total_ram_mb


class ActionsMixin:
    """All user-action handlers.

    Relies on the following state initialised by :class:`NiftiApp`:
      ``_img, _gray, _path, _hu_vol, _E_map, _labels, _hu_cal,
      _E_stats, _porosity, _axis_labels, _ww, _wc, _slice_cache``.
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
        for attr in ('_gray', '_img', '_hu_vol', '_E_map', '_labels'):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        gc.collect()
        self._slice_cache.set_volume(None)
        self._reset_tri_artists()
        self._hu_cal = {}
        # New file → new intensity distribution: re-arm the auto threshold
        # and reset the display window. A stale Window W/C from the
        # previous file can map the new data entirely transparent in the
        # 3-D view ("volume disappears").
        try:
            self._void_thresh_var.set("auto")
            self._reset_window()
        except Exception:
            pass
        self._log_sep(f"Loading: {Path(path).name}")

        # nibabel can mmap plain .nii but has to decompress .nii.gz entirely
        # into RAM. For large gzipped files that's the quickest way to OOM;
        # steer the user at the Gunzip button early.
        try:
            sz_mb = Path(path).stat().st_size / (1024 ** 2)
            if path.lower().endswith('.gz') and sz_mb > 500:
                self._append_log(
                    f"  Large gzipped file ({sz_mb:.0f} MiB). nibabel must "
                    f"decompress to RAM before any read — consider the "
                    f"Gunzip button on the top bar first for lower memory "
                    f"use.",
                    'warn',
                )
        except OSError:
            pass

        # On RAM-constrained machines, offer to pre-crop huge volumes so
        # the user never has to materialise them fully. If they accept,
        # we swap `path` for the cropped sidecar and continue.
        cropped_path = self._maybe_offer_precrop(path)
        if cropped_path is not None:
            path = cropped_path

        self._status_var.set(f"Opening {Path(path).name}...")
        self._prog.begin_staged()
        threading.Thread(target=self._staged_load, args=(path,), daemon=True).start()

    # Large-file pre-crop

    #: File size above which we start offering the pre-load crop dialog.
    _PRECROP_FILE_MB = 1500.0
    #: Only prompt when installed RAM is at or below this (GiB).
    _PRECROP_RAM_GB_CAP = 16.0

    def _maybe_offer_precrop(self, path: str) -> str | None:
        """Return a path to a cropped sidecar the user asked us to create,
        or ``None`` to mean "just load *path* as-is".

        Triggered only when the file is large (>1.5 GiB) *and* the system
        has ≤16 GiB of RAM. Peeking the NIfTI header via ``nib.load`` does
        not materialise the volume — nibabel returns a proxy, so the crop
        reads only the needed bytes through the file mmap.
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
            # Strip both ``.nii`` and ``.nii.gz`` for a clean stem.
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
            form.addRow(f"{ax} (0 – {dim}):", row)
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
            self.after(0, self._prog.advance)

            gray = LazyGrayVolume(img.dataobj)
            self.after(0, self._prog.advance)

            self._img = img
            self._path = Path(path)
            self._gray = gray
            self._ww = None; self._wc = None
            self._axis_labels = get_axis_labels(img.affine)
            self._slice_cache.set_volume(gray)
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
            self.after(0, self._update_tri_sliders)
            self.after(0, self._refresh_triplanar)
            self.after(0, self._prog.advance)

            self.after(80, self._update_stats)
            self.after(120, self._prog.advance)
            self.after(300, self._prog.done)
            self.after(300, lambda: self._status_var.set(f"Loaded: {Path(path).name}"))

            threading.Thread(
                target=self._materialise_full_volume_bg, daemon=True,
            ).start()
        except Exception as ex:
            self._append_log(f"  Error loading: {ex}", 'err')
            self.after(0, lambda: self._status_var.set("Error loading file."))
            self.after(0, self._prog.stop)

    def _materialise_full_volume_bg(self):
        try:
            g = self._gray
            if not isinstance(g, LazyGrayVolume):
                return

            # Estimate the float32 footprint of the full volume and skip the
            # eager materialisation if it would consume more than ~60% of
            # free RAM. Staying on the lazy proxy keeps the slice viewer
            # responsive (nibabel serves slices straight from the mmap);
            # the heavy compute actions (`_get_gray`) still materialise on
            # demand, which is where the user is paying for it anyway.
            est_mb = g.size * 4 / (1024 ** 2)
            free_mb = available_ram_mb()
            if free_mb is not None and est_mb > 0.60 * free_mb:
                self._append_log(
                    f"  Volume is {est_mb:.0f} MiB float32 vs {free_mb:.0f} "
                    f"MiB free — staying on the lazy proxy. Slicing reads "
                    f"through the file mmap; compute steps will still "
                    f"materialise on demand.",
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
                "  Not enough RAM to materialise full volume — staying lazy.",
                'warn',
            )
        except Exception as ex:
            self._append_log(f"  Background materialisation error: {ex}", 'err')

    def _get_gray(self):
        if self._gray is None and self._img is not None:
            self._gray = LazyGrayVolume(self._img.dataobj)
            self._slice_cache.set_volume(self._gray)
            self.after(0, self._update_tri_sliders)
        if isinstance(self._gray, LazyGrayVolume):
            return self._gray.to_array()
        return self._gray

    def _get_gray_lazy(self):
        """Like :meth:`_get_gray` but never materialises the volume.

        Returns whatever ``self._gray`` is (ndarray or LazyGrayVolume) —
        both support ``shape`` and per-slice ``[:, :, z]`` access, which
        is all the streaming code paths need.  Use this from anything
        that must work on volumes larger than RAM.
        """
        if self._gray is None and self._img is not None:
            self._gray = LazyGrayVolume(self._img.dataobj)
            self._slice_cache.set_volume(self._gray)
            self.after(0, self._update_tri_sliders)
        return self._gray

    def _ram_guard(self, bytes_per_voxel: float, what: str) -> bool:
        """Return True if a full-resolution *what* fits in RAM, else warn.

        Estimates ``voxels × bytes_per_voxel`` against available memory
        and pops an actionable message instead of letting numpy die with
        an opaque MemoryError.
        """
        if self._img is None:
            return False
        vox = 1
        for s in self._img.shape[:3]:
            vox *= int(s)
        needed_mb = vox * bytes_per_voxel / 1024 ** 2
        avail_mb = available_ram_mb()
        if avail_mb is not None and needed_mb > avail_mb * 0.85:
            msg = (
                f"{what} needs ~{needed_mb / 1024:.1f} GiB at full "
                f"resolution, but only {avail_mb / 1024:.1f} GiB RAM is "
                f"free.\n\n"
                f"Options:\n"
                f"  •  Crop the volume first (Crop tool) — a region of "
                f"interest is usually enough.\n"
                f"  •  Use the Porosity tab / histogram / report — those "
                f"work at any size (they stream and downsample "
                f"automatically)."
            )
            self.after(0, lambda: QMessageBox.warning(
                self, "Not enough RAM", msg,
            ))
            self._append_log(
                f"  {what}: needs ~{needed_mb / 1024:.1f} GiB, "
                f"{avail_mb / 1024:.1f} GiB free — aborted. "
                f"Crop first or use the streaming tools.", 'warn',
            )
            return False
        return True

    def _resolve_void_thresh(self):
        """Current void threshold as a float; ``'auto'`` → Otsu.

        The computed value is written back into the field (so the user
        sees what was used and can tweak it) and logged.  Returns None
        when no file is loaded and the field is on auto.
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

        vol = self._hu_vol if self._hu_vol is not None \
            else self._get_gray_lazy()
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
            f"(sampled range {float(flat.min()):.0f} … {float(flat.max()):.0f})",
            'teal',
        )
        return t

    def _sane_void_thresh(self, thresh: float) -> float:
        """Sanity-check *thresh* against the actual intensity distribution.

        The classic trap: the HU-style default (−500) applied to an
        uncalibrated scan whose raw intensities start at 0 — the void
        mask is empty and porosity reads 0.000 %.  If the threshold
        selects (almost) nothing or (almost) everything, offer the Otsu
        split instead.  Runs on the GUI thread (shows a dialog).
        Returns the threshold to use.
        """
        try:
            vol = self._hu_vol if self._hu_vol is not None \
                else self._get_gray_lazy()
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
                f"{mn:.0f} … {mx:.0f}, so this threshold probably "
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
                    f"{t_auto:.1f}  (Otsu; volume range {mn:.0f}–{mx:.0f})",
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

    def _require_emap(self) -> bool:
        if self._E_map is None:
            QMessageBox.warning(self, "No E-Map", "Please compute the E-Map first.")
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
                self._append_log(f"  → {out}", 'ok')
                self._set_status("Gunzip complete.", busy=False)
                self.after(0, lambda: self._ask_and_load_gunzip(out))
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Gunzip failed.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    def _ask_and_load_gunzip(self, out):
        reply = QMessageBox.question(
            self, "Done", f"Decompressed to:\n{out}\n\nLoad now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._load_path(str(out))

    def _do_metadata(self):
        if not self._require_img():
            return

        def _run():
            try:
                self._log_sep("Metadata")
                self._set_status("Reading metadata...", busy=True)
                gray_lazy = self._get_gray_lazy() if self._gray is not None else None
                sections = collect_metadata(
                    self._img, gray_lazy, self._hu_vol, self._labels,
                )
                txt = read_metadata(
                    self._img, gray_lazy, self._hu_vol, self._labels,
                )
                self._append_log(txt, 'dim')
                self.after(0, self.show_metadata_sections, sections)
                self.after(0, self._show_tab, 'metadata')
                self._set_status("Metadata read.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Metadata error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

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
                self._append_log(f"  Shape: {gray.shape}  range: {mn:.2f}–{mx:.2f}", 'dim')
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

        threading.Thread(target=_run, daemon=True).start()

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
                self._append_log(f"  New shape: {result.shape}", 'ok')
                if out_path:
                    nib.save(result, str(out_path))
                    self._append_log(f"  Saved → {Path(out_path).name}", 'ok')
                    self._set_status("Reorientation saved.", busy=False)
                    self.after(0, lambda: self._ask_load_after_save(
                        "Load reoriented file now?", out_path))
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

        threading.Thread(target=_run, daemon=True).start()

    def _ask_load_after_save(self, prompt: str, out_path):
        reply = QMessageBox.question(
            self, "Saved", prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._load_path(str(out_path))

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
                self._append_log(f"  New shape: {result.shape}", 'ok')
                if out_path:
                    nib.save(result, str(out_path))
                    self._set_status("Rotation saved.", busy=False)
                    self.after(0, lambda: self._ask_load_after_save(
                        "Load rotated file now?", out_path))
                else:
                    self._img = result
                    self._gray = None
                    self._slice_cache.set_volume(None)
                    self._reset_tri_artists()
                    self._set_status("Rotation done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Rotation error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

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
                result = run_cropper(self._img, ranges['X'], ranges['Y'], ranges['Z'])
                self._append_log(f"  Cropped shape: {result.shape}", 'ok')
                if out_path:
                    nib.save(result, str(out_path))
                    self._set_status("Crop saved.", busy=False)
                    self.after(0, lambda: self._ask_load_after_save(
                        "Load cropped file now?", out_path))
                else:
                    self._img = result
                    self._gray = None
                    self._slice_cache.set_volume(None)
                    self._reset_tri_artists()
                    self._set_status("Crop done.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Crop error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    # Material mapping actions

    def _do_compute_emap(self):
        if not self._require_img():
            return
        # Resolve 'auto' before reading model params — simple mode derives
        # its bilinear split from this same field.
        void_thresh = self._resolve_void_thresh()
        if void_thresh is None:
            QMessageBox.critical(self, "Bad parameters",
                                 "Could not determine a void threshold.")
            return
        try:
            model = self._model_var.get()
            params = self._get_model_params()
            agg_str = self._agg_thresh_var.get().strip()
            agg_thresh = float(agg_str) if agg_str else None
        except Exception as ex:
            QMessageBox.critical(self, "Bad parameters", str(ex))
            return
        # gray + E_map float32 + labels uint8 + working copy ≈ 13 B/voxel.
        if not self._ram_guard(13.0, "Computing a full-resolution E-Map"):
            return
        void_thresh = self._sane_void_thresh(void_thresh)
        # Simple mode ties the bilinear split to the void threshold, so
        # keep them consistent if the threshold was just auto-corrected.
        if params.get("hu_thresh") is not None \
                and not self._mat_advanced_check.isChecked():
            params["hu_thresh"] = void_thresh

        def _run():
            try:
                self._log_sep(f"E-Map Computation  model={model}")
                self._set_status("Computing E-Map...", busy=True)
                gray = self._get_gray()
                hu_vol = self._hu_vol if self._hu_vol is not None else gray

                E_map = compute_E_map(hu_vol, model, params, void_thresh)
                labels = segment_phases(hu_vol, void_thresh, agg_thresh)
                stats, porosity = phase_statistics(hu_vol, E_map, labels)

                self._E_map = E_map
                self._labels = labels
                self._E_stats = stats
                self._porosity = porosity

                self._append_log(f"  E-Map shape : {E_map.shape}", 'teal')
                self._append_log(
                    f"  E range     : {float(E_map.min()):.2f}  –  "
                    f"{float(E_map.max()):.2f} MPa", 'teal',
                )
                self._append_log(
                    f"  E mean ± std: {float(E_map.mean()):.2f} ± "
                    f"{float(E_map.std()):.2f} MPa", 'teal',
                )
                self._append_log(f"  Porosity    : {porosity * 100:.2f}%", 'teal')

                for phase_name, s in stats.items():
                    self._append_log(
                        f"  {phase_name:20s}  vol={s['vol_frac'] * 100:.2f}%  "
                        f"E={s['E_mean_MPa']:.1f} MPa", 'dim',
                    )

                def _upd():
                    self._e_stat_labels["E_min"].set(f"{float(E_map.min()):.2f} MPa")
                    self._e_stat_labels["E_max"].set(f"{float(E_map.max()):.2f} MPa")
                    self._e_stat_labels["E_mean"].set(f"{float(E_map.mean()):.2f} MPa")
                    self._e_stat_labels["E_std"].set(f"{float(E_map.std()):.2f} MPa")
                    self._e_stat_labels["porosity"].set(f"{porosity * 100:.2f}%")

                self.after(0, _upd)
                self.after(0, self._update_emap_sliders)
                self.after(50, self._refresh_emap_viewer)
                self.after(50, self._show_tab, 'emap')
                self._set_status("E-Map complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  E-Map error: {ex}", 'err')
                self._set_status("E-Map error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    # Exports

    def _export_labels_nifti(self):
        if self._labels is None:
            QMessageBox.warning(self, "No labels", "Compute E-Map first.")
            return
        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '')
        out = self._ask_save_path(f"{stem}_phases.nii.gz")
        if not out:
            return

        def _run():
            try:
                self._log_sep("Export Phase Labels NIfTI")
                self._set_status("Saving labels...", busy=True)
                hdr = self._img.header.copy()
                hdr.set_data_dtype(np.uint8)
                lbl_img = nib.Nifti1Image(self._labels, self._img.affine, hdr)
                nib.save(lbl_img, str(out))
                self._append_log(
                    f"  Saved → {Path(out).name}  (0=void, 1=matrix, 2=aggregate)",
                    'ok',
                )
                self._set_status("Labels saved.", busy=False)
            except Exception as ex:
                self._append_log(f"  {ex}", 'err')
                self._set_status("Export error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    def _export_stats_csv(self):
        if not self._E_stats:
            QMessageBox.warning(self, "No stats", "Compute E-Map first.")
            return
        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '')
        out = self._ask_save_path(
            f"{stem}_phase_stats.csv", ext=".csv",
            ftypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not out:
            return
        try:
            lines = [
                "phase,voxels,vol_frac,HU_mean,HU_std,"
                "E_mean_MPa,E_min_MPa,E_max_MPa,E_std_MPa",
            ]
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

    # Background removal

    def _do_remove_background(self):
        """Detect the specimen, strip the exterior background, save NIfTI."""
        if not self._require_img():
            return
        thresh = self._resolve_void_thresh()
        if thresh is None:
            QMessageBox.critical(self, "No threshold",
                                 "Could not determine a threshold.")
            return
        # raw copy + masks + labels ≈ dtype size + 8 B/voxel.
        itemsize = 2
        try:
            itemsize = np.dtype(self._img.get_data_dtype()).itemsize
        except Exception:
            pass
        if not self._ram_guard(float(itemsize * 2 + 8),
                               "Background removal (full resolution)"):
            return
        try:
            margin = max(0, int(float(self._bg_margin_widget.text())))
        except Exception:
            margin = 10
        crop = self._bg_crop_widget.isChecked()

        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '') \
            if self._path else "volume"
        out = self._ask_save_path(f"{stem}_nobg.nii.gz")
        if not out:
            return

        def _run():
            try:
                from ..core.background import remove_background

                self._log_sep("Background Removal")
                self._set_status("Detecting specimen...", busy=True)
                new_img, info = remove_background(
                    self._img, thresh, margin=margin, crop=crop,
                    progress=lambda st: self._set_status(
                        f"Background: {st}...", busy=True,
                    ),
                )
                self._set_status("Saving NIfTI...", busy=True)
                nib.save(new_img, str(out))

                self._append_log(
                    f"  Threshold          : {info['threshold']:.1f}", 'dim')
                self._append_log(
                    f"  Solid components   : {info['solid_components']} "
                    f"(largest kept)", 'dim')
                self._append_log(
                    f"  Background removed : {info['background_pct']:.1f}% "
                    f"of voxels  (fill = {info['fill_value']:.0f})", 'teal')
                if info['bbox'] is not None:
                    self._append_log(
                        f"  Cropped to         : {info['out_shape']}", 'teal')
                self._append_log(f"  Saved → {Path(out).name}", 'ok')
                self._set_status("Background removed.", busy=False)
                self.after(0, self._ask_load_after_save,
                           "Background removed and saved to:", Path(out))
            except Exception as ex:
                self._append_log(f"  Background removal error: {ex}", 'err')
                self._set_status("Background removal error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    # PDF report

    def _do_generate_report(self):
        """Bundle everything computed so far into a multi-page PDF.

        Only the loaded volume is mandatory — sections for phases, voids
        and the E-map are included when their data exists.
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
        model_name, model_params, void_thresh = "", {}, None
        try:
            model_name = self._model_var.get()
            model_params = self._get_model_params()
        except Exception:
            pass
        try:
            void_thresh = float(self._void_thresh_var.get())
        except Exception:
            pass

        def _run():
            try:
                from ..core.report import generate_pdf_report
                from ..utils import OperationCancelled

                self._begin_cancellable()
                self._log_sep("PDF Report")
                self._set_status("Generating PDF report...", busy=True)

                def _stage(st):
                    if self.cancel_requested():
                        raise OperationCancelled()
                    self._set_status(f"Report: {st}...", busy=True)
                # Figures never need full resolution — stream a strided
                # preview so huge volumes work on small machines too.
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
                if self._labels is None:
                    skipped.append("phases")
                if getattr(self, '_void_result', None) is None:
                    skipped.append("voids")
                if self._E_map is None:
                    skipped.append("E-map")
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
                    hu_cal=self._hu_cal or None,
                    hu_vol=self._hu_vol,
                    labels=self._labels,
                    phase_stats=self._E_stats or None,
                    porosity=self._porosity if self._E_stats else None,
                    void_result=getattr(self, '_void_result', None),
                    E_map=self._E_map,
                    model_name=model_name,
                    model_params=model_params,
                    void_thresh=void_thresh,
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

        threading.Thread(target=_run, daemon=True).start()
