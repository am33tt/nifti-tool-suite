"""High-level user-triggered actions.

This mixin contains every ``_do_*`` and ``_export_*`` handler - the
glue between the controls panel and the pure :mod:`niftitool.core`
functions.  Actions that do heavy lifting run in a daemon thread and
marshal results back onto the Qt main thread via :meth:`NiftiApp.after`
(which is a tk-compat shim around :class:`QTimer` / a signal).
"""

from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..config import STATS_SUBSAMPLE_VOXELS
from ..core.calibration import calibrate_to_hu
from ..core.cpp_export import generate_ct_hpp_snippet
from ..core.geometry import run_angle_rotation, run_cropper, run_reorientation
from ..core.histogram import compute_histogram
from ..core.io import LazyGrayVolume, load_nifti, raw_array, run_gunzip, to_gray
from ..core.mapping import compute_E_map
from ..core.metadata import get_axis_labels, read_metadata
from ..core.segmentation import phase_statistics, segment_phases
from ..deps import HAS_NIBABEL, nib, np
from ..utils import available_ram_mb


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
        self._cal_result_var.set("Not calibrated")
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

        self._status_var.set(f"Opening {Path(path).name}...")
        self._prog.begin_staged()
        threading.Thread(target=self._staged_load, args=(path,), daemon=True).start()

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
                gray_full = self._get_gray() if self._gray is not None else None
                txt = read_metadata(self._img, gray_full, self._hu_vol, self._labels)
                self._append_log(txt, 'dim')
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

    def _do_calibrate(self):
        if not self._require_img():
            return
        try:
            air_int = float(self._cal_air_int.get())
            ref_int = float(self._cal_ref_int.get())
            air_hu = float(self._cal_air_hu.get())
            ref_hu = float(self._cal_ref_hu.get())
        except ValueError:
            QMessageBox.critical(self, "Bad input",
                                 "All calibration fields must be numbers.")
            return

        def _run():
            try:
                self._log_sep("HU Calibration")
                self._set_status("Calibrating to HU...", busy=True)
                gray = self._get_gray()
                hu_vol, m, c = calibrate_to_hu(gray, air_int, ref_int, air_hu, ref_hu)
                self._hu_vol = hu_vol
                self._hu_cal = {
                    'm': m, 'c': c,
                    'air_int': air_int, 'ref_int': ref_int,
                }
                result_text = f"HU = {m:.4f}·raw + {c:.2f}"
                self.after(0, lambda: self._cal_result_var.set(result_text))
                self._append_log(f"  Calibration: {result_text}", 'teal')
                self._append_log(
                    f"  HU range: {float(hu_vol.min()):.1f}  to  {float(hu_vol.max()):.1f}",
                    'dim',
                )
                self._set_status("HU calibration complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  Calibration error: {ex}", 'err')
                self._set_status("Calibration error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    def _do_compute_emap(self):
        if not self._require_img():
            return
        if self._hu_vol is None:
            reply = QMessageBox.question(
                self, "No HU calibration",
                "HU calibration has not been applied.\n"
                "The raw intensity will be used directly as HU.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            model = self._model_var.get()
            params = self._get_model_params()
            void_thresh = float(self._void_thresh_var.get())
            agg_str = self._agg_thresh_var.get().strip()
            agg_thresh = float(agg_str) if agg_str else None
        except Exception as ex:
            QMessageBox.critical(self, "Bad parameters", str(ex))
            return

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
                self._set_status("E-Map complete.", busy=False)
            except Exception as ex:
                self._append_log(f"  E-Map error: {ex}", 'err')
                self._set_status("E-Map error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    # Exports

    def _export_emap_nifti(self):
        if not self._require_emap():
            return
        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '')
        out = self._ask_save_path(f"{stem}_Emap.nii.gz")
        if not out:
            return

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
                self._append_log(f"  {ex}", 'err')
                self._set_status("Export error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    def _export_emap_raw(self):
        if not self._require_emap():
            return
        stem = self._path.name.replace('.nii.gz', '').replace('.nii', '')
        out = self._ask_save_path(
            f"{stem}_Emap.raw",
            ext=".raw",
            ftypes=[("Raw binary", "*.raw"), ("All", "*.*")],
        )
        if not out:
            return

        def _run():
            try:
                self._log_sep("Export E-Map raw float32")
                self._set_status("Saving raw binary...", busy=True)
                E_c = np.ascontiguousarray(self._E_map, dtype=np.float32)
                with open(out, 'wb') as f:
                    f.write(E_c.tobytes())
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
                self._append_log(f"  {ex}", 'err')
                self._set_status("Export error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

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

    def _export_ct_hpp(self):
        if not self._require_emap():
            return
        try:
            params = self._get_model_params()
            model = self._model_var.get()
            void_thresh = float(self._void_thresh_var.get())
        except Exception as ex:
            QMessageBox.critical(self, "Parameter error", str(ex))
            return

        snippet = generate_ct_hpp_snippet(
            self._get_gray(),
            self._hu_cal if self._hu_cal else {'m': 1.0, 'c': 0.0},
            self._E_map, self._img, model, params, void_thresh,
            self._hu_cal.get('air_int', 0),
            self._hu_cal.get('ref_int', 199),
        )
        self._cpp_text.config(state='normal')
        self._cpp_text.delete('1.0', 'end')
        self._cpp_text.insert('end', snippet)
        self._cpp_text.config(state='disabled')
        # Switch to the Sim Export tab.
        try:
            idx = self._nb.indexOf(self._export_tab)
            if idx >= 0:
                self._nb.setCurrentIndex(idx)
        except Exception:
            pass
        self._append_log("  ct.hpp snippet generated - see Sim Export tab.", 'teal')

    def _copy_cpp(self):
        txt = self._cpp_text.get('1.0', 'end')
        self.clipboard_clear()
        self.clipboard_append(txt)
        self._append_log("  Copied ct.hpp snippet to clipboard.", 'ok')

    def _save_cpp(self):
        out = self._ask_save_path(
            "ct_snippet.hpp", ext=".hpp",
            ftypes=[("C++ header", "*.hpp *.h"), ("All", "*.*")],
        )
        if not out:
            return
        txt = self._cpp_text.get('1.0', 'end')
        with open(out, 'w') as f:
            f.write(txt)
        self._append_log(f"  Saved → {Path(out).name}", 'ok')
