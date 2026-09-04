"""Simulation-export action handlers.

Everything the Export tab triggers that produces a simulation input:
boundary-condition surfaces as STL, ``simulation_input.json`` and
``export_info.json``.

The NIfTI file remains the specimen geometry; only the selected boundary
regions are turned into STL. Surfaces are written through the full NIfTI
affine, so they land in the same world coordinates as the finite-cell
domain even when the affine flips or permutes axes.
``simulation_input.json`` carries ``specimen.bounding_box_mm`` derived from
the same affine, which keeps the solver domain and the exported surfaces
aligned.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QMessageBox, QTableWidgetItem,
)

from ..config import STATS_SUBSAMPLE_VOXELS
from ..core.bc_surface_export import (
    export_nifti_boundary_region_as_stl, nifti_physical_bounds,
)
from ..core.io import LazyGrayVolume
from ..deps import nib, np


#: Physical outward normal for each face direction, in world coordinates.
DIRECTION_NORMAL = {
    "+X": [1.0, 0.0, 0.0], "-X": [-1.0, 0.0, 0.0],
    "+Y": [0.0, 1.0, 0.0], "-Y": [0.0, -1.0, 0.0],
    "+Z": [0.0, 0.0, 1.0], "-Z": [0.0, 0.0, -1.0],
}
DIRECTION_AXIS = {"+X": 0, "-X": 0, "+Y": 1, "-Y": 1, "+Z": 2, "-Z": 2}

BC_PRESETS = ("Fixed support", "Roller support", "Load surface")


class SimExportActionsMixin:
    """Adds the simulation-export handlers to :class:`NiftiApp`."""

    # --- automatic material threshold ---------------------------------

    def _sample_values_as_float(self, values):
        """Reduce any sampled NIfTI values to a scalar float array.

        Handles scalar volumes, structured RGB dtypes and channel-last
        RGB/RGBA arrays, so the automatic threshold also works on
        colour-encoded scans.
        """
        arr = np.asarray(values)

        if arr.dtype.fields is not None:
            names = arr.dtype.names or ()
            lower = {n.lower(): n for n in names}
            if all(ch in lower for ch in ("r", "g", "b")):
                r = arr[lower["r"]].astype(np.float32)
                g = arr[lower["g"]].astype(np.float32)
                b = arr[lower["b"]].astype(np.float32)
                return 0.299 * r + 0.587 * g + 0.114 * b
            if names:
                return arr[names[0]].astype(np.float32)
            raise ValueError("Unsupported structured NIfTI dtype.")

        if arr.ndim >= 2 and arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

        return arr.astype(np.float32, copy=False)

    def _auto_bc_surface_threshold(self):
        """Estimate the material/background grey level with Otsu.

        Returns ``(lower, upper, info)``. The split is measured from the
        loaded volume, so it adapts to any scanner, dtype and intensity
        range.
        """
        if self._img is None:
            raise RuntimeError("No NIfTI image is loaded.")

        source = self._gray
        if source is None:
            source = LazyGrayVolume(self._img.dataobj)

        if isinstance(source, LazyGrayVolume):
            values = source.subsample_flat(STATS_SUBSAMPLE_VOXELS)
        else:
            flat = source.ravel()
            step = max(1, flat.size // STATS_SUBSAMPLE_VOXELS)
            values = flat[::step] if flat.size > STATS_SUBSAMPLE_VOXELS else flat

        values = np.asarray(self._sample_values_as_float(values),
                            dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise RuntimeError("No finite voxel values found for thresholding.")

        v_min, v_max = float(values.min()), float(values.max())
        p1 = float(np.percentile(values, 1))
        p99 = float(np.percentile(values, 99))

        if v_max <= v_min:
            return v_min, v_max, {"min": v_min, "max": v_max, "p1": p1,
                                  "p99": p99, "method": "constant-data"}

        counts, edges = np.histogram(values, bins=256, range=(v_min, v_max))
        centres = 0.5 * (edges[:-1] + edges[1:])
        total = float(counts.sum())
        lower = p1
        if total > 0:
            prob = counts.astype(np.float64) / total
            omega = np.cumsum(prob)
            mu = np.cumsum(prob * centres)
            mu_total = mu[-1]
            denom = omega * (1.0 - omega)
            denom[denom == 0.0] = np.nan
            with np.errstate(invalid="ignore"):
                between = ((mu_total * omega - mu) ** 2) / denom
            if not np.all(np.isnan(between)):
                lower = float(centres[int(np.nanargmax(between))])

        upper = v_max
        if upper <= lower:
            lower, upper = p1, v_max

        return float(lower), float(upper), {
            "min": v_min, "max": v_max, "p1": p1, "p99": p99,
            "method": "otsu-sampled",
        }

    def _nifti_intensity_range_for_threshold(self, threshold_info):
        """Intensity range used to normalise ``void_threshold`` into [0, 1].

        Integer volumes use the full dtype range, which is the convention
        the C++ pipeline expects; anything else uses the sampled range.
        """
        if self._img is not None:
            try:
                dtype = np.dtype(self._img.get_data_dtype())
                if np.issubdtype(dtype, np.integer):
                    info = np.iinfo(dtype)
                    return float(info.min), float(info.max), f"{dtype.name}_dtype_range"
            except Exception:
                pass
        lo = threshold_info.get("min")
        hi = threshold_info.get("max")
        if lo is None or hi is None:
            return None, None, "unknown"
        return float(lo), float(hi), "sampled_range_fallback"

    def _set_void_threshold_from_otsu_result(self, lower_threshold,
                                             threshold_info):
        """Write the normalised Otsu split into the simulation settings."""
        try:
            lo, hi, how = self._nifti_intensity_range_for_threshold(threshold_info)
            if lo is None or hi is None or hi <= lo:
                return
            value = (float(lower_threshold) - lo) / (hi - lo)
            value = max(0.0, min(1.0, value))

            self._auto_void_threshold_info = {
                "method": threshold_info.get("method"),
                "raw_threshold": float(lower_threshold),
                "void_threshold": float(value),
                "scale_min": float(lo), "scale_max": float(hi),
                "scale_method": how,
                "otsu_sample_min": threshold_info.get("min"),
                "otsu_sample_max": threshold_info.get("max"),
            }
            if hasattr(self, "_sim_void_threshold_spin"):
                self.after(0, lambda v=value:
                           self._sim_void_threshold_spin.setValue(v))
            self.after(0, lambda v=value, raw=float(lower_threshold), m=how:
                       self._append_log(
                           f"  Auto-filled void_threshold = {v:.4f} from the "
                           f"Otsu split {raw:.6g} using {m}", 'dim'))
        except Exception as ex:
            self.after(0, lambda err=str(ex): self._append_log(
                f"  Could not auto-fill void_threshold: {err}", 'warn'))

    def _autofill_void_threshold_from_otsu(self):
        """Pre-fill the simulation void threshold in the background."""
        if self._img is None:
            return

        def _run():
            try:
                lower, _upper, info = self._auto_bc_surface_threshold()
                self._set_void_threshold_from_otsu_result(lower, info)
            except Exception as ex:
                self.after(0, lambda err=str(ex): self._append_log(
                    f"  Could not auto-fill void_threshold: {err}", 'warn'))

        threading.Thread(target=_run, daemon=True).start()

    # --- boundary surface STL -----------------------------------------

    def _export_nifti_boundary_surface_stl(self):
        """Export one selected boundary surface as an ASCII STL."""
        if not self._require_img():
            return

        try:
            name = self._bc_surface_name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Missing output name",
                                    "Please enter an STL output file name.")
                return
            if not name.lower().endswith(".stl"):
                name += ".stl"

            face_direction = self._bc_surface_direction_combo.currentText().strip()
            x0 = float(self._bc_surface_x0.value())
            x1 = float(self._bc_surface_x1.value())
            y0 = float(self._bc_surface_y0.value())
            y1 = float(self._bc_surface_y1.value())
            z0 = float(self._bc_surface_z0.value())
            z1 = float(self._bc_surface_z1.value())

            auto = bool(self._bc_surface_auto_threshold_check.isChecked())
            if auto:
                lower_threshold, upper_threshold, threshold_info = \
                    self._auto_bc_surface_threshold()
                self._set_void_threshold_from_otsu_result(lower_threshold,
                                                          threshold_info)
            else:
                lower_threshold = float(self._bc_surface_lower_threshold.value())
                upper_threshold = float(self._bc_surface_upper_threshold.value())
                threshold_info = {"method": "manual", "min": None, "max": None,
                                  "p1": None, "p99": None}

            surface_band_mm = float(self._bc_surface_band_spin.value())
            normal_offset_mm = float(self._bc_surface_normal_offset_spin.value())
            keep_largest = bool(self._bc_surface_keep_largest_check.isChecked())
        except Exception as ex:
            QMessageBox.warning(
                self, "Input error",
                f"Could not read the boundary surface settings:\n{ex}")
            return

        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            QMessageBox.warning(self, "Invalid region",
                                "Each range end must be greater than its start.")
            return
        if upper_threshold <= lower_threshold:
            QMessageBox.warning(
                self, "Invalid threshold",
                "The upper threshold must be greater than the lower one.")
            return

        try:
            bounds_min, bounds_max = nifti_physical_bounds(self._img)
            lengths = bounds_max - bounds_min
            roi_min = tuple(
                float(bounds_min[i] + (p / 100.0) * lengths[i])
                for i, p in enumerate((x0, y0, z0))
            )
            roi_max = tuple(
                float(bounds_min[i] + (p / 100.0) * lengths[i])
                for i, p in enumerate((x1, y1, z1))
            )
        except Exception as ex:
            QMessageBox.warning(self, "Region error",
                                f"Could not compute physical bounds:\n{ex}")
            return

        default_path = (self._path.parent / name if self._path is not None
                        else Path(name))
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save boundary surface STL", str(default_path),
            "STL files (*.stl);;All files (*.*)",
        )
        if not out_path:
            return
        out_path = Path(out_path)

        def _run():
            try:
                self._log_sep("Export NIfTI Boundary Surface STL")
                self._set_status("Extracting boundary surface...", busy=True)

                self._append_log(f"  Output file    = {out_path.name}", 'dim')
                self._append_log(f"  Face direction = {face_direction}", 'dim')
                self._append_log(
                    f"  Region %       = X {x0:.1f}-{x1:.1f}   "
                    f"Y {y0:.1f}-{y1:.1f}   Z {z0:.1f}-{z1:.1f}", 'dim')
                self._append_log(
                    f"  Threshold      = {lower_threshold:.6g} to "
                    f"{upper_threshold:.6g}  ({threshold_info['method']})",
                    'teal')
                if threshold_info.get("min") is not None:
                    self._append_log(
                        f"  Data range     = {threshold_info['min']:.6g} to "
                        f"{threshold_info['max']:.6g}   "
                        f"p1/p99 = {threshold_info['p1']:.6g} / "
                        f"{threshold_info['p99']:.6g}", 'dim')
                self._append_log(
                    f"  Surface band   = {surface_band_mm:.3f} mm   "
                    f"normal offset = {normal_offset_mm:+.3f} mm   "
                    f"keep largest = {keep_largest}", 'dim')
                self._append_log(
                    f"  Region mm      = ({roi_min[0]:.3f}, {roi_min[1]:.3f}, "
                    f"{roi_min[2]:.3f}) → ({roi_max[0]:.3f}, "
                    f"{roi_max[1]:.3f}, {roi_max[2]:.3f})", 'dim')

                result = export_nifti_boundary_region_as_stl(
                    img=self._img,
                    output_path=out_path,
                    lower_threshold=lower_threshold,
                    upper_threshold=upper_threshold,
                    roi_bounds_mm=(roi_min, roi_max),
                    face_directions=[face_direction],
                    solid_name=out_path.stem,
                    pad_voxels=2,
                    max_faces=500_000,
                    surface_band_mm=surface_band_mm,
                    normal_offset_mm=normal_offset_mm,
                    keep_largest_component=keep_largest,
                    clean_material_mask=True,
                    mask_cleanup_iterations=0,
                )

                if not hasattr(self, "_bc_surface_exports"):
                    self._bc_surface_exports = []

                normal = DIRECTION_NORMAL.get(face_direction)
                translation = ([float(v * normal_offset_mm) for v in normal]
                               if normal is not None else None)
                preset = self._guess_bc_preset(out_path.stem)

                record = {
                    "id": len(self._bc_surface_exports) + 1,
                    "name": out_path.stem,
                    "stl_file": out_path.name,
                    "stl_file_absolute": str(out_path),
                    "type": "surface_marker",
                    "role": self._bc_surface_role(out_path.stem),
                    "method": "nifti_threshold_surface",
                    "face_direction": face_direction,
                    "face_axis": DIRECTION_AXIS.get(face_direction),
                    "normal": normal,
                    "roi_percent": {"x": [x0, x1], "y": [y0, y1], "z": [z0, z1]},
                    "roi_bounds_mm": {"min": list(roi_min), "max": list(roi_max)},
                    "threshold": {
                        "mode": threshold_info.get("method"),
                        "lower": float(lower_threshold),
                        "upper": float(upper_threshold),
                        "data_min": threshold_info.get("min"),
                        "data_max": threshold_info.get("max"),
                        "p1": threshold_info.get("p1"),
                        "p99": threshold_info.get("p99"),
                    },
                    "surface_extraction": {
                        "surface_band_mm": float(surface_band_mm),
                        "normal_offset_mm": float(normal_offset_mm),
                        "normal_offset_sign_convention":
                            "positive_outward_negative_inward",
                        "applied_translation_mm": translation,
                        "keep_largest_component": bool(keep_largest),
                        "pad_voxels": 2,
                        "max_faces": 500_000,
                    },
                    "export_result": {
                        "num_faces": int(result.num_faces),
                        "num_triangles": int(result.num_triangles),
                        "roi_index_min": list(result.roi_index_min),
                        "roi_index_max": list(result.roi_index_max),
                    },
                    "bc_preset": preset,
                    "simulation_definition":
                        self._simulation_definition_for_preset(preset),
                }
                self._bc_surface_exports.append(record)
                self.after(0, self._refresh_bc_assignment_table)

                self._append_log(f"  Saved → {out_path.name}", 'ok')
                self._append_log(
                    f"  Faces = {result.num_faces}   triangles = "
                    f"{result.num_triangles}", 'teal')
                self._append_log(
                    f"  Added to the boundary-condition list as "
                    f"'{record['name']}'  (preset: {preset})", 'ok')
                self._set_status("Boundary surface STL exported.", busy=False)
            except Exception as ex:
                self._append_log(f"  Boundary surface export error: {ex}", 'err')
                self._set_status("Boundary surface export error.", busy=False)

        threading.Thread(target=_run, daemon=True).start()

    # --- boundary-condition bookkeeping -------------------------------

    @staticmethod
    def _bc_surface_role(surface_name: str) -> str:
        name = surface_name.lower()
        if "load" in name:
            return "load_surface"
        if "support" in name or "roller" in name:
            return "support_surface"
        return "boundary_surface"

    def _guess_bc_preset(self, surface_name: str) -> str:
        """Pick a sensible default preset from the surface name."""
        name = (surface_name or "").lower()
        if any(k in name for k in ("load", "pull", "push")):
            return "Load surface"
        if "roller" in name:
            return "Roller support"
        return "Fixed support"

    def _parse_sim_load_direction(self):
        """Return ``(axis_index, sign)`` for the chosen load direction."""
        direction = getattr(self, "_sim_load_direction", "+Y")
        combo = getattr(self, "_sim_load_direction_combo", None)
        if combo is not None:
            direction = combo.currentText().strip()
            self._sim_load_direction = direction
        direction = direction.strip().upper()
        if (len(direction) != 2 or direction[0] not in "+-"
                or direction[1] not in "XYZ"):
            raise ValueError(f"Invalid load direction: {direction}")
        return {"X": 0, "Y": 1, "Z": 2}[direction[1]], (
            1.0 if direction[0] == "+" else -1.0)

    def _simulation_definition_for_preset(self, preset: str) -> dict:
        """Compact boundary-condition definition for the C++ solver."""
        preset = (preset or "Fixed support").strip()
        if preset == "Fixed support":
            return {"type": "Dirichlet", "constrained_axes": [0, 1, 2],
                    "displacement_values": [0.0, 0.0, 0.0],
                    "is_load_surface": False, "measure_reaction_force": False}
        if preset == "Roller support":
            return {"type": "Dirichlet", "constrained_axes": [1, 2],
                    "displacement_values": [0.0, 0.0],
                    "is_load_surface": False, "measure_reaction_force": False}
        if preset == "Load surface":
            axis, sign = self._parse_sim_load_direction()
            return {"type": "Dirichlet", "constrained_axes": [int(axis)],
                    "displacement_values": [float(sign)],
                    "is_load_surface": True, "measure_reaction_force": True}
        raise ValueError(f"Unsupported boundary-condition preset: {preset}")

    def _refresh_bc_assignment_table(self):
        """Redraw the preset table from ``self._bc_surface_exports``."""
        table = getattr(self, "_bc_assignment_table", None)
        if table is None:
            return
        surfaces = getattr(self, "_bc_surface_exports", [])
        table.setRowCount(len(surfaces))

        for row, surface in enumerate(surfaces):
            surface["id"] = row + 1
            stl_name = Path(
                surface.get("stl_file", surface.get("name", f"bc{row + 1}"))
            ).name
            preset = surface.get("bc_preset") or self._guess_bc_preset(
                surface.get("name", stl_name))
            surface["bc_preset"] = preset

            id_item = QTableWidgetItem(str(row + 1))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, id_item)

            stl_item = QTableWidgetItem(stl_name)
            stl_item.setFlags(stl_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 1, stl_item)

            combo = QComboBox(table)
            combo.addItems(list(BC_PRESETS))
            combo.setCurrentText(preset)

            def _on_changed(text, row_index=row):
                surfaces_now = getattr(self, "_bc_surface_exports", [])
                if row_index < len(surfaces_now):
                    surfaces_now[row_index]["bc_preset"] = text
                    surfaces_now[row_index]["simulation_definition"] = \
                        self._simulation_definition_for_preset(text)

            combo.currentTextChanged.connect(_on_changed)
            table.setCellWidget(row, 2, combo)

        table.resizeRowsToContents()

    def _sync_bc_presets_from_table(self):
        """Copy the visible dropdown values back into the export records."""
        table = getattr(self, "_bc_assignment_table", None)
        if table is None:
            return
        for row, surface in enumerate(getattr(self, "_bc_surface_exports", [])):
            combo = table.cellWidget(row, 2)
            if combo is None:
                continue
            preset = combo.currentText().strip()
            surface["bc_preset"] = preset
            surface["simulation_definition"] = \
                self._simulation_definition_for_preset(preset)

    def _remove_selected_bc_surface(self):
        """Drop the selected row from the boundary-condition list."""
        table = getattr(self, "_bc_assignment_table", None)
        surfaces = getattr(self, "_bc_surface_exports", [])
        if table is None or not surfaces:
            return
        row = table.currentRow()
        if row < 0 or row >= len(surfaces):
            QMessageBox.information(
                self, "No selection",
                "Select a row in the table first. This removes the entry "
                "from the list only, the STL file on disk is kept.")
            return
        removed = surfaces.pop(row)
        self._refresh_bc_assignment_table()
        self._append_log(
            f"  Removed '{removed.get('name')}' from the boundary-condition "
            f"list (the STL file itself was not deleted).", 'warn')

    def _clear_bc_surfaces(self):
        """Empty the boundary-condition list."""
        if not getattr(self, "_bc_surface_exports", []):
            return
        reply = QMessageBox.question(
            self, "Clear boundary conditions",
            "Remove every boundary surface from the list?\n\n"
            "The exported STL files on disk are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._bc_surface_exports = []
        self._refresh_bc_assignment_table()
        self._append_log("  Boundary-condition list cleared.", 'warn')

    # --- geometry metadata --------------------------------------------

    def _nifti_geometry_for_simulation_json(self) -> dict:
        """Affine-aware geometry block for the simulation input.

        The C++ pipeline overrides its default positive-spacing domain when
        ``specimen.bounding_box_mm`` is present. The affine-derived bounds
        written here keep the exported STL surfaces on the finite-cell
        domain for flipped-axis (LPS, LAS) volumes.
        """
        if self._img is None:
            raise RuntimeError("No NIfTI image is loaded.")
        shape = [int(v) for v in self._img.shape[:3]]
        spacing = [float(v) for v in self._img.header.get_zooms()[:3]]
        affine = np.asarray(self._img.affine, dtype=float)
        orientation = "".join(nib.aff2axcodes(affine))
        bbox_min, bbox_max = nifti_physical_bounds(self._img)
        return {
            "coordinate_system": orientation,
            "units": "mm",
            "shape_voxels": shape,
            "voxel_spacing_mm": spacing,
            "physical_size_mm": [float(v) for v in (bbox_max - bbox_min)],
            "bounding_box_mm": {
                "min": [float(v) for v in bbox_min],
                "max": [float(v) for v in bbox_max],
            },
            "affine": [[float(v) for v in row] for row in affine.tolist()],
        }

    def _nifti_stem(self, filename: str | None = None) -> str:
        """File stem, handling ``.nii.gz`` correctly."""
        name = filename or (self._path.name if self._path is not None
                            else "specimen.nii")
        low = name.lower()
        if low.endswith(".nii.gz"):
            return name[:-7]
        if low.endswith(".nii"):
            return name[:-4]
        return Path(name).stem

    def _update_simulation_input_defaults(self):
        """Refresh the simulation-input fields after a file is loaded."""
        if self._path is None:
            return
        if hasattr(self, "_sim_nifti_filename_edit"):
            self._sim_nifti_filename_edit.setText(self._path.name)
        if hasattr(self, "_sim_output_stem_edit"):
            self._sim_output_stem_edit.setText(self._nifti_stem(self._path.name))
        self._autofill_void_threshold_from_otsu()

    # --- JSON exports --------------------------------------------------

    def _export_simulation_input_json(self):
        """Write the compact ``simulation_input.json`` for the C++ solver."""
        if not self._require_img():
            return
        self._sync_bc_presets_from_table()

        surfaces = getattr(self, "_bc_surface_exports", [])
        if not surfaces:
            QMessageBox.warning(
                self, "No boundary surfaces",
                "Export at least one boundary surface STL before writing "
                "simulation_input.json.")
            return

        try:
            nifti_file = self._sim_nifti_filename_edit.text().strip()
            output_stem = self._sim_output_stem_edit.text().strip()
            E_ref = float(self._sim_E_ref_spin.value())
            nu = float(self._sim_nu_spin.value())
            void_threshold = float(self._sim_void_threshold_spin.value())
            penalty_factor = float(self._sim_penalty_factor_edit.text().strip())
            body_force = float(self._sim_body_force_edit.text().strip())
            n_steps = int(self._sim_n_steps_spin.value())
            max_disp = float(self._sim_max_disp_spin.value())
        except Exception as ex:
            QMessageBox.warning(self, "Invalid setting",
                                f"Could not read the simulation settings:\n{ex}")
            return

        problems = []
        if not nifti_file:
            problems.append("The NIfTI file name is empty.")
        elif Path(nifti_file).name != nifti_file:
            problems.append("Give the NIfTI file name only, not a full path.")
        if not output_stem:
            problems.append("The output stem is empty.")
        if not 0.0 <= void_threshold <= 1.0:
            problems.append("void_threshold must lie between 0 and 1.")
        if body_force < 0.0:
            problems.append("The body force must not be negative.")
        if problems:
            QMessageBox.warning(self, "Invalid simulation input",
                                "\n".join(problems))
            return

        boundary_conditions = []
        load_surfaces = 0
        for i, surface in enumerate(surfaces, start=1):
            stl_file = Path(surface.get("stl_file", "")).name
            if not stl_file:
                QMessageBox.warning(
                    self, "Missing STL file name",
                    f"Boundary surface {i} has no STL file name.")
                return
            preset = surface.get("bc_preset") or self._guess_bc_preset(
                surface.get("name", stl_file))
            definition = self._simulation_definition_for_preset(preset)
            if definition.get("is_load_surface"):
                load_surfaces += 1
            boundary_conditions.append({
                "id": int(i),
                "name": surface.get("name") or Path(stl_file).stem,
                "stl_file": stl_file,
                "simulation_definition": definition,
            })

        if load_surfaces == 0:
            reply = QMessageBox.question(
                self, "No load surface",
                "No boundary surface is assigned as a load surface, so the "
                "simulation would have nothing driving it.\n\nExport anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        geometry = self._nifti_geometry_for_simulation_json()
        simulation_input = {
            "specimen": {
                "nifti_file": nifti_file,
                "bounding_box_mm": geometry["bounding_box_mm"],
                "coordinate_system": geometry["coordinate_system"],
                "units": geometry["units"],
            },
            "material": {
                "E_ref": E_ref, "nu": nu, "units": "kN/mm^2",
                "void_threshold": void_threshold,
            },
            "body_force": {
                "direction": [0, -1, 0],
                "magnitude_kN_per_mm3": body_force,
            },
            "boundary_conditions": boundary_conditions,
            "solver": {"penalty_factor": penalty_factor},
            "load_stepping": {"n_steps": n_steps,
                              "max_displacement_mm": max_disp},
            "output": {"stem": output_stem},
        }

        default_path = (self._path.parent / "simulation_input.json"
                        if self._path is not None
                        else Path("simulation_input.json"))
        out = self._ask_save_path(
            str(default_path), ext=".json",
            ftypes=[("JSON", "*.json"), ("All", "*.*")])
        if not out:
            return

        try:
            self._log_sep("Export simulation_input.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(simulation_input, f, indent=4)
            bbox = simulation_input["specimen"]["bounding_box_mm"]
            self._append_log(f"  Saved → {Path(out).name}", 'ok')
            self._append_log(f"  nifti_file = {nifti_file}", 'dim')
            self._append_log(
                f"  bounding_box_mm = {bbox['min']} → {bbox['max']}  "
                f"({geometry['coordinate_system']})", 'teal')
            self._append_log(
                "  The bounding box keeps the finite-cell domain in the same "
                "world coordinates as the exported STL surfaces.", 'dim')
            self._append_log(
                f"  Boundary conditions = {len(boundary_conditions)}  "
                f"(load surfaces: {load_surfaces})", 'dim')
            self._append_log(
                f"  Material = E_ref {E_ref:g} kN/mm^2, nu {nu:g}, "
                f"void_threshold {void_threshold:.4f}", 'teal')
            self._set_status("simulation_input.json saved.", busy=False)
        except Exception as ex:
            self._append_log(f"  simulation_input.json error: {ex}", 'err')
            self._set_status("simulation_input.json error.", busy=False)

    def _export_info_json(self):
        """Write the preprocessing record used for traceability.

        This is not the solver input. It records what the application did to
        the volume, so a result can be traced back to the thresholds,
        regions, corrections and rotations that produced it.
        """
        if not self._require_img():
            return

        def json_safe(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(v) for v in value]
            if isinstance(value, np.ndarray):
                return json_safe(value.tolist())
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    return str(value)
            return value

        def compact(surface: dict, idx: int) -> dict:
            return {
                "id": int(idx),
                "name": surface.get("name"),
                "stl_file": Path(surface.get("stl_file", "")).name,
                "stl_file_absolute": surface.get("stl_file_absolute"),
                "method": surface.get("method"),
                "face_direction": surface.get("face_direction"),
                "face_axis": surface.get("face_axis"),
                "normal": surface.get("normal"),
                "roi_percent": surface.get("roi_percent"),
                "roi_bounds_mm": surface.get("roi_bounds_mm"),
                "threshold": surface.get("threshold"),
                "surface_extraction": surface.get("surface_extraction"),
                "export_result": surface.get("export_result"),
                "bc_assignment": {
                    "preset": surface.get("bc_preset"),
                    "simulation_definition": surface.get("simulation_definition"),
                },
            }

        out = self._ask_save_path(
            "export_info.json", ext=".json",
            ftypes=[("JSON", "*.json"), ("All", "*.*")])
        if not out:
            return

        geometry = self._nifti_geometry_for_simulation_json()
        source_nifti = {
            "filename": self._path.name if self._path else None,
            "path": str(self._path) if self._path else None,
            "coordinate_system": geometry["coordinate_system"],
            "units": "mm",
            "shape_voxels": geometry["shape_voxels"],
            "voxel_spacing_mm": geometry["voxel_spacing_mm"],
            "physical_size_mm": geometry["physical_size_mm"],
            "bounding_box_mm": geometry["bounding_box_mm"],
            "affine": geometry["affine"],
            "data_dtype": str(self._img.get_data_dtype()),
        }

        info = {
            "schema_version": "2.1",
            "export_date": date.today().isoformat(),
            "purpose": "preprocessing_record",
            "note": ("For traceability and debugging. The C++ simulation "
                     "reads simulation_input.json, not this file."),
            "source_nifti": source_nifti,
            "boundary_surface_exports": [
                compact(s, i) for i, s in
                enumerate(getattr(self, "_bc_surface_exports", []), start=1)
            ],
            "preprocessing": {
                "rotation_history": getattr(self, "_rotation_history", []),
                "beam_hardening_correction":
                    getattr(self, "_bh_fit_info", None),
                "pore_threshold": getattr(self, "_pore_threshold_info", None),
                "auto_void_threshold":
                    getattr(self, "_auto_void_threshold_info", None),
                "bc_stls_same_coords_as_nifti": True,
                "simulation_input_includes_bounding_box_mm": True,
            },
        }

        try:
            self._log_sep("Export export_info.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(json_safe(info), f, indent=4)
            self._append_log(f"  Saved → {Path(out).name}", 'ok')
            self._append_log(
                f"  source_nifti      = {source_nifti['filename']}", 'dim')
            self._append_log(
                f"  boundary surfaces = "
                f"{len(info['boundary_surface_exports'])}", 'dim')
            self._set_status("export_info.json saved.", busy=False)
        except Exception as ex:
            self._append_log(f"  export_info.json error: {ex}", 'err')
            self._set_status("export_info.json error.", busy=False)
