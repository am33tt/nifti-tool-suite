"""Export & Simulation tab.

Produces the multi-page PDF report, the boundary-condition surfaces
extracted from the NIfTI mask and written as STL, the
``simulation_input.json`` that tells the C++ FCM pipeline what each surface
means, and ``export_info.json`` as a preprocessing record.

Per-void measurements are exported from the Porosity tab and the indicator
field from the Binarise tab, next to the analyses that produce them.

The NIfTI file remains the specimen geometry. Only the selected boundary
regions become STL, so no geometry is remeshed or approximated.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QScrollArea, QSpinBox, QTableWidget, QVBoxLayout,
    QWidget,
)

from ..config import BG, BORDER, PANEL, TEXT, TEXT_DIM
from .widgets import hline, styled_btn


def _section_title(text, parent):
    lbl = QLabel(text, parent)
    lbl.setFont(QFont("Segoe UI Semibold", 10))
    lbl.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
    return lbl


def _hint(text, parent, bg=BG):
    lbl = QLabel(text, parent)
    lbl.setFont(QFont("Segoe UI", 9))
    lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {bg};")
    lbl.setWordWrap(True)
    return lbl


def _panel(parent):
    frame = QFrame(parent)
    frame.setStyleSheet(
        f"QFrame {{ background-color: {PANEL}; border: 1px solid {BORDER}; }}"
    )
    return frame


def _row(parent, spacing=8):
    w = QWidget(parent)
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    return w, lay


class ExportTabMixin:
    """Adds the Export & Simulation tab to :class:`NiftiApp`."""

    def _build_export_tab(self, parent):
        root_lay = QVBoxLayout(parent)
        root_lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root_lay.addWidget(scroll)

        outer = QWidget()
        scroll.setWidget(outer)
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(6)

        title = QLabel("Export & Simulation", outer)
        title.setFont(QFont("Segoe UI Semibold", 11))
        title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        lay.addWidget(title)
        lay.addWidget(_hint(
            "Save what you have computed and build the input files for the "
            "finite-cell simulation. The PDF report bundles everything; the "
            "CSVs hold the raw numbers; the STL surfaces and "
            "simulation_input.json drive the solver.", outer,
        ))
        lay.addWidget(hline(outer))

        # --- reports and data exports ---------------------------------
        lay.addWidget(_section_title("Report", outer))
        self._add_button_rows(outer, lay, [
            ("Generate PDF Report",
             "Multi-page report: slices, histogram, pore-threshold "
             "sensitivity, void analytics, beam-hardening correction, "
             "binarisation",
             self._do_generate_report, True),
        ])

        lay.addWidget(hline(outer))

        # --- simulation_input.json settings ---------------------------
        lay.addWidget(_section_title("Simulation input settings", outer))

        meta = _panel(outer)
        meta_lay = QVBoxLayout(meta)
        meta_lay.setContentsMargins(12, 8, 12, 8)
        meta_lay.setSpacing(8)

        r1, r1l = _row(meta)
        r1l.addWidget(QLabel("Simulation NIfTI file:", r1))
        self._sim_nifti_filename_edit = QLineEdit("filename.nii", r1)
        self._sim_nifti_filename_edit.setMinimumWidth(240)
        r1l.addWidget(self._sim_nifti_filename_edit)

        r1l.addWidget(QLabel("Load direction:", r1))
        self._sim_load_direction_combo = QComboBox(r1)
        self._sim_load_direction_combo.addItems(
            ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        )
        self._sim_load_direction_combo.setCurrentText(
            getattr(self, "_sim_load_direction", "+Y")
        )
        self._sim_load_direction_combo.currentTextChanged.connect(
            lambda text: setattr(self, "_sim_load_direction", text)
        )
        r1l.addWidget(self._sim_load_direction_combo)

        r1l.addWidget(QLabel("Output stem:", r1))
        self._sim_output_stem_edit = QLineEdit("specimen01", r1)
        self._sim_output_stem_edit.setMinimumWidth(160)
        r1l.addWidget(self._sim_output_stem_edit)
        r1l.addStretch(1)
        meta_lay.addWidget(r1)

        r2, r2l = _row(meta)
        r2l.addWidget(QLabel("E_ref:", r2))
        self._sim_E_ref_spin = QDoubleSpinBox(r2)
        self._sim_E_ref_spin.setRange(0.0, 1_000_000.0)
        self._sim_E_ref_spin.setDecimals(6)
        self._sim_E_ref_spin.setValue(30.0)
        self._sim_E_ref_spin.setMinimumWidth(120)
        self._sim_E_ref_spin.setKeyboardTracking(False)
        r2l.addWidget(self._sim_E_ref_spin)
        r2l.addWidget(QLabel("kN/mm^2", r2))

        r2l.addWidget(QLabel("nu:", r2))
        self._sim_nu_spin = QDoubleSpinBox(r2)
        self._sim_nu_spin.setRange(0.0, 0.499)
        self._sim_nu_spin.setDecimals(4)
        self._sim_nu_spin.setValue(0.2)
        self._sim_nu_spin.setMinimumWidth(90)
        self._sim_nu_spin.setKeyboardTracking(False)
        r2l.addWidget(self._sim_nu_spin)

        r2l.addWidget(QLabel("void threshold:", r2))
        self._sim_void_threshold_spin = QDoubleSpinBox(r2)
        self._sim_void_threshold_spin.setRange(0.0, 1.0)
        self._sim_void_threshold_spin.setDecimals(4)
        self._sim_void_threshold_spin.setValue(0.47)
        self._sim_void_threshold_spin.setMinimumWidth(100)
        self._sim_void_threshold_spin.setKeyboardTracking(False)
        r2l.addWidget(self._sim_void_threshold_spin)
        r2l.addStretch(1)
        meta_lay.addWidget(r2)

        r3, r3l = _row(meta)
        r3l.addWidget(QLabel("penalty factor:", r3))
        self._sim_penalty_factor_edit = QLineEdit("100000.0", r3)
        self._sim_penalty_factor_edit.setMaximumWidth(110)
        r3l.addWidget(self._sim_penalty_factor_edit)

        r3l.addWidget(QLabel("n steps:", r3))
        self._sim_n_steps_spin = QSpinBox(r3)
        self._sim_n_steps_spin.setRange(1, 100000)
        self._sim_n_steps_spin.setValue(10)
        self._sim_n_steps_spin.setMinimumWidth(90)
        r3l.addWidget(self._sim_n_steps_spin)

        r3l.addWidget(QLabel("max displacement:", r3))
        self._sim_max_disp_spin = QDoubleSpinBox(r3)
        self._sim_max_disp_spin.setRange(0.0, 1_000_000.0)
        self._sim_max_disp_spin.setDecimals(6)
        self._sim_max_disp_spin.setValue(0.5)
        self._sim_max_disp_spin.setSuffix(" mm")
        self._sim_max_disp_spin.setMinimumWidth(140)
        self._sim_max_disp_spin.setKeyboardTracking(False)
        r3l.addWidget(self._sim_max_disp_spin)
        r3l.addStretch(1)
        meta_lay.addWidget(r3)

        r4, r4l = _row(meta)
        r4l.addWidget(QLabel("body force:", r4))
        self._sim_body_force_edit = QLineEdit("0", r4)
        self._sim_body_force_edit.setMaximumWidth(120)
        r4l.addWidget(self._sim_body_force_edit)
        r4l.addWidget(QLabel("kN/mm^3  (direction 0, -1, 0)", r4))
        r4l.addStretch(1)
        meta_lay.addWidget(r4)

        lay.addWidget(meta)
        lay.addWidget(hline(outer))

        # --- NIfTI boundary surface STL export ------------------------
        lay.addWidget(_section_title("NIfTI boundary surface export", outer))

        surf = _panel(outer)
        surf.setMinimumHeight(210)
        s_lay = QVBoxLayout(surf)
        s_lay.setContentsMargins(12, 8, 12, 8)
        s_lay.setSpacing(8)

        s_lay.addWidget(_hint(
            "Extracts one boundary-condition surface directly from the NIfTI "
            "material mask and writes it as an STL in the same physical "
            "coordinates as the NIfTI affine, so the surface lands exactly on "
            "the finite-cell domain even for flipped or permuted axes. The "
            "region of interest is given as a percentage of the physical "
            "bounding box, so it is independent of voxel count and specimen "
            "shape.", surf, bg=PANEL,
        ))

        sr1, sr1l = _row(surf)
        sr1l.addWidget(QLabel("Output STL name:", sr1))
        self._bc_surface_name_edit = QLineEdit(
            "selected_boundary_surface.stl", sr1
        )
        self._bc_surface_name_edit.setMinimumWidth(240)
        sr1l.addWidget(self._bc_surface_name_edit)

        sr1l.addWidget(QLabel("Face direction:", sr1))
        self._bc_surface_direction_combo = QComboBox(sr1)
        self._bc_surface_direction_combo.addItems(
            ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        )
        self._bc_surface_direction_combo.setCurrentText("+Y")
        sr1l.addWidget(self._bc_surface_direction_combo)
        sr1l.addStretch(1)
        s_lay.addWidget(sr1)

        sr2, sr2l = _row(surf)
        sr2l.addWidget(QLabel("X range %:", sr2))
        self._bc_surface_x0 = self._pct_spin(sr2, 20.0)
        sr2l.addWidget(self._bc_surface_x0)
        sr2l.addWidget(QLabel("to", sr2))
        self._bc_surface_x1 = self._pct_spin(sr2, 80.0)
        sr2l.addWidget(self._bc_surface_x1)

        sr2l.addWidget(QLabel("Y range %:", sr2))
        self._bc_surface_y0 = self._pct_spin(sr2, 95.0)
        sr2l.addWidget(self._bc_surface_y0)
        sr2l.addWidget(QLabel("to", sr2))
        self._bc_surface_y1 = self._pct_spin(sr2, 100.0)
        sr2l.addWidget(self._bc_surface_y1)
        sr2l.addStretch(1)
        s_lay.addWidget(sr2)

        sr3, sr3l = _row(surf)
        sr3l.addWidget(QLabel("Z range %:", sr3))
        self._bc_surface_z0 = self._pct_spin(sr3, 45.0)
        sr3l.addWidget(self._bc_surface_z0)
        sr3l.addWidget(QLabel("to", sr3))
        self._bc_surface_z1 = self._pct_spin(sr3, 55.0)
        sr3l.addWidget(self._bc_surface_z1)

        sr3l.addWidget(QLabel("Threshold:", sr3))
        self._bc_surface_lower_threshold = QDoubleSpinBox(sr3)
        self._bc_surface_lower_threshold.setRange(-1e7, 1e7)
        self._bc_surface_lower_threshold.setDecimals(1)
        self._bc_surface_lower_threshold.setValue(80.0)
        self._bc_surface_lower_threshold.setKeyboardTracking(False)
        sr3l.addWidget(self._bc_surface_lower_threshold)
        sr3l.addWidget(QLabel("to", sr3))
        self._bc_surface_upper_threshold = QDoubleSpinBox(sr3)
        self._bc_surface_upper_threshold.setRange(-1e7, 1e7)
        self._bc_surface_upper_threshold.setDecimals(1)
        self._bc_surface_upper_threshold.setValue(220.0)
        self._bc_surface_upper_threshold.setKeyboardTracking(False)
        sr3l.addWidget(self._bc_surface_upper_threshold)

        self._bc_surface_auto_threshold_check = QCheckBox("Auto threshold", sr3)
        self._bc_surface_auto_threshold_check.setChecked(True)
        sr3l.addWidget(self._bc_surface_auto_threshold_check)

        def _toggle_manual_threshold(auto_enabled: bool):
            self._bc_surface_lower_threshold.setEnabled(not auto_enabled)
            self._bc_surface_upper_threshold.setEnabled(not auto_enabled)

        self._bc_surface_auto_threshold_check.toggled.connect(
            _toggle_manual_threshold
        )
        _toggle_manual_threshold(True)
        sr3l.addStretch(1)
        s_lay.addWidget(sr3)

        sr4, sr4l = _row(surf)
        sr4l.addWidget(QLabel("Surface band:", sr4))
        self._bc_surface_band_spin = QDoubleSpinBox(sr4)
        self._bc_surface_band_spin.setRange(0.0, 100.0)
        self._bc_surface_band_spin.setDecimals(3)
        self._bc_surface_band_spin.setSingleStep(0.1)
        self._bc_surface_band_spin.setValue(0.5)
        self._bc_surface_band_spin.setSuffix(" mm")
        self._bc_surface_band_spin.setKeyboardTracking(False)
        sr4l.addWidget(self._bc_surface_band_spin)

        sr4l.addWidget(QLabel("Normal offset:", sr4))
        self._bc_surface_normal_offset_spin = QDoubleSpinBox(sr4)
        self._bc_surface_normal_offset_spin.setRange(-100.0, 100.0)
        self._bc_surface_normal_offset_spin.setDecimals(3)
        self._bc_surface_normal_offset_spin.setSingleStep(0.05)
        self._bc_surface_normal_offset_spin.setValue(0.0)
        self._bc_surface_normal_offset_spin.setSuffix(" mm")
        self._bc_surface_normal_offset_spin.setKeyboardTracking(False)
        sr4l.addWidget(self._bc_surface_normal_offset_spin)

        self._bc_surface_keep_largest_check = QCheckBox(
            "Keep largest connected surface", sr4,
        )
        self._bc_surface_keep_largest_check.setChecked(True)
        sr4l.addWidget(self._bc_surface_keep_largest_check)

        sr4l.addWidget(styled_btn(
            sr4, "Export Boundary Surface STL",
            self._export_nifti_boundary_surface_stl, teal=True, small=True,
        ))
        sr4l.addStretch(1)
        s_lay.addWidget(sr4)

        lay.addWidget(surf)
        lay.addWidget(hline(outer))

        # --- boundary condition assignment ----------------------------
        lay.addWidget(_section_title("Boundary condition assignment", outer))
        lay.addWidget(_hint(
            "The panel above decides *where* a boundary condition sits; this "
            "table decides *what* it is. Every exported STL is listed here "
            "and written into simulation_input.json with the preset chosen.",
            outer,
        ))

        assign = _panel(outer)
        a_lay = QVBoxLayout(assign)
        a_lay.setContentsMargins(12, 8, 12, 8)
        a_lay.setSpacing(8)

        self._bc_assignment_table = QTableWidget(0, 3, assign)
        self._bc_assignment_table.setHorizontalHeaderLabels(
            ["ID", "STL file", "BC preset"]
        )
        self._bc_assignment_table.verticalHeader().setVisible(False)
        self._bc_assignment_table.setMinimumHeight(120)
        header = self._bc_assignment_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        a_lay.addWidget(self._bc_assignment_table)

        bt, btl = _row(assign)
        btl.addWidget(styled_btn(
            bt, "Remove selected surface", self._remove_selected_bc_surface,
            small=True,
        ))
        btl.addWidget(styled_btn(
            bt, "Clear all", self._clear_bc_surfaces, small=True,
        ))
        btl.addStretch(1)
        a_lay.addWidget(bt)

        lay.addWidget(assign)
        lay.addWidget(hline(outer))

        # --- simulation file exports ----------------------------------
        lay.addWidget(_section_title("Simulation files", outer))
        self._add_button_rows(outer, lay, [
            ("Export simulation_input.json",
             "Compact C++ input: specimen + bounding_box_mm, material, "
             "boundary conditions, solver, load stepping, output",
             self._export_simulation_input_json, True),
            ("Export export_info.json",
             "Preprocessing record: region of interest, thresholds, affine, "
             "rotation history, corrections applied",
             self._export_info_json, False),
        ])

        lay.addStretch(1)

    # helpers

    @staticmethod
    def _pct_spin(parent, value: float):
        spin = QDoubleSpinBox(parent)
        spin.setRange(0.0, 100.0)
        spin.setDecimals(1)
        spin.setValue(float(value))
        spin.setSuffix(" %")
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _add_button_rows(parent, layout, rows):
        """Add one framed ``[button]  explanation`` row per entry."""
        for label, tip, cmd, accent in rows:
            frame = _panel(parent)
            f_lay = QHBoxLayout(frame)
            f_lay.setContentsMargins(12, 8, 12, 8)
            f_lay.setSpacing(12)
            f_lay.addWidget(
                styled_btn(frame, label, cmd, accent=accent, teal=not accent)
            )
            f_lay.addWidget(_hint(tip, frame, bg=PANEL), 1)
            layout.addWidget(frame)
