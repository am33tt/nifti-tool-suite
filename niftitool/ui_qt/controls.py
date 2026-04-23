"""Left-hand controls panel.

Everything the user adjusts before running an action lives here:
viewer settings, intensity stats, HU calibration, material model,
reorientation, angle rotation, crop ROI.

Each tool is built into its own :class:`_ToolPanel` and stored in
``self._tool_panels``; the main window stacks them and exposes one at
a time via the top navigation bar.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QRadioButton, QVBoxLayout,
    QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, CMAPS, CT_PRESETS, ENTRY_BG, MATERIAL_PRESETS, PANEL,
    TEAL, TEXT, TEXT_DIM,
)
from .widgets import IntRangeRow, styled_btn, styled_entry


class _ToolPanel(QWidget):
    """One tool page shown in the left stacked panel.

    Exposes ``.content`` / ``.content_layout`` so the existing section
    body code (originally written for ``CollapsibleSection``) keeps
    working unchanged.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        title_lbl = QLabel(title, self)
        title_lbl.setFont(QFont("Segoe UI Semibold", 12))
        title_lbl.setStyleSheet(
            f"color: {ACCENT}; background-color: transparent;"
        )
        root.addWidget(title_lbl)

        divider = QFrame(self)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {BORDER};")
        root.addWidget(divider)

        self.content = QWidget(self)
        self.content.setStyleSheet("background-color: transparent;")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 6, 0, 0)
        self.content_layout.setSpacing(3)
        root.addWidget(self.content)
        root.addStretch(1)


# small shims

class _WidgetVar:
    """Shim that lets call-sites use ``.get()`` / ``.set()`` on a QLineEdit
    or QComboBox or QCheckBox (like ``tk.StringVar`` / ``tk.BooleanVar``).
    """

    def __init__(self, widget):
        self._w = widget

    def get(self):
        if isinstance(self._w, QLineEdit):
            return self._w.text()
        if isinstance(self._w, QComboBox):
            return self._w.currentText()
        if isinstance(self._w, QCheckBox):
            return bool(self._w.isChecked())
        return None

    def set(self, value):
        if isinstance(self._w, QLineEdit):
            self._w.setText(str(value))
        elif isinstance(self._w, QComboBox):
            idx = self._w.findText(str(value))
            if idx >= 0:
                self._w.setCurrentIndex(idx)
            else:
                self._w.setEditText(str(value))
        elif isinstance(self._w, QCheckBox):
            self._w.setChecked(bool(value))


def _small_label(text, parent=None, *, width=None, color=TEXT_DIM, bold=False):
    q = QLabel(text, parent) if parent is not None else QLabel(text)
    font = QFont("Segoe UI", 9)
    font.setBold(bold)
    q.setFont(font)
    q.setStyleSheet(f"color: {color}; background-color: transparent;")
    if width is not None:
        q.setFixedWidth(width)
    return q


class ControlsMixin:
    """Adds ``_build_controls`` and all its helpers to :class:`NiftiApp`."""

    def _build_controls(self):
        """Populate ``self._tool_panels`` - one :class:`_ToolPanel` per tool.

        The main window owns a :class:`QStackedWidget` that swaps these in
        as the user clicks the top navigation buttons.
        """
        self._tool_panels: dict[str, _ToolPanel] = {}

        # File info
        sec = _ToolPanel("📋 Metadata / File Info")
        self._tool_panels["metadata"] = sec
        btn = styled_btn(sec.content, "Read Full Metadata",
                         self._do_metadata, small=True)
        sec.content_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Viewer
        sec = _ToolPanel("🎨 Viewer Controls")
        self._tool_panels["viewer"] = sec

        # Colormap / preset rows
        for label, attr_name, default, values in [
            ("Colormap", "_cmap_widget", "gray", CMAPS),
            ("CT Preset", "_preset_widget", "-- preset --", list(CT_PRESETS.keys())),
        ]:
            row = QWidget(sec.content); rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2); rl.setSpacing(4)
            rl.addWidget(_small_label(label, row, width=70))
            cb = QComboBox(row)
            cb.addItems(values)
            idx = cb.findText(default)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            cb.setFixedWidth(160)
            rl.addWidget(cb)
            rl.addStretch(1)
            sec.content_layout.addWidget(row)
            setattr(self, attr_name, cb)

        self._cmap_var = _WidgetVar(self._cmap_widget)
        self._preset_var = _WidgetVar(self._preset_widget)
        self._cmap_widget.currentTextChanged.connect(
            lambda _t: self._refresh_triplanar()
        )
        self._preset_widget.currentTextChanged.connect(
            lambda _t: self._apply_preset()
        )

        # Window W / C entries
        for label, attr_name in [("Window W", "_ww_widget"),
                                 ("Window C", "_wc_widget")]:
            row = QWidget(sec.content); rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2); rl.setSpacing(4)
            rl.addWidget(_small_label(label, row, width=70))
            e = styled_entry(row, width=8)
            e.setText("auto")
            rl.addWidget(e)
            rl.addStretch(1)
            sec.content_layout.addWidget(row)
            setattr(self, attr_name, e)

        self._ww_var = _WidgetVar(self._ww_widget)
        self._wc_var = _WidgetVar(self._wc_widget)

        # Apply/Reset buttons
        btn_row = QWidget(sec.content); brl = QHBoxLayout(btn_row)
        brl.setContentsMargins(0, 4, 0, 0); brl.setSpacing(4)
        brl.addWidget(styled_btn(btn_row, "Apply W/C", self._apply_window_manual, small=True))
        brl.addWidget(styled_btn(btn_row, "Reset Auto", self._reset_window, small=True))
        brl.addStretch(1)
        sec.content_layout.addWidget(btn_row)

        # Intensity statistics
        sec = _ToolPanel("📊 Intensity Statistics")
        self._tool_panels["stats"] = sec
        sec.content_layout.addWidget(
            styled_btn(sec.content, "Compute Histogram",
                       self._do_histogram, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        self._stat_labels: dict = {}
        for key in ("Min", "Max", "Mean", "Std", "p1", "p99"):
            row = QWidget(sec.content); rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 1, 0, 1); rl.setSpacing(4)
            rl.addWidget(_small_label(f"{key:>4}:", row, width=40,
                                      color=TEXT_DIM))
            lv = QLabel("--", row)
            lv.setFont(QFont("Consolas", 10))
            lv.setStyleSheet(f"color: {ACCENT}; background-color: transparent;")
            rl.addWidget(lv); rl.addStretch(1)
            sec.content_layout.addWidget(row)
            self._stat_labels[key] = _LabelVar(lv)

        # HU calibration
        sec = _ToolPanel("🔬 HU Calibration")
        self._tool_panels["hu"] = sec
        sec.content_layout.addWidget(
            _small_label("Two-point linear: HU = m·raw + c", sec.content)
        )

        def _cal_row(lbl_text, attr_name, default, tip=""):
            r = QWidget(sec.content); rl = QHBoxLayout(r)
            rl.setContentsMargins(0, 2, 0, 2); rl.setSpacing(4)
            rl.addWidget(_small_label(lbl_text, r, width=110))
            e = styled_entry(r, width=9); e.setText(str(default))
            rl.addWidget(e)
            if tip:
                rl.addWidget(_small_label(tip, r))
            rl.addStretch(1)
            sec.content_layout.addWidget(r)
            setattr(self, attr_name, e)

        _cal_row("Air intensity", "_cal_air_int_widget", "0", "raw px = air")
        _cal_row("Ref intensity", "_cal_ref_int_widget", "199", "raw px = ref material")
        _cal_row("Air HU", "_cal_air_hu_widget", "-1000", "HU of air")
        _cal_row("Ref HU", "_cal_ref_hu_widget", "0", "HU of ref (water=0)")

        self._cal_air_int = _WidgetVar(self._cal_air_int_widget)
        self._cal_ref_int = _WidgetVar(self._cal_ref_int_widget)
        self._cal_air_hu = _WidgetVar(self._cal_air_hu_widget)
        self._cal_ref_hu = _WidgetVar(self._cal_ref_hu_widget)

        tip_lbl = _small_label(
            "(Use histogram peak for air; known material for ref)",
            sec.content,
        )
        tip_lbl.setWordWrap(True)
        sec.content_layout.addWidget(tip_lbl)

        self._cal_result_label = QLabel("Not calibrated", sec.content)
        self._cal_result_label.setFont(QFont("Consolas", 10))
        self._cal_result_label.setStyleSheet(
            f"color: {TEAL}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._cal_result_label)
        self._cal_result_var = _LabelVar(self._cal_result_label)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "Apply Calibration",
                       self._do_calibrate, teal=True, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        # Material mapping
        sec = _ToolPanel("⚙️  Material Mapping  →  E [MPa]")
        self._tool_panels["material"] = sec

        pr_row = QWidget(sec.content); prl = QHBoxLayout(pr_row)
        prl.setContentsMargins(0, 2, 0, 2); prl.setSpacing(4)
        prl.addWidget(_small_label("Preset", pr_row, width=70))
        self._mat_preset_widget = QComboBox(pr_row)
        self._mat_preset_widget.addItems(list(MATERIAL_PRESETS.keys()))
        self._mat_preset_widget.setCurrentText("AM Concrete (default)")
        self._mat_preset_widget.setFixedWidth(210)
        prl.addWidget(self._mat_preset_widget); prl.addStretch(1)
        sec.content_layout.addWidget(pr_row)
        self._mat_preset_var = _WidgetVar(self._mat_preset_widget)
        self._mat_preset_widget.currentTextChanged.connect(
            lambda _t: self._apply_material_preset()
        )

        mod_row = QWidget(sec.content); mrl = QHBoxLayout(mod_row)
        mrl.setContentsMargins(0, 2, 0, 2); mrl.setSpacing(4)
        mrl.addWidget(_small_label("Model", mod_row, width=70))
        self._model_group = QButtonGroup(mod_row)
        self._model_radios: dict = {}
        for m in ("linear", "power", "bilinear", "table"):
            rb = QRadioButton(m, mod_row)
            rb.setStyleSheet(f"color: {TEXT}; background-color: transparent;")
            rb.setFont(QFont("Segoe UI", 9))
            self._model_group.addButton(rb)
            self._model_radios[m] = rb
            mrl.addWidget(rb)
        self._model_radios["bilinear"].setChecked(True)
        mrl.addStretch(1)
        sec.content_layout.addWidget(mod_row)

        # ``_model_var`` is a tk-compat shim reading from the radio group.
        self._model_var = _RadioVar(self._model_radios, default="bilinear")
        for rb in self._model_radios.values():
            rb.toggled.connect(self._on_model_toggled)

        self._model_params_frame = QWidget(sec.content)
        self._model_params_layout = QVBoxLayout(self._model_params_frame)
        self._model_params_layout.setContentsMargins(0, 2, 0, 2)
        self._model_params_layout.setSpacing(2)
        self._model_params_frame.setStyleSheet(
            f"background-color: {PANEL};"
        )
        sec.content_layout.addWidget(self._model_params_frame)
        self._model_param_vars: dict = {}
        self._model_param_widgets: dict = {}

        # Void thresh / agg thresh
        vt_row = QWidget(sec.content); vrl = QHBoxLayout(vt_row)
        vrl.setContentsMargins(0, 2, 0, 2); vrl.setSpacing(4)
        vrl.addWidget(_small_label("Void thresh", vt_row, width=90))
        self._void_thresh_widget = styled_entry(vt_row, width=8)
        self._void_thresh_widget.setText("-500")
        vrl.addWidget(self._void_thresh_widget)
        vrl.addWidget(_small_label("HU", vt_row)); vrl.addStretch(1)
        sec.content_layout.addWidget(vt_row)
        self._void_thresh_var = _WidgetVar(self._void_thresh_widget)

        ag_row = QWidget(sec.content); arl = QHBoxLayout(ag_row)
        arl.setContentsMargins(0, 2, 0, 2); arl.setSpacing(4)
        arl.addWidget(_small_label("Aggreg. thresh", ag_row, width=90))
        self._agg_thresh_widget = styled_entry(ag_row, width=8)
        arl.addWidget(self._agg_thresh_widget)
        arl.addWidget(_small_label("HU  (blank = 2-phase)", ag_row))
        arl.addStretch(1)
        sec.content_layout.addWidget(ag_row)
        self._agg_thresh_var = _WidgetVar(self._agg_thresh_widget)

        self._mat_note_label = QLabel("", sec.content)
        self._mat_note_label.setWordWrap(True)
        self._mat_note_label.setFont(QFont("Segoe UI", 9))
        self._mat_note_label.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._mat_note_label)
        self._mat_note_var = _LabelVar(self._mat_note_label)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "▶  Compute E-Map",
                       self._do_compute_emap, accent=True, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        # E stats
        self._e_stat_labels: dict = {}
        for key in ("E_min", "E_max", "E_mean", "E_std", "porosity"):
            row = QWidget(sec.content); rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 1, 0, 1); rl.setSpacing(4)
            rl.addWidget(_small_label(f"{key.replace('_', ' '):>10}:", row,
                                      width=80, color=TEXT_DIM))
            lv = QLabel("--", row)
            lv.setFont(QFont("Consolas", 10))
            lv.setStyleSheet(f"color: {TEAL}; background-color: transparent;")
            rl.addWidget(lv); rl.addStretch(1)
            sec.content_layout.addWidget(row)
            self._e_stat_labels[key] = _LabelVar(lv)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "Load Preset JSON...",
                       self._load_preset_json, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        self._update_model_panel()
        self._apply_material_preset()

        # Reorientation
        sec = _ToolPanel("🔄 Reorientation  (no resample)")
        self._tool_panels["reorient"] = sec

        row = QWidget(sec.content); rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 2, 0, 2); rl.setSpacing(4)
        rl.addWidget(_small_label("Target", row, width=50))
        self._reorient_target_widget = styled_entry(row, width=8)
        self._reorient_target_widget.setText("RAS")
        for preset in ("RAS", "LPS", "LAS"):
            b = styled_btn(row, preset,
                           lambda p=preset: self._reorient_target_widget.setText(p),
                           small=True)
            rl.addWidget(b)
        rl.addWidget(self._reorient_target_widget)
        rl.addStretch(1)
        sec.content_layout.addWidget(row)
        self._reorient_target = _WidgetVar(self._reorient_target_widget)

        self._reorient_save_widget = QCheckBox("Choose save path", sec.content)
        self._reorient_save_widget.setChecked(True)
        self._reorient_save_widget.setFont(QFont("Segoe UI", 9))
        self._reorient_save_widget.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._reorient_save_widget)
        self._reorient_save = _WidgetVar(self._reorient_save_widget)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "Run Reorientation",
                       self._do_reorient, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        # Angle Rotation
        sec = _ToolPanel("↩  Angle Rotation  (resample)")
        self._tool_panels["rotate"] = sec

        row = QWidget(sec.content); rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 2, 0, 2); rl.setSpacing(4)
        rl.addWidget(_small_label("Axis", row, width=50))
        self._rot_axis_group = QButtonGroup(row)
        self._rot_axis_radios: dict = {}
        for ax in ('x', 'y', 'z'):
            rb = QRadioButton(ax.upper(), row)
            rb.setFont(QFont("Segoe UI", 10))
            rb.setStyleSheet(f"color: {TEXT}; background-color: transparent;")
            self._rot_axis_group.addButton(rb)
            self._rot_axis_radios[ax] = rb
            rl.addWidget(rb)
        self._rot_axis_radios['z'].setChecked(True)
        rl.addStretch(1)
        sec.content_layout.addWidget(row)
        self._rot_axis = _RadioVar(self._rot_axis_radios, default="z")

        row2 = QWidget(sec.content); r2l = QHBoxLayout(row2)
        r2l.setContentsMargins(0, 2, 0, 2); r2l.setSpacing(4)
        r2l.addWidget(_small_label("Angle °", row2, width=50))
        self._rot_angle_widget = styled_entry(row2, width=7)
        self._rot_angle_widget.setText("45")
        r2l.addWidget(self._rot_angle_widget)
        for deg in (90, 180, 270):
            b = styled_btn(row2, str(deg),
                           lambda d=deg: self._rot_angle_widget.setText(str(d)),
                           small=True)
            r2l.addWidget(b)
        r2l.addStretch(1)
        sec.content_layout.addWidget(row2)
        self._rot_angle = _WidgetVar(self._rot_angle_widget)

        self._rot_save_widget = QCheckBox("Choose save path", sec.content)
        self._rot_save_widget.setChecked(True)
        self._rot_save_widget.setFont(QFont("Segoe UI", 9))
        self._rot_save_widget.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._rot_save_widget)
        self._rot_save = _WidgetVar(self._rot_save_widget)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "Run Rotation",
                       self._do_rotate, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        # Crop
        sec = _ToolPanel("✂️  Crop  (ROI selection)")
        self._tool_panels["crop"] = sec

        self._crop_rows: dict = {}
        for axis in ('X', 'Y', 'Z'):
            r = IntRangeRow(sec.content, axis)
            sec.content_layout.addWidget(r)
            self._crop_rows[axis] = r

        btn_row2 = QWidget(sec.content); b2l = QHBoxLayout(btn_row2)
        b2l.setContentsMargins(0, 4, 0, 0); b2l.setSpacing(4)
        b2l.addWidget(styled_btn(btn_row2, "Auto-fill from shape",
                                 self._autofill_crop, small=True))
        b2l.addStretch(1)
        sec.content_layout.addWidget(btn_row2)

        self._crop_save_widget = QCheckBox("Choose save path", sec.content)
        self._crop_save_widget.setChecked(True)
        self._crop_save_widget.setFont(QFont("Segoe UI", 9))
        self._crop_save_widget.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._crop_save_widget)
        self._crop_save = _WidgetVar(self._crop_save_widget)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "Run Crop",
                       self._do_crop, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

    # Model params panel

    def _on_model_toggled(self, checked: bool):
        # Only rebuild once, on the *newly-checked* button.
        if not checked:
            return
        self._update_model_panel()

    def _update_model_panel(self):
        # Clear children
        while self._model_params_layout.count():
            item = self._model_params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._model_param_vars.clear()
        self._model_param_widgets.clear()

        model = self._model_var.get()
        f = self._model_params_frame
        lay = self._model_params_layout

        def param_row(lbl_text, key, default, unit=""):
            r = QWidget(f); rl = QHBoxLayout(r)
            rl.setContentsMargins(0, 2, 0, 2); rl.setSpacing(4)
            rl.addWidget(_small_label(lbl_text, r, width=100))
            e = styled_entry(r, width=10); e.setText(str(default))
            rl.addWidget(e)
            if unit:
                rl.addWidget(_small_label(unit, r))
            rl.addStretch(1)
            lay.addWidget(r)
            self._model_param_widgets[key] = e
            self._model_param_vars[key] = _WidgetVar(e)

        if model == "linear":
            lay.addWidget(_small_label("E = a·HU + b  [MPa]", f))
            param_row("a  (slope)", "a", "20.0", "MPa/HU")
            param_row("b  (intercept)", "b", "0.0", "MPa")
            param_row("E_void", "E_void", "0.001", "MPa")

        elif model == "power":
            lay.addWidget(_small_label("E = a · HU^b  [MPa]", f))
            param_row("a  (scale)", "a", "0.09", "MPa")
            param_row("b  (exponent)", "b", "1.92")
            param_row("HU min clamp", "hu_min_clamp", "1.0", "HU")
            param_row("E_void", "E_void", "0.001", "MPa")

        elif model == "bilinear":
            lay.addWidget(_small_label("Piecewise: void | solid", f))
            param_row("HU threshold", "hu_thresh", "-200", "HU")
            param_row("E_void", "E_void", "0.001", "MPa")
            param_row("E_solid", "E_solid", "30000.0", "MPa")

        elif model == "table":
            lay.addWidget(_small_label("Piecewise-linear table\nEnter HU values:", f))
            self._table_text_widget = QPlainTextEdit(f)
            self._table_text_widget.setStyleSheet(
                f"QPlainTextEdit {{ background-color: {ENTRY_BG}; "
                f"color: {TEXT}; border: 1px solid {BORDER}; }}"
            )
            self._table_text_widget.setFont(QFont("Consolas", 9))
            self._table_text_widget.setFixedHeight(110)
            self._table_text_widget.setPlainText(
                "# HU_value  E_MPa\n"
                "-1000       0.001\n"
                " -500       0.001\n"
                "    0    5000.0\n"
                "  500   25000.0\n"
                " 1000   40000.0\n"
            )
            lay.addWidget(self._table_text_widget)
            lay.addWidget(_small_label("(Lines starting with # are comments)", f))
            # Provide ``_table_text`` with a ``.get('1.0','end')``-like API
            # so actions.py / this file's _get_model_params can stay
            # close to the tk form.
            self._table_text = _PlainTextShim(self._table_text_widget)

    def _get_model_params(self):
        model = self._model_var.get()
        vars_ = self._model_param_vars
        try:
            if model == "linear":
                return {
                    "a": float(vars_["a"].get()),
                    "b": float(vars_["b"].get()),
                    "E_void": float(vars_["E_void"].get()),
                }
            if model == "power":
                return {
                    "a": float(vars_["a"].get()),
                    "b": float(vars_["b"].get()),
                    "hu_min_clamp": float(vars_["hu_min_clamp"].get()),
                    "E_void": float(vars_["E_void"].get()),
                }
            if model == "bilinear":
                return {
                    "hu_thresh": float(vars_["hu_thresh"].get()),
                    "E_void": float(vars_["E_void"].get()),
                    "E_solid": float(vars_["E_solid"].get()),
                }
            if model == "table":
                lines = self._table_text.get('1.0', 'end').strip().splitlines()
                hu_vals: list[float] = []
                E_vals: list[float] = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    hu_vals.append(float(parts[0]))
                    E_vals.append(float(parts[1]))
                return {"hu": hu_vals, "E": E_vals}
        except Exception as ex:
            raise ValueError(f"Bad parameter input: {ex}")

    def _apply_material_preset(self, *_args):
        name = self._mat_preset_var.get()
        preset = MATERIAL_PRESETS.get(name)
        if not preset:
            return
        self._model_var.set(preset.get("model", "linear"))
        self._void_thresh_var.set(str(preset.get("void_thresh", "-500")))
        self._mat_note_var.set(preset.get("notes", ""))
        self._update_model_panel()
        params = preset.get("params", {})
        for k, v in params.items():
            if k in self._model_param_vars:
                self._model_param_vars[k].set(str(v))
        self._cal_air_int.set(str(preset.get("hu_air", "0")))
        self._cal_ref_int.set(str(preset.get("hu_ref", "199")))

    def _load_preset_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load material preset JSON", "",
            "JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            MATERIAL_PRESETS.update(data)
            # Refresh combo items
            self._mat_preset_widget.blockSignals(True)
            self._mat_preset_widget.clear()
            self._mat_preset_widget.addItems(list(MATERIAL_PRESETS.keys()))
            self._mat_preset_widget.blockSignals(False)
            self._append_log(
                f"  Loaded {len(data)} preset(s) from {Path(path).name}", 'ok',
            )
        except Exception as ex:
            QMessageBox.critical(self, "JSON Error", str(ex))

    # Windowing helpers

    def _apply_preset(self, *_args):
        name = self._preset_var.get()
        val = CT_PRESETS.get(name)
        if val is None:
            return
        ww, wc = val
        self._ww_var.set(str(ww)); self._wc_var.set(str(wc))
        self._ww = float(ww); self._wc = float(wc)
        self._refresh_triplanar()

    def _apply_window_manual(self):
        try:
            self._ww = float(self._ww_var.get())
            self._wc = float(self._wc_var.get())
        except ValueError:
            self._ww = self._wc = None
            self._ww_var.set("auto"); self._wc_var.set("auto")
        self._refresh_triplanar()

    def _reset_window(self):
        self._ww = self._wc = None
        self._ww_var.set("auto"); self._wc_var.set("auto")
        self._preset_var.set("-- preset --")
        self._refresh_triplanar()

    # Crop autofill

    def _autofill_crop(self):
        if self._img is None:
            return
        for axis, dim in zip(('X', 'Y', 'Z'), self._img.shape[:3]):
            self._crop_rows[axis].start_var.set("0")
            self._crop_rows[axis].end_var.set(str(dim))


# small shims reused above

class _LabelVar:
    """QLabel wrapper that exposes ``get()`` / ``set(value)``."""

    def __init__(self, label: QLabel):
        self._lbl = label

    def get(self) -> str:
        return self._lbl.text()

    def set(self, value) -> None:
        self._lbl.setText(str(value))


class _RadioVar:
    """QButtonGroup shim with tk.StringVar-compatible ``get()`` / ``set()``."""

    def __init__(self, radios: dict, default: str):
        self._radios = radios
        self._default = default

    def get(self) -> str:
        for key, rb in self._radios.items():
            if rb.isChecked():
                return key
        return self._default

    def set(self, value) -> None:
        rb = self._radios.get(str(value))
        if rb is not None:
            rb.setChecked(True)


class _PlainTextShim:
    """Tkinter-Text ``.get('1.0','end')`` compatibility shim."""

    def __init__(self, widget: QPlainTextEdit):
        self._w = widget

    def get(self, _start='1.0', _end='end'):
        return self._w.toPlainText()
