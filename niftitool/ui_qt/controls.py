"""Left-hand controls panel.

Holds the settings adjusted before running an action: viewer settings,
intensity statistics, the void/pore threshold, binarisation, beam-hardening
correction, background removal, reorientation, rotation and the crop
region.

Each tool is built into its own :class:`_ToolPanel` and stored in
``self._tool_panels``. The main window stacks them and shows one at a time
via the top navigation bar.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QRadioButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from ..config import ACCENT, BORDER, CMAPS, CT_PRESETS, TEXT, TEXT_DIM
from .widgets import IntRangeRow, styled_btn, styled_entry


class _ToolPanel(QWidget):
    """One tool page shown in the left stacked panel.

    Exposes ``.content`` and ``.content_layout``, the container the section
    body appends its rows to.
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
    """Gives QLineEdit, QComboBox and QCheckBox the ``.get()`` and ``.set()``
    interface the call sites expect from ``tk.StringVar``.
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
    else:
        # An unwrapped label reports its whole text as a minimum width, which
        # the row and then the tool column inherit, pushing the panels past
        # the edge of the pane. Wrapping makes the minimum one word wide, and
        # heightForWidth lets the layout add the rows the label needs.
        _make_wrappable(q)
    return q


def _make_wrappable(label: QLabel, minimum: int = 40) -> QLabel:
    """Let *label* wrap and shrink instead of widening its column."""
    label.setWordWrap(True)
    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
    policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
    policy.setHeightForWidth(True)
    label.setSizePolicy(policy)
    label.setMinimumWidth(minimum)
    return label


def _hint(text, parent):
    """Full-width wrapped note, for the line under a control row."""
    q = QLabel(text, parent)
    q.setFont(QFont("Segoe UI", 8))
    q.setStyleSheet(f"color: {TEXT_DIM}; background-color: transparent;")
    q.setContentsMargins(0, 0, 0, 4)
    return _make_wrappable(q)


def _field_row(parent, label, widget, *, label_width=92):
    """``label  [widget]`` row that never forces the column wider."""
    row = QWidget(parent)
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.setSpacing(4)
    lay.addWidget(_small_label(label, row, width=label_width))
    lay.addWidget(widget)
    lay.addStretch(1)
    return row


class ControlsMixin:
    """Adds ``_build_controls`` and all its helpers to :class:`NiftiApp`."""

    def _build_controls(self):
        """Populate ``self._tool_panels``, one :class:`_ToolPanel` per tool.

        The main window owns a :class:`QStackedWidget` that swaps these in as
        the user clicks the top navigation buttons.
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

        # Void / pore threshold
        sec = _ToolPanel("\u25d1  Threshold")
        self._tool_panels["threshold"] = sec

        sec.content_layout.addWidget(_hint(
            "One grey level separates pore/void from solid material. Every "
            "measurement downstream \u2014 porosity, background removal, "
            "binarisation \u2014 reads it from here, so they always agree "
            "with each other.", sec.content,
        ))

        self._void_thresh_widget = styled_entry(sec.content, width=9)
        self._void_thresh_widget.setText("auto")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Threshold", self._void_thresh_widget))
        self._void_thresh_var = _WidgetVar(self._void_thresh_widget)

        sec.content_layout.addWidget(_hint(
            "'auto' = Otsu on a subsample of the volume. The Porosity tab's "
            "\u25b6 Auto Threshold measures it properly from the specimen "
            "interior and writes the result back here.", sec.content,
        ))

        sec.content_layout.addWidget(
            styled_btn(sec.content, "Auto (Otsu)", self._auto_void_thresh,
                       small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        # Background removal
        sec = _ToolPanel("🧹 Background Removal")
        self._tool_panels["background"] = sec

        sec.content_layout.addWidget(_hint(
            "Finds the specimen as the largest connected solid region with "
            "its internal pores kept, removes everything outside it and "
            "saves a clean new NIfTI. Uses the grey level from the Threshold "
            "panel. The volume is streamed slab by slab and the result is "
            "written as it is computed, so the file size is not limited by "
            "the machine's RAM, and every voxel that survives is copied "
            "unchanged.", sec.content,
        ))

        self._bg_boundary_combo = QComboBox(sec.content)
        for label, mode in (
            ("largest solid body", "component"),
            ("radial envelope", "envelope"),
        ):
            self._bg_boundary_combo.addItem(label, mode)
        sec.content_layout.addWidget(
            _field_row(sec.content, "Outer boundary", self._bg_boundary_combo))

        self._bg_fill_combo = QComboBox(sec.content)
        for label, mode in (
            ("zero", "zero"),
            ("air median", "air-median"),
            ("grey level...", "value"),
        ):
            self._bg_fill_combo.addItem(label, mode)
        sec.content_layout.addWidget(
            _field_row(sec.content, "Fill outside", self._bg_fill_combo))

        self._bg_fill_widget = styled_entry(sec.content, width=9)
        self._bg_fill_widget.setText("0")
        self._bg_fill_row = _field_row(
            sec.content, "Grey level", self._bg_fill_widget)
        self._bg_fill_row.setVisible(False)
        sec.content_layout.addWidget(self._bg_fill_row)
        self._bg_fill_combo.currentIndexChanged.connect(
            lambda _index: self._bg_fill_row.setVisible(
                self._bg_fill_combo.currentData() == "value"))

        self._bg_margin_widget = styled_entry(sec.content, width=6)
        self._bg_margin_widget.setText("10")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Margin", self._bg_margin_widget))

        self._bg_crop_widget = QCheckBox(
            "Crop to specimen bounding box", sec.content,
        )
        self._bg_crop_widget.setChecked(True)
        self._bg_crop_widget.setFont(QFont("Segoe UI", 9))
        self._bg_crop_widget.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._bg_crop_widget)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "▶  Remove Background && Save...",
                       self._do_remove_background, accent=True, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        # Binarisation for the finite cell method
        sec = _ToolPanel("\u25a3  Binarise")
        self._tool_panels["binarize"] = sec

        sec.content_layout.addWidget(_hint(
            "Writes the indicator field the finite cell method integrates "
            "against: uint8, 1 where material, 0 where pore or background. "
            "The threshold is measured inside the specimen envelope, so air "
            "around the part cannot drag it. Results appear in the Binarise "
            "tab.", sec.content,
        ))

        self._bin_method_combo = QComboBox(sec.content)
        self._bin_method_combo.addItems(
            ["otsu", "isodata", "valley", "sigma", "triangle", "manual"])
        sec.content_layout.addWidget(
            _field_row(sec.content, "Method", self._bin_method_combo))

        self._bin_k_widget = styled_entry(sec.content, width=6)
        self._bin_k_widget.setText("3.0")
        sec.content_layout.addWidget(
            _field_row(sec.content, "sigma k", self._bin_k_widget))

        self._bin_manual_widget = styled_entry(sec.content, width=10)
        self._bin_manual_widget.setPlaceholderText("grey value")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Manual value", self._bin_manual_widget))

        self._bin_smooth_widget = styled_entry(sec.content, width=6)
        self._bin_smooth_widget.setText("0")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Smooth", self._bin_smooth_widget))

        self._bin_downsample_widget = styled_entry(sec.content, width=6)
        self._bin_downsample_widget.setText("1")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Downsample", self._bin_downsample_widget))

        self._bin_minvoid_widget = styled_entry(sec.content, width=6)
        self._bin_minvoid_widget.setText("0")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Min void", self._bin_minvoid_widget))

        self._bin_largest_check = QCheckBox(
            "Keep only the largest body", sec.content)
        self._bin_largest_check.setFont(QFont("Segoe UI", 9))
        self._bin_largest_check.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;")
        sec.content_layout.addWidget(self._bin_largest_check)

        self._bin_refine_check = QCheckBox(
            "Refine inside the specimen", sec.content)
        self._bin_refine_check.setChecked(True)
        self._bin_refine_check.setFont(QFont("Segoe UI", 9))
        self._bin_refine_check.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;")
        sec.content_layout.addWidget(self._bin_refine_check)

        bin_btns = QWidget(sec.content); bbl = QHBoxLayout(bin_btns)
        bbl.setContentsMargins(0, 4, 0, 0); bbl.setSpacing(4)
        bbl.addWidget(styled_btn(bin_btns, "\u25b6  Preview",
                                 self._do_binarize_preview, accent=True,
                                 small=True))
        bbl.addWidget(styled_btn(bin_btns, "Write NIfTI...",
                                 self._do_binarize_write, small=True))
        bbl.addStretch(1)
        sec.content_layout.addWidget(bin_btns)

        sec.content_layout.addWidget(
            styled_btn(sec.content, "Threshold sweep",
                       self._do_binarize_sweep, small=True),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        sec.content_layout.addWidget(_hint(
            "The sweep reports the void fraction against threshold = mode "
            "\u2212 k\u00b7sigma. A real pore population shows up as a "
            "plateau; a value picked off the steep part is measuring the "
            "material's own noise.", sec.content,
        ))

        # Beam-hardening correction
        sec = _ToolPanel("\u2600  Beam Hardening")
        self._tool_panels["beam_hardening"] = sec

        sec.content_layout.addWidget(_hint(
            "A polychromatic beam hardens as it penetrates, so the surface "
            "layer of a specimen reconstructs brighter than its core. The "
            "depth of every voxel below the real specimen surface is "
            "measured with a distance transform \u2014 not as a distance to "
            "the image border \u2014 so cylinders, notched and T-shaped "
            "parts, scans with air around them and anisotropic voxels are "
            "all handled correctly. Results appear in the Beam Hardening "
            "tab.", sec.content,
        ))

        self._bh_poly_deg = styled_entry(sec.content, width=6)
        self._bh_poly_deg.setText("4")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Poly degree", self._bh_poly_deg))

        self._bh_bin_mm = styled_entry(sec.content, width=6)
        self._bh_bin_mm.setText("0.25")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Bin width", self._bh_bin_mm))

        self._bh_stride_widget = styled_entry(sec.content, width=6)
        self._bh_stride_widget.setText("0")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Stride", self._bh_stride_widget))

        self._bh_border_check = QCheckBox(
            "Image border is specimen surface", sec.content,
        )
        self._bh_border_check.setChecked(True)
        self._bh_border_check.setFont(QFont("Segoe UI", 9))
        self._bh_border_check.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._bh_border_check)

        self._bh_removebg_check = QCheckBox(
            "Zero the background outside", sec.content,
        )
        self._bh_removebg_check.setFont(QFont("Segoe UI", 9))
        self._bh_removebg_check.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        sec.content_layout.addWidget(self._bh_removebg_check)

        self._bh_dtype_combo = QComboBox(sec.content)
        self._bh_dtype_combo.addItems(["float32", "int16"])
        sec.content_layout.addWidget(
            _field_row(sec.content, "Output dtype", self._bh_dtype_combo))

        bh_btns = QWidget(sec.content); bhl = QHBoxLayout(bh_btns)
        bhl.setContentsMargins(0, 4, 0, 0); bhl.setSpacing(4)
        bhl.addWidget(styled_btn(bh_btns, "\u25b6  Measure",
                                 self._do_beam_hardening_fit, accent=True,
                                 small=True))
        bhl.addWidget(styled_btn(bh_btns, "Apply && Save...",
                                 self._do_beam_hardening_apply, small=True))
        bhl.addStretch(1)
        sec.content_layout.addWidget(bh_btns)

        # Reorientation
        sec = _ToolPanel("🔄  Reorient")
        self._tool_panels["reorient"] = sec

        sec.content_layout.addWidget(_hint(
            "Relabels the axes to the target orientation by flipping and "
            "permuting them. No interpolation, so no voxel value changes.",
            sec.content,
        ))

        self._reorient_target_widget = styled_entry(sec.content, width=8)
        self._reorient_target_widget.setText("RAS")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Target", self._reorient_target_widget,
                       label_width=54))

        preset_row = QWidget(sec.content); prl = QHBoxLayout(preset_row)
        prl.setContentsMargins(0, 2, 0, 2); prl.setSpacing(4)
        for preset in ("RAS", "LPS", "LAS"):
            prl.addWidget(styled_btn(
                preset_row, preset,
                lambda p=preset: self._reorient_target_widget.setText(p),
                small=True))
        prl.addStretch(1)
        sec.content_layout.addWidget(preset_row)
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
        sec = _ToolPanel("↩  Rotate  (resamples)")
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

        self._rot_angle_widget = styled_entry(sec.content, width=7)
        self._rot_angle_widget.setText("45")
        sec.content_layout.addWidget(
            _field_row(sec.content, "Angle °", self._rot_angle_widget,
                       label_width=54))

        quick = QWidget(sec.content); qrl = QHBoxLayout(quick)
        qrl.setContentsMargins(0, 2, 0, 2); qrl.setSpacing(4)
        for deg in (90, 180, 270):
            qrl.addWidget(styled_btn(
                quick, str(deg),
                lambda d=deg: self._rot_angle_widget.setText(str(d)),
                small=True))
        qrl.addStretch(1)
        sec.content_layout.addWidget(quick)
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
        sec = _ToolPanel("✂️  Crop")
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

    # Automatic void/solid threshold

    def _auto_void_thresh(self):
        """Pick the void/solid split automatically (Otsu on a subsample)."""
        if self._img is None:
            QMessageBox.warning(self, "No file",
                                "Please open a NIfTI file first.")
            return

        def _run():
            try:
                from ..core.segmentation import otsu_threshold

                self._set_status("Finding threshold (Otsu)...", busy=True)
                vol = self._get_gray_lazy()
                if hasattr(vol, "subsample_flat"):
                    # Lazy volume, sample without materialising it.
                    flat = vol.subsample_flat(2_000_000)
                else:
                    flat = vol.ravel()
                    if flat.size > 2_000_000:
                        flat = flat[:: flat.size // 2_000_000]
                t = otsu_threshold(flat)
                if self._abort_if_stopped("Auto threshold"):
                    return
                self.after(0, self._void_thresh_var.set, f"{t:.1f}")
                self._append_log(
                    f"  Auto void threshold (Otsu): {t:.1f}", 'teal')
                self._set_status("Auto threshold set.", busy=False)
            except Exception as ex:
                self._append_log(f"  Auto threshold error: {ex}", 'err')
                self._set_status("Auto threshold error.", busy=False)

        self._run_task("Auto threshold", _run)

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
