"""3-D viewer backed by VTK.

Embeds the VTK render window via QVTKRenderWindowInteractor.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontMetrics, QIntValidator
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QSlider,
    QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, AXIS_COLOR, BG, BORDER, ENTRY_BG, PANEL2, TEXT, TEXT_DIM, WARN,
)
from ..core.io import LazyGrayVolume
from ..core.windowing import auto_window
from ..deps import HAS_VTK, np
from ..utils import available_ram_mb
from .widgets import styled_btn


class View3DMixin:
    """Adds the GPU 3-D View tab to :class:`NiftiApp`."""

    # build

    def _build_3d_view(self, parent):
        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not HAS_VTK:
            lbl = QLabel(
                "VTK not installed.\n\n"
                "Install with:  pip install vtk\n\n"
                "VTK is required for the GPU 3-D viewer (Slicer uses "
                "the same library).",
                parent,
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {WARN}; background-color: {BG};")
            lbl.setFont(QFont("Segoe UI", 10))
            root.addWidget(lbl)
            return

        import vtk
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        # top toolbar
        ctrl = QWidget(parent)
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(8, 4, 8, 4)
        ctrl_lay.setSpacing(8)
        ctrl_lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        base_font = QFont("Segoe UI", 9)
        BAR_H = 26  # unified pixel height for every toolbar control

        # Mode radios
        mode_row = QWidget(ctrl)
        mr_lay = QHBoxLayout(mode_row)
        mr_lay.setContentsMargins(0, 0, 0, 0); mr_lay.setSpacing(6)
        mr_lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._3d_mode_group = QButtonGroup(mode_row)
        self._3d_mode_radios: dict = {}
        mode_rb_style = (
            f"QRadioButton {{ color: {TEXT_DIM}; background-color: transparent; }}"
            f"QRadioButton:checked {{ color: {ACCENT}; font-weight: bold; }}"
        )
        bold_font = QFont("Segoe UI", 9)
        bold_font.setBold(True)
        for label, value in (("Block", "volume"), ("Sliced", "planes")):
            rb = QRadioButton(label, mode_row)
            rb.setFont(base_font)
            rb.setStyleSheet(mode_rb_style)
            rb.setFixedHeight(BAR_H)
            # Reserve room for the bold checked label so switching modes does
            # not reflow the toolbar; 24 px covers the indicator and its gap.
            rb.setMinimumWidth(QFontMetrics(bold_font).horizontalAdvance(label) + 24)
            self._3d_mode_group.addButton(rb)
            self._3d_mode_radios[value] = rb
            mr_lay.addWidget(rb)
        self._3d_mode_radios["volume"].setChecked(True)
        self._3d_mode = _RadioVar(self._3d_mode_radios, default="volume")
        for rb in self._3d_mode_radios.values():
            rb.toggled.connect(self._on_3d_mode_toggled)
        ctrl_lay.addWidget(mode_row, 0, Qt.AlignmentFlag.AlignVCenter)

        render_btn = styled_btn(ctrl, "▶  Render", self._do_3d_render,
                                accent=True, small=True)
        render_btn.setFixedHeight(BAR_H)
        ctrl_lay.addWidget(render_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        reset_btn = styled_btn(ctrl, "Reset View", self._reset_3d_camera, small=True)
        reset_btn.setFixedHeight(BAR_H)
        ctrl_lay.addWidget(reset_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        # Camera presets. Labels follow radiology convention (A=anterior,
        # P=posterior, L/R=patient left/right, S=superior, I=inferior).
        preset_lbl = QLabel("View:", ctrl)
        preset_lbl.setFont(base_font)
        preset_lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: transparent;")
        ctrl_lay.addWidget(preset_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        self._3d_view_combo = QComboBox(ctrl)
        self._3d_view_combo.setFont(base_font)
        self._3d_view_combo.setFixedHeight(BAR_H)
        self._3d_view_combo.setToolTip("Jump the camera to a preset angle.")
        for label, preset in (
            ("Select view", None),
            ("Superior (top)",      "superior"),
            ("Inferior (bottom)",   "inferior"),
            ("Anterior (front)",    "anterior"),
            ("Posterior (back)",    "posterior"),
            ("Left (sagittal)",     "left"),
            ("Right (sagittal)",    "right"),
            ("Isometric",           "iso"),
        ):
            self._3d_view_combo.addItem(label, preset)
        self._3d_view_combo.currentIndexChanged.connect(self._on_3d_view_preset_picked)
        ctrl_lay.addWidget(self._3d_view_combo, 0, Qt.AlignmentFlag.AlignVCenter)

        sync_btn = styled_btn(ctrl, "Sync to Tri-Planar",
                              self._sync_3d_to_triplanar, small=True)
        sync_btn.setFixedHeight(BAR_H)
        ctrl_lay.addWidget(sync_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        self._3d_axes_cb = QCheckBox("Axes", ctrl)
        self._3d_axes_cb.setChecked(False)
        self._3d_axes_cb.setFont(base_font)
        self._3d_axes_cb.setFixedHeight(BAR_H)
        self._3d_axes_cb.setToolTip(
            "Show the labelled bounding-box axes on the volume. "
            "The small XYZ gizmo in the corner is always visible."
        )
        self._3d_axes_cb.setStyleSheet(f"color: {TEXT}; background-color: transparent;")
        self._3d_axes_cb.toggled.connect(self._on_3d_axes_toggled)
        ctrl_lay.addWidget(self._3d_axes_cb, 0, Qt.AlignmentFlag.AlignVCenter)

        self._3d_move_cb = QCheckBox("Move", ctrl)
        self._3d_move_cb.setChecked(False)
        self._3d_move_cb.setFont(base_font)
        self._3d_move_cb.setFixedHeight(BAR_H)
        self._3d_move_cb.setToolTip(
            "When on, click-drag translates the picked volume or STL "
            "instead of rotating the view, useful for aligning an STL "
            "overlay with the volume."
        )
        self._3d_move_cb.setStyleSheet(f"color: {TEXT}; background-color: transparent;")
        self._3d_move_cb.toggled.connect(self._on_3d_move_toggled)
        ctrl_lay.addWidget(self._3d_move_cb, 0, Qt.AlignmentFlag.AlignVCenter)

        stl_btn = styled_btn(ctrl, "+ STL", self._load_stl_dialog, small=True)
        stl_btn.setFixedHeight(BAR_H)
        ctrl_lay.addWidget(stl_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        ctrl_lay.addStretch(1)
        root.addWidget(ctrl)

        # STL overlay list (hidden until the user loads one)
        self._stl_list_frame = QFrame(parent)
        self._stl_list_frame.setStyleSheet(f"background-color: {PANEL2};")
        stl_lay = QVBoxLayout(self._stl_list_frame)
        stl_lay.setContentsMargins(8, 2, 8, 2)
        stl_lay.setSpacing(2)
        self._stl_list_layout = stl_lay
        self._stl_list_frame.setVisible(False)
        root.addWidget(self._stl_list_frame)

        # slider bar (only used in "planes" mode)
        sliders = QFrame(parent)
        sliders.setStyleSheet(f"background-color: {PANEL2};")
        sgrid = QGridLayout(sliders)
        sgrid.setContentsMargins(8, 4, 8, 4)
        sgrid.setHorizontalSpacing(16)

        self._3d_idx: dict = {'X': 0, 'Y': 0, 'Z': 0}
        self._3d_idx_widgets: dict = {}
        self._3d_sliders: dict = {}
        self._3d_plane_visible: dict = {'X': True, 'Y': True, 'Z': True}
        self._3d_plane_checkboxes: dict = {}
        labels = {'X': "Sagittal (X)", 'Y': "Coronal (Y)", 'Z': "Axial (Z)"}
        for col, ax in enumerate(('X', 'Y', 'Z')):
            cf = QWidget(sliders)
            cf_lay = QVBoxLayout(cf)
            cf_lay.setContentsMargins(0, 0, 0, 0); cf_lay.setSpacing(2)

            title_row = QWidget(cf)
            title_lay = QHBoxLayout(title_row)
            title_lay.setContentsMargins(0, 0, 0, 0); title_lay.setSpacing(4)
            # Per-axis visibility toggle.
            vis_cb = QCheckBox(title_row)
            vis_cb.setChecked(True)
            vis_cb.setToolTip(f"Show / hide the {labels[ax]} slice plane.")
            vis_cb.setStyleSheet("background-color: transparent;")
            vis_cb.toggled.connect(
                lambda on, a=ax: self._on_3d_plane_visibility_toggled(a, on)
            )
            title_lay.addWidget(vis_cb)
            self._3d_plane_checkboxes[ax] = vis_cb

            title_lbl = QLabel(labels[ax], title_row)
            title_lbl.setFont(QFont("Segoe UI", 9))
            title_lbl.setStyleSheet(f"color: {ACCENT}; background-color: {PANEL2};")
            title_lay.addWidget(title_lbl, 1)
            cf_lay.addWidget(title_row)

            row_w = QWidget(cf)
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(0, 0, 0, 0); row_lay.setSpacing(4)
            sl = QSlider(Qt.Orientation.Horizontal, row_w)
            sl.setMinimum(0); sl.setMaximum(100)
            row_lay.addWidget(sl, 1)
            idx_edit = QLineEdit("--", row_w)
            idx_edit.setFont(QFont("Segoe UI", 9))
            idx_edit.setStyleSheet(
                f"color: {ACCENT}; background-color: {PANEL2}; "
                f"border: 1px solid {BORDER}; padding: 1px 3px;"
            )
            idx_edit.setFixedWidth(50)
            idx_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            idx_edit.setValidator(QIntValidator(0, 10_000, idx_edit))
            idx_edit.setToolTip("Type a slice index and press Enter to jump.")
            row_lay.addWidget(idx_edit)
            cf_lay.addWidget(row_w)

            sgrid.addWidget(cf, 0, col)
            sgrid.setColumnStretch(col, 1)

            sl.valueChanged.connect(
                lambda val, a=ax: self._on_3d_drag(a, val)
            )
            idx_edit.editingFinished.connect(
                lambda a=ax, e=idx_edit: self._on_3d_idx_entered(a, e)
            )
            self._3d_sliders[ax] = sl
            self._3d_idx_widgets[ax] = idx_edit

        root.addWidget(sliders)

        # Tk-compat shim for ``_3d_idx_var[ax].set(...)``
        self._3d_idx_var = {
            ax: _QLabelVar(lbl) for ax, lbl in self._3d_idx_widgets.items()
        }

        # VTK render widget
        self._vtk_widget = QVTKRenderWindowInteractor(parent)
        root.addWidget(self._vtk_widget, 1)

        rw = self._vtk_widget.GetRenderWindow()
        self._vtk_renderer = vtk.vtkRenderer()
        self._vtk_renderer.SetBackground(1.0, 1.0, 1.0)
        self._vtk_renderer.SetBackground2(0.92, 0.94, 0.98)
        self._vtk_renderer.GradientBackgroundOn()
        rw.AddRenderer(self._vtk_renderer)

        self._vtk_iren = rw.GetInteractor()
        self._vtk_iren.Initialize()
        # Do not call ``Start()``: the Qt event loop drives the interactor.
        # While the camera moves, vtkSmartVolumeMapper trades sample distance
        # for speed to meet the update rate. The EndInteractionEvent observer
        # below restores full quality once it stops.
        self._vtk_iren.SetDesiredUpdateRate(15.0)
        self._vtk_iren.SetStillUpdateRate(0.5)
        self._vtk_iren.AddObserver(
            'EndInteractionEvent', self._on_3d_interaction_end
        )

        # Two interactor styles: camera (rotate the scene) and actor (drag
        # the grabbed prop), camera by default. ``MotionFactor`` damps both.
        self._style_camera = vtk.vtkInteractorStyleTrackballCamera()
        self._style_actor = vtk.vtkInteractorStyleTrackballActor()
        # Only the camera style exposes ``SetMotionFactor``; the actor style
        # translates in screen pixels.
        for style in (self._style_camera, self._style_actor):
            if hasattr(style, 'SetMotionFactor'):
                style.SetMotionFactor(6.0)
            style.AddObserver(
                'EndInteractionEvent', self._on_3d_interaction_end
            )
        self._vtk_iren.SetInteractorStyle(self._style_camera)

        self._setup_3d_lighting()

        self._vtk_image = None
        self._vtk_volume = None
        self._vtk_plane_actors = []
        self._vtk_plane_borders = {}
        self._vtk_outline_actor = None
        self._vtk_stats = None
        self._vtk_built_for_id = None
        self._vtk_built_shape = None
        self._vtk_first_render_done = False

        self._3d_redraw_pending = None

        # Corner orientation marker plus a cube-axes actor on the volume
        # bounds, reporting voxel extents in the viewport.
        self._cube_axes = None
        self._build_axes_indicators()

        # STL overlay registry. Each entry:
        # {"path": Path, "name": str, "color": (r,g,b), "actor": vtkActor,
        #  "row": QWidget, "checkbox": QCheckBox}
        self._stl_overlays: list[dict] = []

    # STL overlays (boundary-condition preprocessing)

    # Qualitative palette for BC regions, cycled when more STLs than colors
    # are loaded.
    _STL_PALETTE = (
        "#E5484D", "#2FA84F", "#3B82F6", "#F59E0B",
        "#8B5CF6", "#14B8A6", "#EC4899", "#06B6D4",
    )

    def _load_stl_dialog(self):
        if not HAS_VTK:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self._vtk_widget, "Load STL overlay",
            "", "STL files (*.stl);;All files (*)",
        )
        for p in paths:
            self._add_stl_overlay(p)

    def _add_stl_overlay(self, path: str):
        import vtk
        p = Path(path)
        if not p.exists():
            self._append_log(f"  STL not found: {p}", 'err')
            return

        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(p))
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        mapper.ScalarVisibilityOff()

        color_hex = self._STL_PALETTE[len(self._stl_overlays) % len(self._STL_PALETTE)]
        rgb = self._hex_to_rgb(color_hex)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*rgb)
        prop.SetOpacity(0.55)
        prop.SetEdgeVisibility(0)

        self._vtk_renderer.AddActor(actor)

        row, checkbox = self._make_stl_row(p.name, color_hex)
        self._stl_list_layout.addWidget(row)
        self._stl_list_frame.setVisible(True)

        entry = {
            "path": p, "name": p.name, "color": color_hex,
            "actor": actor, "row": row, "checkbox": checkbox,
        }
        self._stl_overlays.append(entry)

        checkbox.toggled.connect(
            lambda on, e=entry: self._on_stl_visibility(e, on)
        )

        # With no volume framing the scene, reset the camera onto the STL.
        if self._vtk_image is None and len(self._stl_overlays) == 1:
            self._vtk_renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()
        self._append_log(f"  Loaded STL overlay: {p.name}", 'info')

    def _make_stl_row(self, name: str, color_hex: str):
        row = QWidget(self._stl_list_frame)
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(6)

        cb = QCheckBox(row)
        cb.setChecked(True)
        cb.setStyleSheet("background-color: transparent;")
        row_lay.addWidget(cb)

        dot = QLabel(row)
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(
            f"background-color: {color_hex}; border-radius: 5px;"
        )
        row_lay.addWidget(dot)

        lbl = QLabel(name, row)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet(f"color: {TEXT}; background-color: transparent;")
        row_lay.addWidget(lbl, 1)

        focus = QPushButton("Focus", row)
        focus.setFlat(True)
        focus.setToolTip("Zoom the camera to this STL's bounds.")
        focus.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: transparent; border: none;"
        )
        focus.clicked.connect(lambda _=False, r=row: self._focus_stl_by_row(r))
        row_lay.addWidget(focus)

        rm = QPushButton("×", row)
        rm.setFixedWidth(22)
        rm.setFlat(True)
        rm.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: transparent; "
            f"border: none; font-weight: bold;"
        )
        rm.clicked.connect(lambda _=False, r=row: self._remove_stl_row(r))
        row_lay.addWidget(rm)

        return row, cb

    def _focus_stl_by_row(self, row):
        entry = next((e for e in self._stl_overlays if e["row"] is row), None)
        if entry is not None:
            self._focus_on_stl(entry)

    def _on_stl_visibility(self, entry: dict, visible: bool):
        entry["actor"].SetVisibility(1 if visible else 0)
        self._vtk_widget.GetRenderWindow().Render()

    def _on_3d_move_toggled(self, checked: bool):
        """Swap the interactor style between rotate-camera and translate-actor.

        Actor mode drags an STL overlay into alignment with the volume when
        the two were authored in different coordinate frames.
        """
        style = self._style_actor if checked else self._style_camera
        self._vtk_iren.SetInteractorStyle(style)

    def _focus_on_stl(self, entry: dict):
        """Zoom the camera onto a single STL's bounds."""
        actor = entry["actor"]
        b = actor.GetBounds()
        self._vtk_renderer.ResetCamera(b)
        self._vtk_widget.GetRenderWindow().Render()

    def _remove_stl_row(self, row):
        entry = next((e for e in self._stl_overlays if e["row"] is row), None)
        if entry is None:
            return
        self._vtk_renderer.RemoveActor(entry["actor"])
        entry["row"].setParent(None)
        entry["row"].deleteLater()
        self._stl_overlays.remove(entry)
        if not self._stl_overlays:
            self._stl_list_frame.setVisible(False)
        self._vtk_widget.GetRenderWindow().Render()

    # axis indicators

    def _build_axes_indicators(self):
        """Corner XYZ gizmo that rotates with the camera. Built once."""
        import vtk

        marker = vtk.vtkAxesActor()
        marker.SetXAxisLabelText("X")
        marker.SetYAxisLabelText("Y")
        marker.SetZAxisLabelText("Z")
        for cap in (marker.GetXAxisCaptionActor2D(),
                    marker.GetYAxisCaptionActor2D(),
                    marker.GetZAxisCaptionActor2D()):
            tp = cap.GetCaptionTextProperty()
            tp.SetColor(0.1, 0.1, 0.1)
            tp.ShadowOff()
            tp.BoldOff()
            tp.ItalicOff()
        self._axes_marker_actor = marker

        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(marker)
        widget.SetInteractor(self._vtk_iren)
        widget.SetViewport(0.0, 0.0, 0.18, 0.22)
        # The corner gizmo is always on. The "Axes" checkbox governs the
        # on-volume cube-axes actor, not this marker.
        widget.SetEnabled(1)
        widget.InteractiveOff()
        self._axes_marker_widget = widget

    @staticmethod
    def _hex_to_rgb(h: str):
        h = h.lstrip('#')
        return (int(h[0:2], 16) / 255.0,
                int(h[2:4], 16) / 255.0,
                int(h[4:6], 16) / 255.0)

    def _ensure_cube_axes(self, shape, spacing):
        """Rebuild the cube-axes bounds and tick style for the current volume.

        Ticks are in world units (voxel index * spacing), so a 400x400x600
        volume at unit spacing reads 400, 400, 600 on the X, Y, Z edges.
        """
        import vtk

        nx, ny, nz = shape
        sx, sy, sz = spacing
        bounds = (0.0, nx * sx, 0.0, ny * sy, 0.0, nz * sz)

        if self._cube_axes is None:
            axes = vtk.vtkCubeAxesActor()
            axes.SetCamera(self._vtk_renderer.GetActiveCamera())
            # Titles are a single space rather than "": an empty string makes
            # the internal vtkVectorText report "Text is not set!" every
            # frame. Opacity 0 hides the space.
            axes.SetXTitle(" "); axes.SetYTitle(" "); axes.SetZTitle(" ")
            axes.SetXUnits(""); axes.SetYUnits(""); axes.SetZUnits("")
            axes.SetFlyModeToStaticEdges()
            axes.XAxisMinorTickVisibilityOff()
            axes.YAxisMinorTickVisibilityOff()
            axes.ZAxisMinorTickVisibilityOff()
            axes.DrawXGridlinesOff()
            axes.DrawYGridlinesOff()
            axes.DrawZGridlinesOff()
            # Switch labels to screen-space 2D text so SetFontSize has a
            # predictable effect (VTK 8.x and later support SetUse2DMode).
            try:
                axes.SetUse2DMode(1)
            except AttributeError:
                pass
            for i, ax_letter in enumerate(('X', 'Y', 'Z')):
                rgb = self._hex_to_rgb(AXIS_COLOR[ax_letter])
                tp_title = axes.GetTitleTextProperty(i)
                tp_title.SetOpacity(0.0)
                tp_title.SetFontSize(1)
                lp = axes.GetLabelTextProperty(i)
                lp.SetColor(*rgb)
                lp.SetFontSize(11)
                lp.BoldOn()
                lp.ShadowOn()
                lp.ItalicOff()
            for line_prop in (axes.GetXAxesLinesProperty(),
                              axes.GetYAxesLinesProperty(),
                              axes.GetZAxesLinesProperty()):
                line_prop.SetLineWidth(3.0)
            axes.GetXAxesLinesProperty().SetColor(*self._hex_to_rgb(AXIS_COLOR['X']))
            axes.GetYAxesLinesProperty().SetColor(*self._hex_to_rgb(AXIS_COLOR['Y']))
            axes.GetZAxesLinesProperty().SetColor(*self._hex_to_rgb(AXIS_COLOR['Z']))
            # Gridline properties also govern the small ticks at each label.
            for tick_prop in (axes.GetXAxesGridlinesProperty(),
                              axes.GetYAxesGridlinesProperty(),
                              axes.GetZAxesGridlinesProperty()):
                tick_prop.SetLineWidth(1.5)
            self._cube_axes = axes

        self._cube_axes.SetBounds(*bounds)
        self._cube_axes.SetVisibility(self._3d_axes_cb.isChecked())

    def _on_3d_axes_toggled(self, checked: bool):
        # Toggles the on-volume cube-axes actor only; the corner gizmo stays.
        if self._cube_axes is not None:
            self._cube_axes.SetVisibility(bool(checked))
        if hasattr(self, '_vtk_widget'):
            self._vtk_widget.GetRenderWindow().Render()

    def _on_volume_materialised(self):
        """The lazy proxy was promoted to a real ndarray; invalidate VTK
        image data built from the previous array."""
        self._vtk_built_for_id = None
        self._vtk_built_shape = None
        # Render now only if the 3-D tab is current; app.py handles the
        # deferred case on tab switch.
        if hasattr(self, '_nb') and self._nb.currentWidget() is self._viewer3d_tab:
            self._do_3d_render()

    def maybe_auto_render_3d(self):
        """Called by app.py when the 3-D View tab becomes current."""
        if not HAS_VTK or self._gray is None:
            return
        if not self._vtk_first_render_done:
            self._do_3d_render()

    # slider / mode handlers

    def _on_3d_mode_toggled(self, checked: bool):
        if not checked:
            return
        # Mode determines which actor is in the renderer.
        self._do_3d_render()

    def _on_3d_mode_changed(self):
        self._do_3d_render()

    def _on_3d_drag(self, axis, val):
        idx = int(val)
        self._3d_idx[axis] = idx
        w = self._3d_idx_widgets[axis]
        w.blockSignals(True)
        w.setText(str(idx))
        w.blockSignals(False)
        if self._3d_mode.get() != "planes":
            return
        self._update_plane_position(axis, idx)

    def _on_3d_idx_entered(self, axis, edit):
        """Jump the 3-D slice slider to a typed index.

        Only visible in "planes" mode. The slider is synced in either mode so
        the value survives a mode switch.
        """
        txt = edit.text().strip()
        if not txt:
            edit.setText(str(self._3d_idx[axis]))
            return
        try:
            idx = int(txt)
        except ValueError:
            edit.setText(str(self._3d_idx[axis]))
            return
        sl = self._3d_sliders[axis]
        idx = max(sl.minimum(), min(idx, sl.maximum()))
        edit.setText(str(idx))
        if sl.value() == idx:
            return
        sl.setValue(idx)

    def _sync_3d_to_triplanar(self):
        if self._gray is None:
            return
        for ax in ('X', 'Y', 'Z'):
            idx = int(getattr(self, '_tri_idx', {}).get(ax, 0))
            self._3d_idx[ax] = idx
            self._3d_idx_widgets[ax].setText(str(idx))
            sl = self._3d_sliders[ax]
            sl.blockSignals(True)
            sl.setValue(idx)
            sl.blockSignals(False)
            if self._3d_mode.get() == "planes":
                self._update_plane_position(ax, idx)

    def _reset_3d_camera(self):
        if not HAS_VTK or self._vtk_image is None:
            return
        self._vtk_renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def _on_3d_view_preset_picked(self, index: int):
        """Apply the chosen preset, then snap the combo back to the
        placeholder so the same view can be picked again."""
        preset = self._3d_view_combo.itemData(index)
        if preset is None:
            return
        self._set_3d_view_preset(preset)
        self._3d_view_combo.blockSignals(True)
        self._3d_view_combo.setCurrentIndex(0)
        self._3d_view_combo.blockSignals(False)

    def _set_3d_view_preset(self, preset: str):
        """Aim the camera at a canonical anatomical view.

        Voxel index axes are (X=sagittal, Y=coronal, Z=axial), so in world
        space patient-left is +X, anterior +Y and superior +Z. The preset
        gives a unit direction from focal point to camera; ``ResetCamera``
        fits the distance.
        """
        if not HAS_VTK:
            return
        ren = self._vtk_renderer
        bounds = ren.ComputeVisiblePropBounds()
        # Empty scene, ComputeVisiblePropBounds returns an inverted range.
        if bounds[0] > bounds[1]:
            return
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])

        # (direction-from-focal-point-to-camera, view-up)
        presets = {
            "superior":  (( 0,  0,  1), (0,  1, 0)),
            "inferior":  (( 0,  0, -1), (0,  1, 0)),
            "anterior":  (( 0, -1,  0), (0,  0, 1)),
            "posterior": (( 0,  1,  0), (0,  0, 1)),
            "left":      ((-1,  0,  0), (0,  0, 1)),
            "right":     (( 1,  0,  0), (0,  0, 1)),
            "iso":       (( 1, -1,  1), (0,  0, 1)),
        }
        spec = presets.get(preset)
        if spec is None:
            return
        direction, view_up = spec

        diag = (
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        ) ** 0.5
        d = max(diag, 1.0)

        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx + direction[0] * d,
                        cy + direction[1] * d,
                        cz + direction[2] * d)
        cam.SetViewUp(*view_up)
        ren.ResetCamera()
        ren.ResetCameraClippingRange()
        self._vtk_widget.GetRenderWindow().Render()

    # render

    def _pick_downsample(self, shape) -> int:
        """Pick an isotropic stride so the GPU copy fits the free RAM budget.

        Returns 1 (no downsample) when there is headroom or when psutil is
        unavailable."""
        nx, ny, nz = shape
        voxels = float(nx) * float(ny) * float(nz)
        # float32 scalars on the GPU side
        est_full_mb = voxels * 4.0 / (1024.0 ** 2)
        free_mb = available_ram_mb()
        if free_mb is None or est_full_mb <= 0.35 * free_mb:
            return 1
        budget_mb = max(256.0, 0.25 * free_mb)
        ds = 1
        while est_full_mb / (ds ** 3) > budget_mb and ds < 8:
            ds += 1
        return ds

    def _get_3d_array(self):
        """Return the 3-D array to upload, downsampled when RAM is tight.

        For :class:`LazyGrayVolume` the strided read comes from the
        mmap-backed proxy, avoiding a float32 copy of a multi-GiB file.
        """
        if self._gray is None:
            return None, 1
        shape = self._gray.shape

        ds = self._pick_downsample(shape)

        if isinstance(self._gray, LazyGrayVolume):
            if ds == 1:
                self._set_status("Materialising volume for 3-D...", busy=True)
                arr = self._get_gray()
                return arr, 1
            self._set_status(
                f"Large volume - loading at 1/{ds} for 3-D...", busy=True,
            )
            arr = np.asarray(self._gray[::ds, ::ds, ::ds], dtype=np.float32)
            return arr, ds

        arr = self._gray
        if arr is None:
            return None, 1
        if ds == 1:
            return arr, 1
        self._set_status(
            f"Large volume - downsampling 1/{ds} for 3-D...", busy=True,
        )
        return np.ascontiguousarray(arr[::ds, ::ds, ::ds]), ds

    def _do_3d_render(self):
        if not HAS_VTK or not self._require_img():
            return

        try:
            arr, ds = self._get_3d_array()
            if arr is None:
                return

            if id(arr) != self._vtk_built_for_id or arr.shape != self._vtk_built_shape:
                self._set_status("Uploading volume to GPU...", busy=True)
                self._upload_volume(arr, downsample=ds)
                self._init_3d_sliders(arr.shape)

            mode = self._3d_mode.get()
            if mode == "volume":
                self._build_volume_actor(arr)
            else:
                self._build_plane_actors(arr)

            if not self._vtk_first_render_done:
                self._vtk_renderer.ResetCamera()
                self._vtk_first_render_done = True
            self._vtk_widget.GetRenderWindow().Render()
            self._set_status("3-D ready.", busy=False)
        except Exception as ex:
            self._append_log(f"  3-D render error: {ex}", 'err')
            self._set_status("3-D render error.", busy=False)

    # VTK helpers

    def _upload_volume(self, arr, downsample: int = 1):
        import vtk
        from vtkmodules.util import numpy_support

        nx, ny, nz = arr.shape
        spacing = self._img.header.get_zooms() if self._img is not None else (1, 1, 1)
        # Scale spacing by the downsample factor so the cube-axes bounds stay
        # in original world units (voxel index * original spacing).
        sx = float(spacing[0]) * float(downsample)
        sy = float(spacing[1]) * float(downsample)
        sz = float(spacing[2]) * float(downsample)

        # Single float32 Fortran-ordered buffer handed to VTK with
        # deep=False, saving a 4*nx*ny*nz byte copy. The Python reference is
        # kept so the buffer outlives the VTK array.
        flat = np.asarray(arr, dtype=np.float32, order='F').ravel(order='F')
        self._vtk_flat_buffer = flat

        vtk_arr = numpy_support.numpy_to_vtk(
            flat, deep=False, array_type=vtk.VTK_FLOAT,
        )
        img = vtk.vtkImageData()
        img.SetDimensions(nx, ny, nz)
        img.SetSpacing(sx, sy, sz)
        img.GetPointData().SetScalars(vtk_arr)

        self._vtk_image = img
        self._vtk_built_for_id = id(arr)
        self._vtk_built_shape = arr.shape

        self._vtk_renderer.RemoveAllViewProps()
        self._vtk_volume = None
        self._vtk_plane_actors = []
        self._vtk_plane_borders = {}
        self._vtk_outline_actor = None
        self._vtk_first_render_done = False

        self._ensure_cube_axes((nx, ny, nz), (sx, sy, sz))
        self._vtk_renderer.AddActor(self._cube_axes)

        # Re-attach STL overlays wiped by RemoveAllViewProps above.
        for entry in getattr(self, '_stl_overlays', ()):
            self._vtk_renderer.AddActor(entry["actor"])

    def _volume_stats(self, arr):
        """Cached (auto_lo, auto_hi, min, max) for the uploaded array.

        Sampled and cached per array; a full min/max scan per render costs
        seconds on a multi-GiB volume.
        """
        key = (id(arr), tuple(arr.shape))
        cached = getattr(self, '_vtk_stats', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        ww, wc = auto_window(arr)
        flat = np.asarray(arr).reshape(-1)
        sample = flat[:: max(1, flat.size // 500_000)]
        stats = (wc - ww / 2.0, wc + ww / 2.0,
                 float(sample.min()), float(sample.max()))
        self._vtk_stats = (key, stats)
        return stats

    def _window_bounds(self, arr):
        """Display window: the user's W/C when set, else the automatic one.

        Shares :func:`auto_window` with the 2-D viewer so a volume looks the
        same in both and a min-max range cannot flatten the specimen.
        """
        p_lo, p_hi, amin, amax = self._volume_stats(arr)
        ww, wc = self._ww, self._wc
        if ww is None or wc is None:
            lo, hi = p_lo, p_hi
        else:
            lo, hi = wc - ww / 2.0, wc + ww / 2.0
            # A stale window from another file maps every voxel to one end of
            # the ramp, hiding the volume.
            if hi <= amin or lo >= amax:
                lo, hi = p_lo, p_hi
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _setup_3d_lighting(self):
        """Key plus fill camera light, for readable shading under rotation."""
        import vtk
        self._vtk_renderer.RemoveAllLights()
        for position, intensity in (((0.6, 0.6, 1.0), 1.0),
                                    ((-0.7, -0.2, 0.6), 0.45)):
            light = vtk.vtkLight()
            light.SetLightTypeToCameraLight()
            light.SetPosition(*position)
            light.SetFocalPoint(0.0, 0.0, 0.0)
            light.SetIntensity(intensity)
            self._vtk_renderer.AddLight(light)

    def _on_3d_interaction_end(self, *_args):
        """Re-render at full sampling once the camera stops moving."""
        if not hasattr(self, '_vtk_widget'):
            return
        try:
            rw = self._vtk_widget.GetRenderWindow()
            rw.SetDesiredUpdateRate(0.001)
            rw.Render()
        except Exception:
            pass

    def _clear_plane_props(self):
        """Remove the slice planes and their outlines from the renderer."""
        for actor in getattr(self, '_vtk_plane_actors', ()):
            self._vtk_renderer.RemoveActor(actor)
        for _src, border in getattr(self, '_vtk_plane_borders', {}).values():
            self._vtk_renderer.RemoveActor(border)
        if getattr(self, '_vtk_outline_actor', None) is not None:
            self._vtk_renderer.RemoveActor(self._vtk_outline_actor)
        self._vtk_plane_actors = []
        self._vtk_plane_borders = {}
        self._vtk_outline_actor = None

    # volume rendering

    def _build_volume_actor(self, arr):
        import vtk
        self._clear_plane_props()

        lo, hi = self._window_bounds(arr)

        if self._vtk_volume is None:
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(self._vtk_image)
            mapper.SetBlendModeToComposite()
            mapper.SetRequestedRenderModeToGPU()

            ctf = vtk.vtkColorTransferFunction()
            otf = vtk.vtkPiecewiseFunction()

            gof = vtk.vtkPiecewiseFunction()

            prop = vtk.vtkVolumeProperty()
            prop.SetColor(ctf)
            prop.SetScalarOpacity(otf)
            prop.SetGradientOpacity(gof)
            prop.SetInterpolationTypeToLinear()
            prop.ShadeOn()
            prop.SetAmbient(0.20)
            prop.SetDiffuse(0.85)
            prop.SetSpecular(0.35)
            prop.SetSpecularPower(24.0)
            # Opacity accumulates per world-distance unit. The default of
            # 1.0 mm is much larger than a CT voxel, which makes an
            # unadjusted volume look like fog.
            spacing = self._vtk_image.GetSpacing()
            prop.SetScalarOpacityUnitDistance(float(min(spacing)))

            vol = vtk.vtkVolume()
            vol.SetMapper(mapper)
            vol.SetProperty(prop)

            self._vtk_volume = vol
            self._vtk_volume_ctf = ctf
            self._vtk_volume_otf = otf
            self._vtk_volume_gof = gof
            self._vtk_renderer.AddVolume(vol)
        else:
            ctf = self._vtk_volume_ctf
            otf = self._vtk_volume_otf
            gof = self._vtk_volume_gof

        span = hi - lo

        # Warm grey ramp, so dense material reads as a solid surface rather
        # than a grey cloud.
        ctf.RemoveAllPoints()
        ctf.AddRGBPoint(lo,                  0.10, 0.10, 0.12)
        ctf.AddRGBPoint(lo + span * 0.35,    0.45, 0.42, 0.38)
        ctf.AddRGBPoint(lo + span * 0.60,    0.82, 0.76, 0.66)
        ctf.AddRGBPoint(lo + span * 0.85,    0.96, 0.93, 0.86)
        ctf.AddRGBPoint(hi,                  1.00, 0.99, 0.95)

        otf.RemoveAllPoints()
        otf.AddPoint(lo,                     0.00)
        otf.AddPoint(lo + span * 0.20,       0.00)
        otf.AddPoint(lo + span * 0.45,       0.08)
        otf.AddPoint(lo + span * 0.70,       0.45)
        otf.AddPoint(hi,                     0.90)

        # Gradient opacity suppresses the homogeneous interior and keeps
        # material boundaries, giving the surface relief.
        gof.RemoveAllPoints()
        gof.AddPoint(0.0,                    0.00)
        gof.AddPoint(span * 0.02,            0.30)
        gof.AddPoint(span * 0.10,            1.00)

    # slice-plane mode

    def _init_3d_sliders(self, shape):
        for i, ax in enumerate(('X', 'Y', 'Z')):
            dim = shape[i]
            mid = dim // 2
            sl = self._3d_sliders[ax]
            sl.blockSignals(True)
            sl.setMinimum(0)
            sl.setMaximum(max(0, dim - 1))
            sl.setValue(mid)
            sl.blockSignals(False)
            self._3d_idx[ax] = mid
            self._3d_idx_widgets[ax].setText(str(mid))

    def _build_plane_actors(self, arr):
        import vtk

        if self._vtk_volume is not None:
            self._vtk_renderer.RemoveVolume(self._vtk_volume)
            self._vtk_volume = None

        lo, hi = self._window_bounds(arr)
        if not self._vtk_plane_actors:
            self._vtk_plane_actors = []
            for axis in ('X', 'Y', 'Z'):
                actor = vtk.vtkImageActor()
                actor.SetInputData(self._vtk_image)
                self._vtk_renderer.AddActor(actor)
                actor.axis_letter = axis
                self._vtk_plane_actors.append(actor)
                self._add_plane_border(axis)
            self._add_volume_outline()

        for actor in self._vtk_plane_actors:
            prop = actor.GetProperty()
            # vtkImageActor maps scalars through its own window/level, which
            # defaults to 255/127.5. CT scalars run to five figures, so
            # without this every voxel saturates and the planes render white.
            prop.SetColorWindow(hi - lo)
            prop.SetColorLevel((hi + lo) / 2.0)
            prop.SetInterpolationTypeToLinear()
            # Slices carry their own intensities, so scene lights must not
            # tint them as the camera moves.
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)

        for axis, actor in zip(('X', 'Y', 'Z'), self._vtk_plane_actors):
            self._update_plane_position(axis, self._3d_idx[axis], render=False)
            actor.SetVisibility(1 if self._3d_plane_visible.get(axis, True) else 0)

    def _add_plane_border(self, axis):
        """Outline one slice plane in its tri-planar axis colour."""
        import vtk
        source = vtk.vtkOutlineSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        border = vtk.vtkActor()
        border.SetMapper(mapper)
        rgb = self._hex_to_rgb(AXIS_COLOR[axis])
        border.GetProperty().SetColor(*rgb)
        border.GetProperty().SetLineWidth(2.0)
        border.GetProperty().SetLighting(False)
        self._vtk_renderer.AddActor(border)
        self._vtk_plane_borders[axis] = (source, border)

    def _add_volume_outline(self):
        """Draw the volume's bounding box so empty space stays legible."""
        import vtk
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(self._vtk_image)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.45, 0.42, 0.38)
        actor.GetProperty().SetLineWidth(1.2)
        actor.GetProperty().SetLighting(False)
        self._vtk_renderer.AddActor(actor)
        self._vtk_outline_actor = actor

    def _on_3d_plane_visibility_toggled(self, axis: str, visible: bool):
        """Show or hide the matching VTK plane actor.

        Only visible in Sliced mode. The state is recorded in either mode so
        a mode switch honours it.
        """
        self._3d_plane_visible[axis] = bool(visible)
        actor = next(
            (a for a in getattr(self, '_vtk_plane_actors', [])
             if getattr(a, 'axis_letter', None) == axis),
            None,
        )
        if actor is None:
            return
        actor.SetVisibility(1 if visible else 0)
        border = getattr(self, '_vtk_plane_borders', {}).get(axis)
        if border is not None:
            border[1].SetVisibility(1 if visible else 0)
        if hasattr(self, '_vtk_widget'):
            self._vtk_widget.GetRenderWindow().Render()

    def _update_plane_position(self, axis, idx, render=True):
        if not self._vtk_plane_actors:
            return
        actor = next(
            (a for a in self._vtk_plane_actors
             if getattr(a, 'axis_letter', None) == axis),
            None,
        )
        if actor is None:
            return
        nx, ny, nz = self._vtk_built_shape
        if axis == 'X':
            actor.SetDisplayExtent(idx, idx, 0, ny - 1, 0, nz - 1)
        elif axis == 'Y':
            actor.SetDisplayExtent(0, nx - 1, idx, idx, 0, nz - 1)
        else:
            actor.SetDisplayExtent(0, nx - 1, 0, ny - 1, idx, idx)
        border = getattr(self, '_vtk_plane_borders', {}).get(axis)
        if border is not None:
            border[0].SetBounds(actor.GetBounds())
            border[1].SetVisibility(actor.GetVisibility())
        if render:
            self._vtk_widget.GetRenderWindow().Render()


# shims

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


class _QLabelVar:
    """QLabel wrapper that exposes ``get()`` / ``set(value)``."""

    def __init__(self, label: QLabel):
        self._lbl = label

    def get(self) -> str:
        return self._lbl.text()

    def set(self, value) -> None:
        self._lbl.setText(str(value))
