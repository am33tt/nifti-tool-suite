"""3-D viewer (VTK-backed, Slicer-style) — PyQt6 port.

Uses :class:`QVTKRenderWindowInteractor` from ``vtkmodules.qt`` to embed
the same VTK render window that 3D Slicer and ParaView use — no
dependency on Tcl/Tk.  The volume-rendering pipeline itself is
unchanged from the tkinter version.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QRadioButton,
    QSlider, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, ENTRY_BG, PANEL2, TEXT, TEXT_DIM, WARN,
)
from ..core.io import LazyGrayVolume
from ..core.windowing import auto_window
from ..deps import HAS_VTK, np
from .widgets import styled_btn


class View3DMixin:
    """Adds the GPU 3-D View tab to :class:`NiftiApp`."""

    # ── build ────────────────────────────────────────────────────────────────

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

        # ── top toolbar ─────────────────────────────────────────────────────
        ctrl = QWidget(parent)
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(8, 4, 8, 4)
        ctrl_lay.setSpacing(6)

        # Mode
        mode_box = QWidget(ctrl)
        mode_box_lay = QVBoxLayout(mode_box)
        mode_box_lay.setContentsMargins(0, 0, 0, 0)
        mode_box_lay.setSpacing(2)
        mode_lbl = QLabel("Mode", mode_box)
        mode_lbl.setFont(QFont("Segoe UI", 9))
        mode_lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        mode_box_lay.addWidget(mode_lbl)
        mode_row = QWidget(mode_box)
        mr_lay = QHBoxLayout(mode_row)
        mr_lay.setContentsMargins(0, 0, 0, 0); mr_lay.setSpacing(4)

        self._3d_mode_group = QButtonGroup(mode_row)
        self._3d_mode_radios: dict = {}
        for label, value in (("Volume", "volume"), ("Slice planes", "planes")):
            rb = QRadioButton(label, mode_row)
            rb.setFont(QFont("Segoe UI", 9))
            rb.setStyleSheet(f"color: {TEXT}; background-color: transparent;")
            self._3d_mode_group.addButton(rb)
            self._3d_mode_radios[value] = rb
            mr_lay.addWidget(rb)
        self._3d_mode_radios["volume"].setChecked(True)
        self._3d_mode = _RadioVar(self._3d_mode_radios, default="volume")
        for rb in self._3d_mode_radios.values():
            rb.toggled.connect(self._on_3d_mode_toggled)
        mode_box_lay.addWidget(mode_row)
        ctrl_lay.addWidget(mode_box)

        ctrl_lay.addWidget(
            styled_btn(ctrl, "▶  Render", self._do_3d_render,
                       accent=True, small=True)
        )
        ctrl_lay.addWidget(
            styled_btn(ctrl, "Reset View", self._reset_3d_camera, small=True)
        )
        ctrl_lay.addWidget(
            styled_btn(ctrl, "Sync to Tri-Planar",
                       self._sync_3d_to_triplanar, small=True)
        )
        ctrl_lay.addStretch(1)
        root.addWidget(ctrl)

        # ── slider bar (only used in "planes" mode) ────────────────────────
        sliders = QFrame(parent)
        sliders.setStyleSheet(f"background-color: {PANEL2};")
        sgrid = QGridLayout(sliders)
        sgrid.setContentsMargins(8, 4, 8, 4)
        sgrid.setHorizontalSpacing(16)

        self._3d_idx: dict = {'X': 0, 'Y': 0, 'Z': 0}
        self._3d_idx_widgets: dict = {}
        self._3d_sliders: dict = {}
        labels = {'X': "Sagittal (X)", 'Y': "Coronal (Y)", 'Z': "Axial (Z)"}
        for col, ax in enumerate(('X', 'Y', 'Z')):
            cf = QWidget(sliders)
            cf_lay = QVBoxLayout(cf)
            cf_lay.setContentsMargins(0, 0, 0, 0); cf_lay.setSpacing(2)
            title_lbl = QLabel(labels[ax], cf)
            title_lbl.setFont(QFont("Segoe UI", 9))
            title_lbl.setStyleSheet(f"color: {ACCENT}; background-color: {PANEL2};")
            cf_lay.addWidget(title_lbl)

            row_w = QWidget(cf)
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(0, 0, 0, 0); row_lay.setSpacing(4)
            sl = QSlider(Qt.Orientation.Horizontal, row_w)
            sl.setMinimum(0); sl.setMaximum(100)
            row_lay.addWidget(sl, 1)
            idx_lbl = QLabel("--", row_w)
            idx_lbl.setFont(QFont("Segoe UI", 9))
            idx_lbl.setStyleSheet(f"color: {ACCENT}; background-color: {PANEL2};")
            idx_lbl.setFixedWidth(40)
            row_lay.addWidget(idx_lbl)
            cf_lay.addWidget(row_w)

            sgrid.addWidget(cf, 0, col)
            sgrid.setColumnStretch(col, 1)

            sl.valueChanged.connect(
                lambda val, a=ax: self._on_3d_drag(a, val)
            )
            self._3d_sliders[ax] = sl
            self._3d_idx_widgets[ax] = idx_lbl

        root.addWidget(sliders)

        # Tk-compat shim for ``_3d_idx_var[ax].set(...)``
        self._3d_idx_var = {
            ax: _QLabelVar(lbl) for ax, lbl in self._3d_idx_widgets.items()
        }

        # ── VTK render widget ──────────────────────────────────────────────
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
        # Do NOT call ``Start()`` — the Qt event loop drives the interactor.

        # Per-volume state
        self._vtk_image = None
        self._vtk_volume = None
        self._vtk_plane_actors = []
        self._vtk_built_for_id = None
        self._vtk_built_shape = None
        self._vtk_first_render_done = False

        self._3d_redraw_pending = None

    # ── volume materialisation hook (called from ActionsMixin) ─────────────

    def _on_volume_materialised(self):
        """The lazy proxy was promoted to a real ndarray.  Invalidate any
        VTK image data we built from a previous (different) array."""
        self._vtk_built_for_id = None
        self._vtk_built_shape = None
        # If the user is currently looking at the 3-D tab, render now;
        # otherwise wait until they switch to it (handled by app.py).
        if hasattr(self, '_nb') and self._nb.currentWidget() is self._viewer3d_tab:
            self._do_3d_render()

    def maybe_auto_render_3d(self):
        """Called by app.py when the user switches to the 3-D View tab."""
        if not HAS_VTK or self._gray is None:
            return
        if not self._vtk_first_render_done:
            self._do_3d_render()

    # ── slider / mode handlers ─────────────────────────────────────────────

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
        self._3d_idx_widgets[axis].setText(str(idx))
        if self._3d_mode.get() != "planes":
            return
        self._update_plane_position(axis, idx)

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

    # ── render ─────────────────────────────────────────────────────────────

    def _do_3d_render(self):
        if not HAS_VTK or not self._require_img():
            return

        try:
            if isinstance(self._gray, LazyGrayVolume):
                self._set_status("Materialising volume for 3-D...", busy=True)
                arr = self._get_gray()
            else:
                arr = self._gray
            if arr is None:
                return

            if id(arr) != self._vtk_built_for_id or arr.shape != self._vtk_built_shape:
                self._set_status("Uploading volume to GPU...", busy=True)
                self._upload_volume(arr)
                self._init_3d_sliders(arr.shape)

            mode = self._3d_mode.get()
            if mode == "volume":
                self._build_volume_actor(arr)
            else:
                self._build_plane_actors(arr.shape)

            if not self._vtk_first_render_done:
                self._vtk_renderer.ResetCamera()
                self._vtk_first_render_done = True
            self._vtk_widget.GetRenderWindow().Render()
            self._set_status("3-D ready.", busy=False)
        except Exception as ex:
            self._append_log(f"  3-D render error: {ex}", 'err')
            self._set_status("3-D render error.", busy=False)

    # ── VTK helpers ────────────────────────────────────────────────────────

    def _upload_volume(self, arr):
        import vtk
        from vtkmodules.util import numpy_support

        nx, ny, nz = arr.shape
        spacing = self._img.header.get_zooms() if self._img is not None else (1, 1, 1)
        sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

        flat = np.asfortranarray(arr).ravel(order='F')

        vtk_arr = numpy_support.numpy_to_vtk(
            flat, deep=True, array_type=vtk.VTK_FLOAT,
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
        self._vtk_first_render_done = False

    def _window_bounds(self, arr):
        ww, wc = self._ww, self._wc
        if ww is None or wc is None:
            ww, wc = auto_window(arr)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    # ── volume rendering ───────────────────────────────────────────────────

    def _build_volume_actor(self, arr):
        import vtk
        for actor in self._vtk_plane_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._vtk_plane_actors = []

        lo, hi = self._window_bounds(arr)

        if self._vtk_volume is None:
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(self._vtk_image)
            mapper.SetBlendModeToComposite()
            mapper.SetRequestedRenderModeToGPU()

            ctf = vtk.vtkColorTransferFunction()
            otf = vtk.vtkPiecewiseFunction()

            prop = vtk.vtkVolumeProperty()
            prop.SetColor(ctf)
            prop.SetScalarOpacity(otf)
            prop.SetInterpolationTypeToLinear()
            prop.ShadeOn()
            prop.SetAmbient(0.35)
            prop.SetDiffuse(0.7)
            prop.SetSpecular(0.2)

            vol = vtk.vtkVolume()
            vol.SetMapper(mapper)
            vol.SetProperty(prop)

            self._vtk_volume = vol
            self._vtk_volume_ctf = ctf
            self._vtk_volume_otf = otf
            self._vtk_renderer.AddVolume(vol)
        else:
            ctf = self._vtk_volume_ctf
            otf = self._vtk_volume_otf

        ctf.RemoveAllPoints()
        ctf.AddRGBPoint(lo,                       0.0, 0.0, 0.0)
        ctf.AddRGBPoint(lo + (hi - lo) * 0.5,     0.55, 0.55, 0.55)
        ctf.AddRGBPoint(hi,                       1.0, 1.0, 1.0)

        otf.RemoveAllPoints()
        otf.AddPoint(lo,                          0.00)
        otf.AddPoint(lo + (hi - lo) * 0.25,       0.00)
        otf.AddPoint(lo + (hi - lo) * 0.55,       0.30)
        otf.AddPoint(hi,                          0.85)

    # ── slice-plane mode ───────────────────────────────────────────────────

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

    def _build_plane_actors(self, shape):
        import vtk

        if self._vtk_volume is not None:
            self._vtk_renderer.RemoveVolume(self._vtk_volume)
            self._vtk_volume = None

        nx, ny, nz = shape
        if not self._vtk_plane_actors:
            self._vtk_plane_actors = []
            for axis in ('X', 'Y', 'Z'):
                actor = vtk.vtkImageActor()
                actor.SetInputData(self._vtk_image)
                self._vtk_renderer.AddActor(actor)
                actor.axis_letter = axis
                self._vtk_plane_actors.append(actor)

        for axis, actor in zip(('X', 'Y', 'Z'), self._vtk_plane_actors):
            self._update_plane_position(axis, self._3d_idx[axis], render=False)

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
        if render:
            self._vtk_widget.GetRenderWindow().Render()


# ── shims ────────────────────────────────────────────────────────────────────

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
