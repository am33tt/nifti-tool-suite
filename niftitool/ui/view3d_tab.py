"""3-D viewer (VTK-backed, Slicer-style).

Uses :class:`vtkmodules.tk.vtkTkRenderWidget` to embed a hardware-
accelerated OpenGL render window directly in the Tk notebook tab.  This
is the same rendering stack that 3D Slicer uses, so rotation / zoom /
pan are GPU-driven and stay smooth on hundred-MB volumes.

Two view modes
--------------
* **Volume** — true GPU volume rendering (``vtkSmartVolumeMapper``)
  with a window/level-driven opacity ramp.  This shows the *whole
  object* as a translucent solid, just like Slicer's 3-D view.
* **Slice planes** — three orthogonal textured planes
  (``vtkImageActor``) at the current X / Y / Z slice indices, that the
  user can move with sliders.

The volume's ``vtkImageData`` is built once (after the loader's
background materialisation completes) and re-used across mode
switches and window-level changes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from ..config import (
    ACCENT, BG, BORDER, ENTRY_BG, ERR, FONT_SMALL, FONT_UI, PANEL2,
    SLIDER_DEBOUNCE_MS, TEXT, TEXT_DIM, WARN,
)
from ..core.io import LazyGrayVolume
from ..core.windowing import auto_window
from ..deps import HAS_VTK, np
from .widgets import styled_btn


class View3DMixin:
    """Adds the GPU 3-D View tab to :class:`NiftiApp`."""

    # ── build ────────────────────────────────────────────────────────────────

    def _build_3d_view(self, parent):
        if not HAS_VTK:
            tk.Label(
                parent,
                text="VTK not installed.\n\n"
                     "Install with:  pip install vtk\n\n"
                     "VTK is required for the GPU 3-D viewer (Slicer uses "
                     "the same library).",
                bg=BG, fg=WARN, font=FONT_UI, justify='center',
            ).pack(expand=True, padx=20, pady=20)
            return

        import vtk
        from vtkmodules.tk.vtkTkRenderWidget import vtkTkRenderWidget

        # ── top toolbar ─────────────────────────────────────────────────────
        ctrl = tk.Frame(parent, bg=BG)
        ctrl.pack(fill='x', padx=8, pady=4)

        # Mode
        mode_box = tk.Frame(ctrl, bg=BG); mode_box.pack(side='left', padx=6)
        tk.Label(mode_box, text="Mode", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(anchor='w')
        self._3d_mode = tk.StringVar(value="volume")
        mode_row = tk.Frame(mode_box, bg=BG); mode_row.pack(anchor='w')
        for label, value in (("Volume", "volume"), ("Slice planes", "planes")):
            tk.Radiobutton(
                mode_row, text=label, value=value, variable=self._3d_mode,
                bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
                activebackground=BG, font=FONT_SMALL,
                command=self._on_3d_mode_changed,
            ).pack(side='left')

        styled_btn(ctrl, "▶  Render", self._do_3d_render,
                   accent=True, small=True).pack(side='left', padx=10, pady=(14, 0))
        styled_btn(ctrl, "Reset View", self._reset_3d_camera,
                   small=True).pack(side='left', padx=4, pady=(14, 0))
        styled_btn(ctrl, "Sync to Tri-Planar", self._sync_3d_to_triplanar,
                   small=True).pack(side='left', padx=4, pady=(14, 0))

        # ── slider bar (only used in "planes" mode) ────────────────────────
        sliders = tk.Frame(parent, bg=PANEL2, pady=4); sliders.pack(fill='x')
        self._3d_idx: dict = {'X': 0, 'Y': 0, 'Z': 0}
        self._3d_idx_var: dict = {}
        self._3d_sliders: dict = {}
        labels = {'X': "Sagittal (X)", 'Y': "Coronal (Y)", 'Z': "Axial (Z)"}
        for col, ax in enumerate(('X', 'Y', 'Z')):
            cf = tk.Frame(sliders, bg=PANEL2)
            cf.grid(row=0, column=col, sticky='ew', padx=8, pady=2)
            sliders.columnconfigure(col, weight=1)
            tk.Label(cf, text=labels[ax], bg=PANEL2, fg=ACCENT,
                     font=FONT_SMALL).pack(anchor='w')
            row = tk.Frame(cf, bg=PANEL2); row.pack(fill='x')
            var = tk.StringVar(value="--")
            self._3d_idx_var[ax] = var
            sl = tk.Scale(
                row, from_=0, to=100, orient='horizontal',
                bg=PANEL2, fg=TEXT, troughcolor=BORDER,
                highlightthickness=0, bd=0, showvalue=False,
                command=lambda val, a=ax: self._on_3d_drag(a, val),
            )
            sl.pack(side='left', fill='x', expand=True)
            tk.Label(row, textvariable=var, bg=PANEL2, fg=ACCENT,
                     font=FONT_SMALL, width=5).pack(side='left')
            self._3d_sliders[ax] = sl

        # ── VTK render widget ──────────────────────────────────────────────
        self._vtk_widget = vtkTkRenderWidget(parent, width=900, height=650)
        self._vtk_widget.pack(fill='both', expand=True)

        rw = self._vtk_widget.GetRenderWindow()
        self._vtk_renderer = vtk.vtkRenderer()
        self._vtk_renderer.SetBackground(1.0, 1.0, 1.0)  # white background
        self._vtk_renderer.SetBackground2(0.92, 0.94, 0.98)
        self._vtk_renderer.GradientBackgroundOn()
        rw.AddRenderer(self._vtk_renderer)

        # Per-volume state
        self._vtk_image = None          # vtkImageData currently uploaded
        self._vtk_volume = None         # vtkVolume actor (for "volume" mode)
        self._vtk_plane_actors = []     # list of vtkImageActor (for "planes" mode)
        self._vtk_built_for_id = None   # id() of the numpy array we built from
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
        if hasattr(self, '_nb') and self._nb.select() == str(self._viewer3d_tab):
            self._do_3d_render()

    def maybe_auto_render_3d(self):
        """Called by app.py when the user switches to the 3-D View tab."""
        if not HAS_VTK or self._gray is None:
            return
        if not self._vtk_first_render_done:
            self._do_3d_render()

    # ── slider / mode handlers ─────────────────────────────────────────────

    def _on_3d_mode_changed(self):
        # Mode determines which actor is in the renderer.
        self._do_3d_render()

    def _on_3d_drag(self, axis, val):
        idx = int(float(val))
        self._3d_idx[axis] = idx
        self._3d_idx_var[axis].set(str(idx))
        if self._3d_mode.get() != "planes":
            return
        # Move the corresponding plane actor; this is cheap (no rebuild).
        self._update_plane_position(axis, idx)

    def _sync_3d_to_triplanar(self):
        if self._gray is None:
            return
        for ax in ('X', 'Y', 'Z'):
            idx = int(getattr(self, '_tri_idx', {}).get(ax, 0))
            self._3d_idx[ax] = idx
            self._3d_idx_var[ax].set(str(idx))
            self._3d_sliders[ax].set(idx)
            if self._3d_mode.get() == "planes":
                self._update_plane_position(ax, idx)

    def _reset_3d_camera(self):
        if not HAS_VTK or self._vtk_image is None:
            return
        self._vtk_renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    # ── render ─────────────────────────────────────────────────────────────

    def _do_3d_render(self):
        """Build (or rebuild) the VTK actors and trigger a render.

        All VTK calls happen on the Tk main thread because VTK is not
        thread-safe.  The expensive numpy → vtkImageData copy still
        happens here, but only once per loaded volume; subsequent
        renders just update transfer functions or move plane actors.
        """
        if not HAS_VTK or not self._require_img():
            return

        try:
            # Force materialisation if still lazy — VTK needs a real array.
            if isinstance(self._gray, LazyGrayVolume):
                self._set_status("Materialising volume for 3-D...", busy=True)
                arr = self._get_gray()
            else:
                arr = self._gray
            if arr is None:
                return

            # Rebuild vtkImageData if the underlying ndarray changed.
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
        """Copy the float32 numpy volume into a fresh ``vtkImageData``."""
        import vtk
        from vtkmodules.util import numpy_support

        nx, ny, nz = arr.shape
        spacing = self._img.header.get_zooms() if self._img is not None else (1, 1, 1)
        sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

        # VTK indexes scalars as flat[i + nx*j + nx*ny*k] for voxel (i,j,k).
        # numpy in C-order has flat[i*ny*nz + j*nz + k] which is the wrong
        # order, so we Fortran-ravel (== C-order over the transposed array).
        # This costs one full copy but is unavoidable: VTK keeps its own.
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

        # New volume → drop all previous actors so old data is released.
        self._vtk_renderer.RemoveAllViewProps()
        self._vtk_volume = None
        self._vtk_plane_actors = []
        self._vtk_first_render_done = False

    def _window_bounds(self, arr):
        """Lo/Hi intensity for the colour and opacity transfer functions."""
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
        # Drop any plane actors from a previous mode switch.
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

        # (Re)build the transfer functions for the current window/level.
        ctf.RemoveAllPoints()
        ctf.AddRGBPoint(lo,                       0.0, 0.0, 0.0)
        ctf.AddRGBPoint(lo + (hi - lo) * 0.5,     0.55, 0.55, 0.55)
        ctf.AddRGBPoint(hi,                       1.0, 1.0, 1.0)

        otf.RemoveAllPoints()
        # Lower 25 % of window is invisible (air / background), opacity
        # ramps up smoothly to the upper end — Slicer's "CT-Bone-ish".
        otf.AddPoint(lo,                          0.00)
        otf.AddPoint(lo + (hi - lo) * 0.25,       0.00)
        otf.AddPoint(lo + (hi - lo) * 0.55,       0.30)
        otf.AddPoint(hi,                          0.85)

    # ── slice-plane mode ───────────────────────────────────────────────────

    def _init_3d_sliders(self, shape):
        for i, ax in enumerate(('X', 'Y', 'Z')):
            dim = shape[i]
            mid = dim // 2
            self._3d_sliders[ax].config(from_=0, to=max(0, dim - 1))
            self._3d_sliders[ax].set(mid)
            self._3d_idx[ax] = mid
            self._3d_idx_var[ax].set(str(mid))

    def _build_plane_actors(self, shape):
        """Create or refresh three orthogonal vtkImageActor planes."""
        import vtk

        # Remove the volume actor if we're switching from volume mode.
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
                actor.axis_letter = axis  # ad-hoc tag we read in updates
                self._vtk_plane_actors.append(actor)

        for axis, actor in zip(('X', 'Y', 'Z'), self._vtk_plane_actors):
            self._update_plane_position(axis, self._3d_idx[axis], render=False)

    def _update_plane_position(self, axis, idx, render=True):
        if not self._vtk_plane_actors:
            return
        # Find the actor we tagged for this axis.
        actor = next(
            (a for a in self._vtk_plane_actors
             if getattr(a, 'axis_letter', None) == axis),
            None,
        )
        if actor is None:
            return
        nx, ny, nz = self._vtk_built_shape
        # vtkImageActor.SetDisplayExtent clamps the actor to a 2-D slab.
        if axis == 'X':
            actor.SetDisplayExtent(idx, idx, 0, ny - 1, 0, nz - 1)
        elif axis == 'Y':
            actor.SetDisplayExtent(0, nx - 1, idx, idx, 0, nz - 1)
        else:
            actor.SetDisplayExtent(0, nx - 1, 0, ny - 1, idx, idx)
        if render:
            self._vtk_widget.GetRenderWindow().Render()
