"""Tri-planar (sagittal, coronal, axial) slice viewer.

Panels are drawn in voxel coordinates but laid out in millimetres, so
anisotropic volumes keep true proportions. Slices are decimated to the
panel size and cached; slider drags are blitted instead of redrawn.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIntValidator
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton, QSizePolicy,
    QSlider, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, AXIS_COLOR, BG, BORDER, ERR, MAX_TRI_DISPLAY_PX, PANEL2,
    SLIDER_DEBOUNCE_MS, TEAL, TEXT, TEXT_DIM,
)
from ..core.windowing import apply_window, auto_window
from ..deps import HAS_MPL, np
from .widgets import style_nav_toolbar, styled_btn


# Hover readout / guide-crosshair update interval (~50 Hz).
TRI_HOVER_INTERVAL = 0.02

# A voxel read slower than this disables live sampling on hover.
TRI_SAMPLE_BUDGET = 0.15

# Panel placement, as figure fractions.
TRI_MARGINS = {'left': 0.02, 'right': 0.98, 'bottom': 0.06, 'top': 0.92,
               'gap': 0.025}


class _AspectLimitNoticeFilter(logging.Filter):
    """Drop matplotlib's note that it re-fitted a panel's view limits.

    The panels pin their own limits (see :meth:`_fit_tri_panels`) while
    holding a fixed data aspect, so whenever matplotlib nudges one it says
    so on its own logger. The nudge is the intended behaviour -- the view is
    being kept undistorted -- but the note is emitted once per panel per
    frame, which floods the console during a window resize or a toolbar
    zoom. Only this one message is dropped; every other matplotlib message
    still comes through.
    """

    PREFIX = "Ignoring fixed"

    def filter(self, record) -> bool:
        return not record.getMessage().startswith(self.PREFIX)


logging.getLogger("matplotlib.axes._base").addFilter(_AspectLimitNoticeFilter())


def _disable_layout_engine(fig):
    """Turn off automatic layout; panel positions are set explicitly.

    Tight layout re-runs on every draw with padding that depends on
    tick-label width, giving uneven gaps between panels.
    """
    try:
        fig.set_layout_engine('none')
    except Exception:
        try:
            fig.set_tight_layout(False)
        except Exception:
            pass


class TriplanarMixin:
    """Adds the Tri-Planar tab to :class:`NiftiApp`."""

    def _build_triplanar(self, parent):
        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not HAS_MPL:
            lbl = QLabel("matplotlib not installed", parent)
            lbl.setStyleSheet(f"color: {ERR}; background-color: {BG};")
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(lbl)
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvasTkAgg,
            NavigationToolbar2QT as NavigationToolbar2Tk,
        )

        self._tri_fig = Figure(facecolor=BG)
        axes = self._tri_fig.subplots(1, 3)
        self._ax_sag, self._ax_cor, self._ax_axi = axes
        for ax in axes:
            ax.set_facecolor(PANEL2)
            ax.axis('off')

        # Welcome banner shown until the first volume is loaded.
        self._tri_welcome = self._tri_fig.text(
            0.5, 0.55, "Welcome!",
            color=ACCENT, fontsize=20, fontweight='bold',
            ha='center', va='center',
        )
        self._tri_welcome_sub = self._tri_fig.text(
            0.5, 0.46,
            "Load a NIfTI file to get started\n"
            "(Ctrl+O, drag and drop, or use \u201cOpen NIfTI\u2026\u201d above).",
            color=TEXT, fontsize=10, ha='center', va='center',
        )

        # Persistent matplotlib artists (created once, updated each frame).
        self._tri_im:      dict = {'X': None, 'Y': None, 'Z': None}
        self._tri_title:   dict = {}
        self._tri_hline:   dict = {}
        self._tri_vline:   dict = {}
        self._tri_compass: dict = {}

        _disable_layout_engine(self._tri_fig)
        self._tri_canvas = FigureCanvasTkAgg(self._tri_fig)
        root.addWidget(self._tri_canvas, 1)
        self._tri_canvas.mpl_connect('button_press_event', self._on_tri_press)
        self._tri_canvas.mpl_connect('motion_notify_event', self._on_tri_motion)
        self._tri_canvas.mpl_connect('button_release_event', self._on_tri_release)
        self._tri_canvas.mpl_connect('resize_event', self._on_tri_resize)
        self._init_tri_probe()

        self._build_tri_measure_bar(parent, root)

        tb_frame = QWidget(parent)
        tb_lay = QHBoxLayout(tb_frame)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(style_nav_toolbar(
            NavigationToolbar2Tk(self._tri_canvas, tb_frame)))
        tb_lay.addStretch(1)

        self._tri_same_scale_check = QCheckBox("Same scale in all views",
                                               tb_frame)
        # Off by default. A common window is harmless on a roughly cubic
        # volume but shrinks the axial view on a tall specimen.
        self._tri_same_scale_check.setChecked(False)
        self._tri_same_scale_check.setFont(QFont("Segoe UI", 9))
        self._tri_same_scale_check.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;")
        self._tri_same_scale_check.toggled.connect(
            self._on_tri_scale_mode_changed)
        tb_lay.addWidget(self._tri_same_scale_check)

        tb_lay.addWidget(styled_btn(tb_frame, "Fit views",
                                    self._on_tri_scale_mode_changed,
                                    small=True))
        root.addWidget(tb_frame)

        # Slider bar
        slider_bar = QFrame(parent)
        slider_bar.setStyleSheet(f"background-color: {PANEL2};")
        sb_grid = QGridLayout(slider_bar)
        sb_grid.setContentsMargins(8, 4, 8, 4)
        sb_grid.setHorizontalSpacing(16)

        self._tri_sliders: dict = {}
        self._tri_idx_widgets: dict = {}
        self._tri_idx = {'X': 0, 'Y': 0, 'Z': 0}
        axis_info = [
            ('X', "Sagittal (X)"),
            ('Y', "Coronal  (Y)"),
            ('Z', "Axial    (Z)"),
        ]
        for col, (ax, label) in enumerate(axis_info):
            ax_col = AXIS_COLOR[ax]
            col_frame = QWidget(slider_bar)
            col_lay = QVBoxLayout(col_frame)
            col_lay.setContentsMargins(0, 0, 0, 0)
            col_lay.setSpacing(2)

            title_lbl = QLabel(label, col_frame)
            title_lbl.setFont(QFont("Segoe UI", 9))
            title_lbl.setStyleSheet(
                f"color: {ax_col}; background-color: {PANEL2};"
            )
            col_lay.addWidget(title_lbl)

            row_w = QWidget(col_frame)
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(4)
            sl = QSlider(Qt.Orientation.Horizontal, row_w)
            sl.setMinimum(0); sl.setMaximum(100)
            row_lay.addWidget(sl, 1)
            idx_edit = QLineEdit("--", row_w)
            idx_edit.setFont(QFont("Consolas", 10))
            idx_edit.setStyleSheet(
                f"color: {ax_col}; background-color: {PANEL2}; "
                f"border: 1px solid {BORDER}; padding: 1px 3px;"
            )
            idx_edit.setFixedWidth(50)
            idx_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            idx_edit.setValidator(QIntValidator(0, 10_000, idx_edit))
            row_lay.addWidget(idx_edit)
            col_lay.addWidget(row_w)

            sb_grid.addWidget(col_frame, 0, col)
            sb_grid.setColumnStretch(col, 1)

            sl.valueChanged.connect(
                lambda val, a=ax: self._on_tri_drag(a, val)
            )
            idx_edit.editingFinished.connect(
                lambda a=ax, e=idx_edit: self._on_tri_idx_entered(a, e)
            )
            self._tri_sliders[ax] = sl
            self._tri_idx_widgets[ax] = idx_edit

        root.addWidget(slider_bar)

        # Export buttons
        export_bar = QWidget(parent)
        eb_lay = QHBoxLayout(export_bar)
        eb_lay.setContentsMargins(6, 2, 6, 2)
        for axis in ('X', 'Y', 'Z'):
            name = {'Z': 'Axial', 'Y': 'Coronal', 'X': 'Sagittal'}[axis]
            eb_lay.addWidget(
                styled_btn(export_bar, f"Export {name} PNG",
                           lambda a=axis: self._export_slice(a), small=True)
            )
        eb_lay.addStretch(1)
        root.addWidget(export_bar)

        # Slider drags take the blit fast path; cmap, window and load take
        # the full refresh.
        self._tri_redraw_pending = None
        self._tri_debounce_timer = QTimer(self)
        self._tri_debounce_timer.setSingleShot(True)
        self._tri_debounce_timer.timeout.connect(self._tri_fast_refresh)

        # Blit state per axes: bg0 is the panel without image or crosshairs
        # (ticks, spines, compass, title, measurements), bg1 is bg0 plus the
        # current slice image. A drag tick restores bg1 for unchanged panels
        # and redraws one image and six lines.
        self._tri_bg0: dict = {}
        self._tri_bg1: dict = {}
        self._tri_bg_valid = False
        self._tri_canvas.mpl_connect('draw_event', self._on_tri_full_draw)

        # One normal draw after the drag settles, to refresh titles and
        # measurement overlays.
        self._tri_sync_timer = QTimer(self)
        self._tri_sync_timer.setSingleShot(True)
        self._tri_sync_timer.timeout.connect(self._tri_settle_refresh)

        # Tk-compat shim for ``_tri_idx_vars['X'].set(...)`` usage elsewhere.
        self._tri_idx_vars = {
            ax: _QLabelVar(lbl) for ax, lbl in self._tri_idx_widgets.items()
        }

    # Slider and click handlers

    def _on_tri_drag(self, axis, val):
        idx = int(val)
        self._tri_idx[axis] = idx
        w = self._tri_idx_widgets[axis]
        w.blockSignals(True)
        w.setText(str(idx))
        w.blockSignals(False)
        if self._gray is None:
            return
        self._tri_debounce_timer.start(self._tri_debounce_ms())
        # Warm the cache around the new position for smooth dragging.
        self._prefetch_tri_neighbors(axis, idx)

    def _on_tri_idx_entered(self, axis, edit):
        """Commit a typed slice index (Enter or focus-out).

        The value is clamped to the slider range. Setting the slider re-emits
        valueChanged, which refreshes the canvas via the debounce timer.
        """
        txt = edit.text().strip()
        if not txt:
            edit.setText(str(self._tri_idx[axis]))
            return
        try:
            idx = int(txt)
        except ValueError:
            edit.setText(str(self._tri_idx[axis]))
            return
        sl = self._tri_sliders[axis]
        idx = max(sl.minimum(), min(idx, sl.maximum()))
        edit.setText(str(idx))
        if sl.value() == idx:
            return
        sl.setValue(idx)

    def _on_tri_press(self, event):
        if self._gray is None or event.inaxes is None:
            return
        ax_map = {self._ax_sag: 'X', self._ax_cor: 'Y', self._ax_axi: 'Z'}
        if event.inaxes not in ax_map:
            return
        axis = ax_map[event.inaxes]
        # Measurement tools react to the left button only.
        if event.button != 1:
            return
        tool = getattr(self, '_tri_tool', 'pan')
        if tool == 'distance':
            self._on_tri_distance_click(axis, event)
        elif tool == 'roi':
            self._on_tri_roi_press(axis, event)
        elif tool == 'profile':
            self._on_tri_profile_click(axis, event)
        else:
            self._on_tri_probe_click(axis, event)

    def _on_tri_motion(self, event):
        if getattr(self, '_tri_tool', 'pan') != 'roi':
            return
        if self._roi_drag_axis is None or event.inaxes is None:
            return
        ax_map = {self._ax_sag: 'X', self._ax_cor: 'Y', self._ax_axi: 'Z'}
        if ax_map.get(event.inaxes) != self._roi_drag_axis:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._update_tri_roi_rect(event.xdata, event.ydata, provisional=True)

    def _on_tri_release(self, event):
        if getattr(self, '_tri_tool', 'pan') != 'roi':
            return
        if self._roi_drag_axis is None:
            return
        if event.inaxes is not None and event.xdata is not None:
            self._update_tri_roi_rect(event.xdata, event.ydata, provisional=False)
        self._finalize_tri_roi()

    def _on_tri_probe_click(self, axis, event):
        ex, ey = int(event.xdata or 0), int(event.ydata or 0)
        g = self._gray
        xi, yi, zi = self._tri_idx['X'], self._tri_idx['Y'], self._tri_idx['Z']
        try:
            if axis == 'Z':
                ix, iy, iz = ex, ey, zi
            elif axis == 'Y':
                ix, iy, iz = ex, yi, ey
            else:
                ix, iy, iz = xi, ex, ey
            ix = max(0, min(ix, g.shape[0] - 1))
            iy = max(0, min(iy, g.shape[1] - 1))
            iz = max(0, min(iz, g.shape[2] - 1))
            raw_val = float(g[ix, iy, iz])
            probe = f"voxel ({ix},{iy},{iz})  raw={raw_val:.1f}"
            self._probe_var.set(probe)
        except Exception:
            pass

    # Measurement tools

    def _build_tri_measure_bar(self, parent, root):
        """Tool selector + status line for distance/ROI measurements."""
        bar = QFrame(parent)
        bar.setStyleSheet(f"background-color: {PANEL2};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.setSpacing(10)

        lbl = QLabel("Measure:", bar)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: transparent;")
        lay.addWidget(lbl)

        self._tri_tool = 'pan'
        self._tri_tool_group = QButtonGroup(bar)
        rb_style = (
            f"QRadioButton {{ color: {TEXT_DIM}; background-color: transparent; }}"
            f"QRadioButton:checked {{ color: {ACCENT}; font-weight: bold; }}"
        )
        self._tri_tool_radios = {}
        for label, value in (
            ("Pan", 'pan'),
            ("Distance", 'distance'),
            ("ROI", 'roi'),
            ("Profile", 'profile'),
        ):
            rb = QRadioButton(label, bar)
            rb.setFont(QFont("Segoe UI", 9))
            rb.setStyleSheet(rb_style)
            if value == 'pan':
                rb.setChecked(True)
            rb.toggled.connect(
                lambda checked, v=value: checked and self._on_tri_tool_changed(v)
            )
            self._tri_tool_group.addButton(rb)
            self._tri_tool_radios[value] = rb
            lay.addWidget(rb)

        clear_btn = QPushButton("Clear", bar)
        clear_btn.setFont(QFont("Segoe UI", 9))
        clear_btn.setStyleSheet(
            f"QPushButton {{ color: {TEXT}; background-color: {PANEL2}; "
            f"border: 1px solid {BORDER}; padding: 2px 10px; }}"
            f"QPushButton:hover {{ border: 1px solid {ACCENT}; }}"
        )
        clear_btn.clicked.connect(self._clear_tri_measurements)
        lay.addWidget(clear_btn)

        self._tri_measure_status = QLabel("", bar)
        self._tri_measure_status.setFont(QFont("Consolas", 9))
        self._tri_measure_status.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        self._tri_measure_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        lay.addWidget(self._tri_measure_status, 1)

        root.addWidget(bar)

        # Per-axis artist caches
        self._dist_points: dict = {'X': [], 'Y': [], 'Z': []}
        self._dist_artists: dict = {'X': [], 'Y': [], 'Z': []}
        self._roi_drag_start = None
        self._roi_drag_axis = None
        self._roi_rect_artist: dict = {'X': None, 'Y': None, 'Z': None}
        self._roi_bounds: dict = {'X': None, 'Y': None, 'Z': None}
        self._profile_points: dict = {'X': [], 'Y': [], 'Z': []}
        self._profile_artists: dict = {'X': [], 'Y': [], 'Z': []}
        self._profile_dialogs: list = []

    def _on_tri_tool_changed(self, tool):
        self._tri_tool = tool
        if tool != 'roi' and self._roi_drag_axis is not None:
            self._roi_drag_axis = None
            self._roi_drag_start = None
        hints = {
            'pan': "",
            'distance': "Distance: click two points on any panel.",
            'roi': "ROI: drag a rectangle on any panel.",
            'profile': "Profile: click two points; a line plot will open.",
        }
        self._tri_measure_status.setText(hints.get(tool, ""))

    def _panel_spacings(self, axis):
        """Return (h_mm, v_mm) physical spacing of the panel `axis`.

        Panels display transposed slices:
            'X' (sagittal): (h, v) = (y, z)
            'Y' (coronal):  (h, v) = (x, z)
            'Z' (axial):    (h, v) = (x, y)
        """
        if self._img is None:
            return 1.0, 1.0
        try:
            dx, dy, dz = self._img.header.get_zooms()[:3]
        except Exception:
            return 1.0, 1.0
        if axis == 'X':
            return float(dy), float(dz)
        if axis == 'Y':
            return float(dx), float(dz)
        return float(dx), float(dy)

    def _panel_ax_obj(self, axis):
        return {'X': self._ax_sag, 'Y': self._ax_cor, 'Z': self._ax_axi}[axis]

    # Distance tool

    def _on_tri_distance_click(self, axis, event):
        pts = self._dist_points[axis]
        if len(pts) >= 2:
            self._clear_axis_distance(axis)
            pts = self._dist_points[axis]
        pts.append((float(event.xdata), float(event.ydata)))
        ax_obj = self._panel_ax_obj(axis)
        marker = ax_obj.plot(
            event.xdata, event.ydata, marker='o', markersize=5,
            markerfacecolor=ACCENT, markeredgecolor='white', markeredgewidth=0.8,
        )[0]
        self._dist_artists[axis].append(marker)

        if len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            line = ax_obj.plot(
                [x0, x1], [y0, y1], color=ACCENT, lw=1.6, alpha=0.95,
            )[0]
            self._dist_artists[axis].append(line)

            h_mm, v_mm = self._panel_spacings(axis)
            dh = (x1 - x0) * h_mm
            dv = (y1 - y0) * v_mm
            dist_mm = float(np.hypot(dh, dv))
            label = f"{dist_mm:.2f} mm"
            txt = ax_obj.text(
                (x0 + x1) / 2.0, (y0 + y1) / 2.0 + 2.0, label,
                color=ACCENT, fontsize=8, fontweight='bold', ha='center',
                va='bottom',
                bbox=dict(
                    facecolor=BG, edgecolor=ACCENT, boxstyle='round,pad=0.2',
                    alpha=0.85,
                ),
            )
            self._dist_artists[axis].append(txt)
            panel_name = {'X': 'Sagittal', 'Y': 'Coronal', 'Z': 'Axial'}[axis]
            self._tri_measure_status.setText(
                f"{panel_name} distance: {dist_mm:.3f} mm  "
                f"(Δh={dh:.2f}, Δv={dv:.2f})"
            )
        self._tri_canvas.draw_idle()

    def _clear_axis_distance(self, axis):
        for art in self._dist_artists[axis]:
            try:
                art.remove()
            except Exception:
                pass
        self._dist_artists[axis] = []
        self._dist_points[axis] = []

    # Profile tool

    def _on_tri_profile_click(self, axis, event):
        pts = self._profile_points[axis]
        if len(pts) >= 2:
            self._clear_axis_profile(axis)
            pts = self._profile_points[axis]
        pts.append((float(event.xdata), float(event.ydata)))

        ax_obj = self._panel_ax_obj(axis)
        marker = ax_obj.plot(
            event.xdata, event.ydata, marker='s', markersize=5,
            markerfacecolor=TEAL, markeredgecolor='white', markeredgewidth=0.8,
        )[0]
        self._profile_artists[axis].append(marker)

        if len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            line = ax_obj.plot(
                [x0, x1], [y0, y1], color=TEAL, lw=1.6, alpha=0.95, linestyle='-',
            )[0]
            self._profile_artists[axis].append(line)
            self._open_profile_plot(axis, pts)

        self._tri_canvas.draw_idle()

    def _clear_axis_profile(self, axis):
        for art in self._profile_artists[axis]:
            try:
                art.remove()
            except Exception:
                pass
        self._profile_artists[axis] = []
        self._profile_points[axis] = []

    def _sample_profile(self, axis, pts):
        """Sample raw (and optionally HU) intensity along a line in panel
        coordinates. Returns (d_mm, raw, hu_or_None, n_samples)."""
        g = self._gray
        (x0, y0), (x1, y1) = pts
        h_mm, v_mm = self._panel_spacings(axis)
        dh = (x1 - x0) * h_mm
        dv = (y1 - y0) * v_mm
        length_mm = float(np.hypot(dh, dv))
        n = max(2, int(np.hypot(x1 - x0, y1 - y0)) + 1)
        n = min(n, 2048)
        h_t = np.linspace(x0, x1, n).astype(np.float32)
        v_t = np.linspace(y0, y1, n).astype(np.float32)

        xi, yi, zi = self._tri_idx['X'], self._tri_idx['Y'], self._tri_idx['Z']
        if axis == 'Z':
            slice2d = np.asarray(g[:, :, zi], dtype=np.float32)
        elif axis == 'Y':
            slice2d = np.asarray(g[:, yi, :], dtype=np.float32)
        else:
            slice2d = np.asarray(g[xi, :, :], dtype=np.float32)

        from ..deps import ndimage
        if ndimage is not None:
            raw_vals = ndimage.map_coordinates(
                slice2d, [h_t, v_t], order=1, mode='nearest',
            )
        else:
            hi = np.clip(np.round(h_t).astype(int), 0, slice2d.shape[0] - 1)
            vi = np.clip(np.round(v_t).astype(int), 0, slice2d.shape[1] - 1)
            raw_vals = slice2d[hi, vi]

        d_mm = np.linspace(0.0, length_mm, n, dtype=np.float32)
        return d_mm, raw_vals, n

    def _open_profile_plot(self, axis, pts):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
        )
        from PyQt6.QtWidgets import QDialog, QVBoxLayout as _QVBoxLayout

        d_mm, raw, n = self._sample_profile(axis, pts)
        panel_name = {'X': 'Sagittal', 'Y': 'Coronal', 'Z': 'Axial'}[axis]

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Intensity profile: {panel_name}")
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.resize(640, 380)
        dlg.setStyleSheet(f"background-color: {BG}; color: {TEXT};")

        lay = _QVBoxLayout(dlg)
        lay.setContentsMargins(8, 8, 8, 8)
        fig = Figure(figsize=(6, 3.4), facecolor=BG)
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL2)
        ax.plot(d_mm, raw, color=TEAL, lw=1.4, label='grey value')
        ax.set_xlabel("distance (mm)", color=TEXT_DIM, fontsize=9)
        ax.set_ylabel("grey value", color=TEAL, fontsize=9)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color(BORDER)


        length_mm = float(d_mm[-1]) if len(d_mm) else 0.0
        ax.set_title(
            f"{panel_name}   length = {length_mm:.2f} mm   samples = {n}",
            color=TEXT, fontsize=10, pad=4,
        )
        fig.tight_layout(pad=0.6)
        canvas.draw_idle()

        self._profile_dialogs.append(dlg)
        dlg.finished.connect(lambda _r, d=dlg: self._profile_dialogs.remove(d)
                             if d in self._profile_dialogs else None)
        dlg.show()

        self._tri_measure_status.setText(
            f"{panel_name} profile: length={length_mm:.2f} mm  samples={n}"
        )

    # ROI tool

    def _on_tri_roi_press(self, axis, event):
        if event.xdata is None or event.ydata is None:
            return
        self._clear_axis_roi(axis)
        self._roi_drag_axis = axis
        self._roi_drag_start = (float(event.xdata), float(event.ydata))
        from matplotlib.patches import Rectangle
        ax_obj = self._panel_ax_obj(axis)
        rect = Rectangle(
            (event.xdata, event.ydata), 0, 0,
            linewidth=1.4, edgecolor=ACCENT, facecolor=ACCENT, alpha=0.15,
        )
        ax_obj.add_patch(rect)
        self._roi_rect_artist[axis] = rect
        self._tri_canvas.draw_idle()

    def _update_tri_roi_rect(self, xdata, ydata, provisional):
        axis = self._roi_drag_axis
        if axis is None or self._roi_drag_start is None:
            return
        rect = self._roi_rect_artist[axis]
        if rect is None:
            return
        x0, y0 = self._roi_drag_start
        x1, y1 = float(xdata), float(ydata)
        rect.set_xy((min(x0, x1), min(y0, y1)))
        rect.set_width(abs(x1 - x0))
        rect.set_height(abs(y1 - y0))
        self._roi_bounds[axis] = (x0, y0, x1, y1)
        if provisional:
            # Cap rubber-band redraws; each one repaints the whole figure.
            now = time.perf_counter()
            if now - getattr(self, '_roi_draw_t', 0.0) < 0.033:
                return
            self._roi_draw_t = now
            self._tri_canvas.draw_idle()

    def _finalize_tri_roi(self):
        axis = self._roi_drag_axis
        self._roi_drag_axis = None
        self._roi_drag_start = None
        if axis is None:
            return
        bounds = self._roi_bounds.get(axis)
        if bounds is None:
            return
        x0, y0, x1, y1 = bounds
        if abs(x1 - x0) < 1.0 or abs(y1 - y0) < 1.0:
            self._clear_axis_roi(axis)
            self._tri_measure_status.setText("ROI too small; drag a larger rectangle.")
            self._tri_canvas.draw_idle()
            return

        g = self._gray
        if g is None:
            return
        h0, h1 = sorted((int(round(x0)), int(round(x1))))
        v0, v1 = sorted((int(round(y0)), int(round(y1))))
        h0 = max(0, h0); v0 = max(0, v0)

        xi, yi, zi = self._tri_idx['X'], self._tri_idx['Y'], self._tri_idx['Z']
        try:
            if axis == 'Z':
                h1 = min(h1, g.shape[0] - 1)
                v1 = min(v1, g.shape[1] - 1)
                block = np.asarray(g[h0:h1 + 1, v0:v1 + 1, zi], dtype=np.float32)
            elif axis == 'Y':
                h1 = min(h1, g.shape[0] - 1)
                v1 = min(v1, g.shape[2] - 1)
                block = np.asarray(g[h0:h1 + 1, yi, v0:v1 + 1], dtype=np.float32)
            else:
                h1 = min(h1, g.shape[1] - 1)
                v1 = min(v1, g.shape[2] - 1)
                block = np.asarray(g[xi, h0:h1 + 1, v0:v1 + 1], dtype=np.float32)
        except Exception:
            self._tri_measure_status.setText("ROI sampling failed.")
            return

        if block.size == 0:
            self._clear_axis_roi(axis)
            return

        h_mm, v_mm = self._panel_spacings(axis)
        area_mm2 = (h1 - h0 + 1) * (v1 - v0 + 1) * h_mm * v_mm
        mean = float(block.mean())
        std  = float(block.std())
        mn   = float(block.min())
        mx   = float(block.max())

        ax_obj = self._panel_ax_obj(axis)
        rect = self._roi_rect_artist[axis]
        lx = rect.get_x() + rect.get_width() / 2.0
        ly = rect.get_y() + rect.get_height() + 1.5
        label_txt = ax_obj.text(
            lx, ly,
            f"μ={mean:.1f} σ={std:.1f}\n{area_mm2:.1f} mm²",
            color=ACCENT, fontsize=7, ha='center', va='bottom', fontweight='bold',
            bbox=dict(
                facecolor=BG, edgecolor=ACCENT, boxstyle='round,pad=0.2', alpha=0.85,
            ),
        )
        self._roi_rect_artist[axis] = (rect, label_txt)

        panel_name = {'X': 'Sagittal', 'Y': 'Coronal', 'Z': 'Axial'}[axis]
        extra = ""
        self._tri_measure_status.setText(
            f"{panel_name} ROI [{h0}:{h1+1}, {v0}:{v1+1}]  "
            f"μ={mean:.2f}  σ={std:.2f}  min={mn:.1f}  max={mx:.1f}  "
            f"area={area_mm2:.2f} mm²{extra}"
        )
        self._tri_canvas.draw_idle()

    def _clear_axis_roi(self, axis):
        art = self._roi_rect_artist.get(axis)
        if art is None:
            return
        items = art if isinstance(art, tuple) else (art,)
        for a in items:
            try:
                a.remove()
            except Exception:
                pass
        self._roi_rect_artist[axis] = None
        self._roi_bounds[axis] = None

    def _clear_tri_measurements(self):
        for ax in ('X', 'Y', 'Z'):
            self._clear_axis_distance(ax)
            self._clear_axis_roi(ax)
            self._clear_axis_profile(ax)
        self._tri_measure_status.setText("")
        self._tri_canvas.draw_idle()

    # Refresh (hot path)

    def _effective_window(self):
        """Current (ww, wc); the auto window is computed once per volume
        and cached."""
        if self._gray is None:
            return None
        ww, wc = self._ww, self._wc
        if ww is None or wc is None:
            aw = getattr(self, '_auto_wwwc', None)
            if aw is None:
                aw = auto_window(self._gray)
                self._auto_wwwc = aw
            ww, wc = aw
        return ww, wc

    def _prefetch_tri_neighbors(self, axis, idx):
        """Warm the slice cache around *idx* in the background, so continued
        dragging hits pre-windowed slices."""
        if getattr(self, '_tri_prefetch_busy', False):
            return
        g = self._gray
        if g is None:
            return
        win = self._effective_window()
        if win is None:
            return
        ww, wc = win
        dim = {'X': g.shape[0], 'Y': g.shape[1], 'Z': g.shape[2]}[axis]
        self._tri_prefetch_busy = True

        if self._slice_cache.has_preview():
            # Preview reads are memory reads; warming them costs more in
            # thread churn than it saves, and reading the *file* ahead is
            # exactly the I/O the preview exists to avoid.
            self._tri_prefetch_busy = False
            return

        cost = getattr(self, '_tri_refresh_cost_ms', 0.0)
        if cost > 80.0:
            self._tri_prefetch_busy = False
            return
        offsets = (1, -1, 2, -2, 3, -3) if cost < 25.0 else (1, -1)
        max_px = self._tri_display_px()

        def _run():
            try:
                for d in offsets:
                    j = idx + d
                    if 0 <= j < dim:
                        self._slice_cache.get(
                            axis, j, ww, wc, max_px=max_px,
                        )
            except Exception:
                pass
            finally:
                self._tri_prefetch_busy = False

        threading.Thread(target=_run, daemon=True).start()

    def _refresh_triplanar(self):
        self._tri_redraw_pending = None
        if not HAS_MPL or self._gray is None:
            return

        # First volume loaded, retire the welcome banner.
        if getattr(self, '_tri_welcome', None) is not None:
            self._tri_welcome.set_visible(False)
            self._tri_welcome_sub.set_visible(False)

        g = self._gray
        cmap = self._cmap_var.get()
        xi = max(0, min(self._tri_idx['X'], g.shape[0] - 1))
        yi = max(0, min(self._tri_idx['Y'], g.shape[1] - 1))
        zi = max(0, min(self._tri_idx['Z'], g.shape[2] - 1))

        ww, wc = self._effective_window()

        cache = self._slice_cache
        max_px = self._tri_display_px()
        sl_X, from_preview = self._display_slice('X', xi, ww, wc, max_px)
        sl_Y, _ = self._display_slice('Y', yi, ww, wc, max_px)
        sl_Z, _ = self._display_slice('Z', zi, ww, wc, max_px)

        # Decimated arrays keep the original extent, so crosshairs, probe
        # clicks and measurements stay in voxel units.
        nx, ny, nz = g.shape[0], g.shape[1], g.shape[2]
        extents = {
            'X': (-0.5, ny - 0.5, -0.5, nz - 0.5),
            'Y': (-0.5, nx - 0.5, -0.5, nz - 0.5),
            'Z': (-0.5, nx - 0.5, -0.5, ny - 0.5),
        }

        panels = [
            (self._ax_sag, 'X', xi, sl_X, f"Sagittal  X={xi}"),
            (self._ax_cor, 'Y', yi, sl_Y, f"Coronal   Y={yi}"),
            (self._ax_axi, 'Z', zi, sl_Z, f"Axial     Z={zi}"),
        ]
        al = self._axis_labels
        compass_spec = {
            'X': (al['y'][0], al['y'][1], al['z'][0], al['z'][1]),
            'Y': (al['x'][0], al['x'][1], al['z'][0], al['z'][1]),
            'Z': (al['x'][0], al['x'][1], al['y'][0], al['y'][1]),
        }
        crosshair_pos = {
            'X': (yi, zi),
            'Y': (xi, zi),
            'Z': (xi, yi),
        }
        # Which slider drives each panel H-line and V-line. Colors come from
        # AXIS_COLOR so a line matches its slider.
        crosshair_axes = {
            'X': ('Z', 'Y'),  # sagittal: horiz = Z-slider, vert = Y-slider
            'Y': ('Z', 'X'),  # coronal
            'Z': ('Y', 'X'),  # axial
        }

        rebuilt = False
        for ax_obj, axis, idx, sl_win, title in panels:
            im = self._tri_im[axis]
            # Compare the image extent only: the artist holds RGBA, so its
            # array has a trailing channel axis the slice does not.
            if im is None or im.get_array().shape[:2] != sl_win.shape:
                rebuilt = True
                ax_obj.clear()
                ax_obj.set_facecolor(PANEL2)
                # Voxel-index ticks, set once; the hot path leaves them.
                ax_obj.tick_params(
                    colors=TEXT_DIM, labelsize=5, length=2, width=0.4,
                    top=False, right=False, direction='out', pad=1,
                )
                for side in ('top', 'right'):
                    ax_obj.spines[side].set_visible(False)
                for side in ('bottom', 'left'):
                    ax_obj.spines[side].set_color(BORDER)
                    ax_obj.spines[side].set_linewidth(0.5)
                im = ax_obj.imshow(
                    self._tri_rgba(sl_win, cmap), origin='lower',
                    interpolation='nearest', extent=extents[axis],
                )
                # The slice is drawn in voxel indices, so one index step is
                # h_mm across and v_mm up and the display aspect must be
                # v_mm/h_mm. 'datalim' keeps the panel box fixed and pads
                # the view instead, so all three panels stay the same size
                # whatever the specimen proportions.
                h_mm, v_mm = self._panel_spacings(axis)
                if h_mm > 0 and v_mm > 0:
                    ax_obj.set_aspect(v_mm / h_mm, adjustable='datalim',
                                      anchor='C')
                # Animated artists are skipped by normal draws; the blit path
                # and _on_tri_full_draw paint them explicitly.
                im.set_animated(True)
                self._tri_im[axis] = im
                self._tri_title[axis] = ax_obj.set_title(
                    title, color=TEXT_DIM, fontsize=8, pad=3,
                )
                lft, rgt, bot, top_ = compass_spec[axis]
                self._tri_compass[axis] = {
                    'L': ax_obj.text(0.02, 0.5, lft, transform=ax_obj.transAxes,
                                     color=ACCENT, fontsize=7, va='center',
                                     ha='left', fontweight='bold'),
                    'R': ax_obj.text(0.98, 0.5, rgt, transform=ax_obj.transAxes,
                                     color=ACCENT, fontsize=7, va='center',
                                     ha='right', fontweight='bold'),
                    'B': ax_obj.text(0.5, 0.02, bot, transform=ax_obj.transAxes,
                                     color=ACCENT, fontsize=7, ha='center',
                                     va='bottom', fontweight='bold'),
                    'T': ax_obj.text(0.5, 0.98, top_, transform=ax_obj.transAxes,
                                     color=ACCENT, fontsize=7, ha='center',
                                     va='top', fontweight='bold'),
                }
                h_pos, v_pos = crosshair_pos[axis]
                h_axis, v_axis = crosshair_axes[axis]
                self._tri_hline[axis] = ax_obj.axhline(
                    h_pos, color=AXIS_COLOR[h_axis], lw=1.8, alpha=1.0,
                    linestyle='--',
                )
                self._tri_vline[axis] = ax_obj.axvline(
                    v_pos, color=AXIS_COLOR[v_axis], lw=1.8, alpha=1.0,
                    linestyle='--',
                )
                self._tri_hline[axis].set_animated(True)
                self._tri_vline[axis].set_animated(True)
            else:
                im.set_data(self._tri_rgba(sl_win, cmap))
                self._tri_title[axis].set_text(title)
                h_pos, v_pos = crosshair_pos[axis]
                self._tri_hline[axis].set_ydata([h_pos, h_pos])
                self._tri_vline[axis].set_xdata([v_pos, v_pos])
                lft, rgt, bot, top_ = compass_spec[axis]
                c = self._tri_compass[axis]
                c['L'].set_text(lft); c['R'].set_text(rgt)
                c['B'].set_text(bot); c['T'].set_text(top_)

        # What each panel shows, so the fast path can skip unchanged ones.
        self._tri_drawn_key = {
            'X': (xi, ww, wc, max_px, from_preview),
            'Y': (yi, ww, wc, max_px, from_preview),
            'Z': (zi, ww, wc, max_px, from_preview),
        }
        # A rebuild drew from the preview; ask for the finer read if one
        # would actually show more, on a worker as always.
        timer = getattr(self, '_tri_sync_timer', None)
        if from_preview and timer is not None:
            timer.start(300)
        if rebuilt:
            self._layout_tri_panels()
            self._fit_tri_panels()
            for axis in ('X', 'Y', 'Z'):
                self._add_tri_scale_bar(self._panel_ax_obj(axis), axis,
                                        extents[axis])
        self._tri_canvas.draw_idle()

    # Blit fast path (slider drags)

    def _on_tri_full_draw(self, _event):
        """Recapture the clean backgrounds after a full draw, then paint the
        animated artists (images and crosshairs) on top.

        No ``canvas.blit()`` here: the callback runs inside the canvas paint
        cycle, so blitting would request a repaint while painting
        ("Recursive repaint detected"). Drawing into the renderer suffices.
        """
        if self._tri_im.get('X') is None:
            self._tri_bg_valid = False
            return
        self._update_tri_px_budget()
        canvas = self._tri_canvas
        try:
            for axis in ('X', 'Y', 'Z'):
                ax = self._panel_ax_obj(axis)
                self._tri_bg0[axis] = canvas.copy_from_bbox(ax.bbox)
            self._tri_bg1 = {}
            self._tri_bg_valid = True
            self._blit_panels({'X', 'Y', 'Z'}, do_blit=False)
        except Exception:
            self._tri_bg_valid = False

    def _blit_panels(self, changed, do_blit=True, only=None):
        """Composite the animated artists over the cached backgrounds.

        Panels whose image is unchanged reuse bg1 and redraw only their two
        crosshair lines. ``do_blit=False`` is for the draw-event callback,
        where the surrounding paint pass flushes the buffer.
        """
        canvas = self._tri_canvas
        for axis in ('X', 'Y', 'Z'):
            if only is not None and axis not in only:
                continue
            ax = self._panel_ax_obj(axis)
            im = self._tri_im.get(axis)
            if im is None or axis not in self._tri_bg0:
                continue
            if axis in changed or axis not in self._tri_bg1:
                canvas.restore_region(self._tri_bg0[axis])
                ax.draw_artist(im)
                self._tri_bg1[axis] = canvas.copy_from_bbox(ax.bbox)
            else:
                canvas.restore_region(self._tri_bg1[axis])
            hl = self._tri_hline.get(axis)
            vl = self._tri_vline.get(axis)
            if hl is not None:
                ax.draw_artist(hl)
            if vl is not None:
                ax.draw_artist(vl)
            for art in self._tri_hover_artists(axis):
                if art.get_visible():
                    ax.draw_artist(art)
            if do_blit:
                canvas.blit(ax.bbox)

    def _tri_fast_refresh(self):
        """Slider-drag refresh: update one slice image + six crosshair
        lines by blitting instead of re-rendering the whole figure."""
        if not HAS_MPL or self._gray is None:
            return
        if not self._tri_bg_valid or self._tri_im.get('X') is None:
            # No cached backgrounds yet (first draw, resize, reload); the
            # full draw recaptures them.
            self._refresh_triplanar()
            return

        t0 = time.perf_counter()
        g = self._gray
        xi = max(0, min(self._tri_idx['X'], g.shape[0] - 1))
        yi = max(0, min(self._tri_idx['Y'], g.shape[1] - 1))
        zi = max(0, min(self._tri_idx['Z'], g.shape[2] - 1))
        ww, wc = self._effective_window()

        cache = self._slice_cache
        cmap_name = self._cmap_var.get()
        max_px = self._tri_display_px()
        changed = set()
        drawn = getattr(self, '_tri_drawn_key', None)
        if drawn is None:
            drawn = self._tri_drawn_key = {}
        # Drags read the resident preview when there is one. A slice along
        # X or Y is strided across the whole file; from the preview it is a
        # memory read, which is the difference between a slider that
        # follows the pointer and one that lags behind it.
        use_preview = cache.has_preview()
        for axis, idx in (('X', xi), ('Y', yi), ('Z', zi)):
            key = (idx, ww, wc, max_px, use_preview)
            if drawn.get(axis) != key:
                sl, _ = self._display_slice(axis, idx, ww, wc, max_px)
                self._tri_im[axis].set_data(self._tri_rgba(sl, cmap_name))
                changed.add(axis)
                drawn[axis] = key
            # Titles are not animated and are drawn only on full draws; the
            # sync timer below refreshes them after the drag settles.
            self._tri_title[axis].set_text(
                {'X': f"Sagittal  X={xi}", 'Y': f"Coronal   Y={yi}",
                 'Z': f"Axial     Z={zi}"}[axis]
            )

        # Crosshairs track all three indices on every panel.
        self._tri_hline['X'].set_ydata([zi, zi])
        self._tri_vline['X'].set_xdata([yi, yi])
        self._tri_hline['Y'].set_ydata([zi, zi])
        self._tri_vline['Y'].set_xdata([xi, xi])
        self._tri_hline['Z'].set_ydata([yi, yi])
        self._tri_vline['Z'].set_xdata([xi, xi])

        try:
            self._blit_panels(changed)
        except Exception:
            self._tri_bg_valid = False
            self._refresh_triplanar()
            return
        self._note_tri_refresh_cost(t0)
        # Full-fidelity draw (titles, measurements) once dragging pauses.
        self._tri_sync_timer.start(300)

    def _update_tri_sliders(self):
        if self._gray is None:
            return
        g = self._gray
        dims = {'X': g.shape[0], 'Y': g.shape[1], 'Z': g.shape[2]}
        for ax, dim in dims.items():
            mid = dim // 2
            sl = self._tri_sliders[ax]
            sl.blockSignals(True)
            sl.setMinimum(0)
            sl.setMaximum(max(0, dim - 1))
            sl.setValue(mid)
            sl.blockSignals(False)
            self._tri_idx[ax] = mid
            self._tri_idx_widgets[ax].setText(str(mid))

    def _reset_tri_artists(self):
        """Drop cached matplotlib artists. Call after the volume shape changes."""
        self._tri_bg0 = {}
        self._tri_bg1 = {}
        self._tri_bg_valid = False
        self._tri_drawn_key = {}
        self._tri_im = {'X': None, 'Y': None, 'Z': None}
        self._tri_title.clear()
        self._tri_hline.clear()
        self._tri_vline.clear()
        self._tri_compass.clear()
        # Measurement artists live on the axes about to be cleared; drop the
        # references so no dead objects are kept.
        for axis in ('X', 'Y', 'Z'):
            self._dist_artists[axis] = []
            self._dist_points[axis] = []
            self._roi_rect_artist[axis] = None
            self._roi_bounds[axis] = None
            self._profile_artists[axis] = []
            self._profile_points[axis] = []
        if hasattr(self, '_tri_measure_status'):
            self._tri_measure_status.setText("")
        self._tri_cursor = {}
        self._tri_cursor_axis = None
        self._tri_probe_ok = True

    # Layout, scale bar and pointer probe

    def _init_tri_probe(self):
        """Set up the live voxel readout and the guide crosshair."""
        self._tri_cursor: dict = {}
        self._tri_cursor_axis = None
        self._tri_hover_t = 0.0
        self._tri_probe_ok = True
        self._tri_px_budget = None
        self._tri_refresh_cost_ms = 0.0
        self._tri_canvas.mpl_connect('motion_notify_event', self._on_tri_hover)
        self._tri_canvas.mpl_connect(
            'axes_leave_event', self._on_tri_hover_leave
        )

    def _tri_lut(self, cmap_name: str):
        """A 256-entry RGBA table for *cmap_name*, built once and reused.

        Matplotlib normally maps a float array through the norm and the
        colormap on every draw. That is measurable: on a panel-sized slice
        it is about half the cost of a frame, and it is the same mapping
        every time. Doing it as a lookup on pre-quantised values and
        handing matplotlib finished RGBA bytes skips it entirely.

        256 levels is what the display has, so nothing visible is lost --
        the window has already mapped the interesting grey range onto
        [0, 1] before this point.
        """
        cache = getattr(self, '_tri_lut_cache', None)
        if cache is None:
            cache = self._tri_lut_cache = {}
        lut = cache.get(cmap_name)
        if lut is None:
            import matplotlib.cm as mcm
            lut = (mcm.get_cmap(cmap_name)(np.linspace(0.0, 1.0, 256))
                   * 255.0).astype(np.uint8)
            cache[cmap_name] = lut
        return lut

    def _tri_rgba(self, windowed, cmap_name: str):
        """Windowed float in [0, 1] -> RGBA uint8, through the table."""
        idx = np.multiply(windowed, 255.0).clip(0, 255).astype(np.uint8)
        return self._tri_lut(cmap_name)[idx]

    def _display_slice(self, axis: str, idx: int, ww: float, wc: float,
                       max_px: int | None):
        """The slice to draw for *axis*, and whether it came from the preview.

        Every read that ends up on screen goes through here, so there is
        one rule rather than three: use the preview whenever it exists, and
        let :meth:`_tri_settle_refresh` upgrade the panel afterwards if the
        preview is genuinely coarser than the panel can show. The point is
        that no path reachable from a slider, a resize or a colormap change
        can ever read the file on the GUI thread.
        """
        use_preview = self._slice_cache.has_preview()
        return self._slice_cache.get(axis, idx, ww, wc, max_px=max_px,
                                     preview=use_preview), use_preview

    def _refine_step(self, axis: str) -> int:
        """How much finer than the preview this panel could actually draw.

        The panels decimate to their own pixel budget anyway. When the
        preview is already at least that fine, a full-resolution read
        produces the *same* samples -- measurably identical, not merely
        similar -- so there is nothing to refine and the read would be pure
        cost. Returns 1 when refinement would not change the picture.
        """
        cache = self._slice_cache
        if not cache.has_preview() or self._gray is None:
            return 1
        max_px = self._tri_display_px()
        if not max_px:
            return cache.preview_step
        shape = tuple(int(d) for d in self._gray.shape[:3])
        plane = {'X': (shape[1], shape[2]), 'Y': (shape[0], shape[2]),
                 'Z': (shape[0], shape[1])}[axis]
        display_step = max(1, int(math.ceil(max(plane) / max_px)))
        return 1 if cache.preview_step <= display_step else cache.preview_step

    def _tri_settle_refresh(self):
        """Once dragging pauses, redraw -- reading the file only if it helps.

        This used to re-read all three panels at full resolution inline. On
        a volume that is not resident, a sagittal read is tens to hundreds
        of milliseconds, and dragging back and forth fires this timer after
        every pause, so the window locked up repeatedly. That is the exact
        cost the preview exists to keep off the interaction path, and it
        had no business running on the GUI thread.

        Two rules now, both borrowed from how a dedicated slice viewer
        behaves: never read during interaction, and never read at all
        unless the result would differ from what is already on screen.
        """
        if not HAS_MPL or self._gray is None:
            self._tri_canvas.draw_idle()
            return

        axes_to_refine = [a for a in ('X', 'Y', 'Z') if self._refine_step(a) > 1]
        if not axes_to_refine or getattr(self, '_tri_refine_busy', False):
            # Nothing finer to show, or a refinement is already running.
            self._tri_canvas.draw_idle()
            return

        g = self._gray
        window = self._effective_window()
        if window is None:
            self._tri_canvas.draw_idle()
            return
        ww, wc = window
        max_px = self._tri_display_px()
        limits = {'X': g.shape[0], 'Y': g.shape[1], 'Z': g.shape[2]}
        wanted = {
            axis: max(0, min(self._tri_idx[axis], limits[axis] - 1))
            for axis in axes_to_refine
        }
        drawn = getattr(self, '_tri_drawn_key', None) or {}
        wanted = {a: i for a, i in wanted.items()
                  if drawn.get(a) != (i, ww, wc, max_px, False)}
        if not wanted:
            self._tri_canvas.draw_idle()
            return

        self._tri_refine_busy = True
        self._tri_canvas.draw_idle()          # show the preview frame now

        def _work():
            results = {}
            try:
                for axis, idx in wanted.items():
                    results[axis] = self._slice_cache.get(
                        axis, idx, ww, wc, max_px=max_px, preview=False)
            except Exception:
                results = {}

            def _install():
                self._tri_refine_busy = False
                # The volume or the position may have moved on while this
                # ran; a stale slice must never be painted over a fresh one.
                if self._gray is not g:
                    return
                keys = getattr(self, '_tri_drawn_key', None) or {}
                cmap_name = self._cmap_var.get()
                for axis, sl in results.items():
                    if self._tri_idx[axis] != wanted[axis]:
                        continue
                    try:
                        self._tri_im[axis].set_data(
                            self._tri_rgba(sl, cmap_name))
                        keys[axis] = (wanted[axis], ww, wc, max_px, False)
                    except Exception:
                        pass
                self._tri_drawn_key = keys
                self._tri_canvas.draw_idle()
            self.after(0, _install)

        self._run_task("Refine slices", _work)

    def start_tri_preview(self):
        """Build the interactive preview for the loaded volume, in background.

        Called after a volume is bound. Until it finishes, dragging reads
        the file exactly as before, so this is an improvement that arrives
        rather than a step the user waits on.
        """
        if self._gray is None or getattr(self, '_tri_preview_busy', False):
            return
        from ..core import preview as preview_mod
        from ..core.io import LazyGrayVolume

        if not isinstance(self._gray, LazyGrayVolume):
            # Already a real array in memory: a slice along any axis is a
            # memory read, which is what the preview exists to provide. The
            # background materialiser promotes small volumes to this state,
            # and building a preview for one would be pure waste.
            return

        shape = tuple(int(d) for d in self._gray.shape[:3])
        step = preview_mod.choose_step(shape)
        if step == 1:
            return                       # already small enough to read directly

        self._tri_preview_busy = True
        mb = preview_mod.estimate_bytes(shape, step) / 1024 ** 2
        self._append_log(
            f"  Building the interactive preview at 1/{step} "
            f"({mb:.0f} MiB) so slider drags do not re-read the file.", 'dim')

        gray = self._gray

        def _work():
            try:
                array, built_step = preview_mod.build_preview(
                    gray, cancel=lambda: bool(self.cancel_requested()))
            except Exception as ex:                   # noqa: BLE001
                self._tri_preview_busy = False
                self._append_log(f"  Preview not built: {ex}", 'dim')
                return

            def _install():
                self._tri_preview_busy = False
                # The volume may have been replaced while this was running.
                if array is None or self._gray is not gray:
                    return
                self._slice_cache.set_preview(array, built_step)
                self._tri_drawn_key = {}
                # The debounce widens itself when refreshes measure slow.
                # Those measurements were taken against file reads; keeping
                # them would pace the slider to a cost that no longer
                # exists.
                self._tri_refresh_cost_ms = 0.0
                self._append_log(
                    f"  Interactive preview ready: {array.shape} at "
                    f"1/{built_step}. Dragging now reads memory; the panels "
                    f"return to full resolution when you stop.", 'ok')
            self.after(0, _install)

        self._run_task("Preview", _work)

    def _note_tri_refresh_cost(self, t0):
        """Track what one blit refresh costs on this machine, in ms."""
        cost = (time.perf_counter() - t0) * 1000.0
        prev = getattr(self, '_tri_refresh_cost_ms', cost)
        self._tri_refresh_cost_ms = prev * 0.7 + cost * 0.3

    def _tri_debounce_ms(self):
        """Slider debounce, widened where a refresh is measurably slow."""
        cost = getattr(self, '_tri_refresh_cost_ms', 0.0)
        return int(max(SLIDER_DEBOUNCE_MS, min(cost * 1.2, 120.0)))

    def _tri_display_px(self):
        """Longest side, in pixels, a decimated slice is rendered at."""
        return getattr(self, '_tri_px_budget', None) or MAX_TRI_DISPLAY_PX

    def _update_tri_px_budget(self):
        """Match the decimation budget to the on-screen panel size.

        Quantised to 64 px so resizing does not churn the slice-cache keys.
        """
        try:
            bbox = self._ax_axi.get_window_extent()
            longest = max(float(bbox.width), float(bbox.height))
            ratio = float(self._tri_canvas.devicePixelRatioF())
        except Exception:
            return
        if longest <= 0:
            return
        px = int(longest * max(1.0, ratio))
        px = max(192, min(px, MAX_TRI_DISPLAY_PX))
        # Round up: rendering below the panel's own resolution is visible.
        self._tri_px_budget = max(192, math.ceil(px / 64.0) * 64)

    def _voxel_sizes(self):
        """(dx, dy, dz) in mm, falling back to unit spacing."""
        try:
            dx, dy, dz = (float(v) for v in self._img.header.get_zooms()[:3])
        except Exception:
            return 1.0, 1.0, 1.0
        if not all(math.isfinite(v) and v > 0 for v in (dx, dy, dz)):
            return 1.0, 1.0, 1.0
        return dx, dy, dz

    def _on_tri_resize(self, _event):
        if self._gray is None:
            return
        self._layout_tri_panels()
        self._fit_tri_panels()
        self._tri_bg_valid = False

    def _tri_view_window(self):
        """Physical window, in mm, containing all three views.

        Sagittal spans ny*dy by nz*dz, coronal nx*dx by nz*dz and axial
        nx*dx by ny*dy, so this box holds the largest extent in each
        direction. Giving every panel the same window puts the three views
        at one scale, so a feature has the same on-screen size in all of
        them and a single scale bar applies to all three.
        """
        g = self._gray
        if g is None:
            return 1.0, 1.0
        nx, ny, nz = (int(v) for v in g.shape[:3])
        dx, dy, dz = self._voxel_sizes()
        width = max(nx * dx, ny * dy)
        height = max(nz * dz, ny * dy)
        if not (width > 0 and height > 0):
            return 1.0, 1.0
        return float(width), float(height)

    def _layout_tri_panels(self):
        """Position the three panels.

        Two arrangements, selected by the "Same scale in all views" switch.

        Same scale on
            Every box is identical and shaped like the physical window that
            contains all three views. One millimetre is the same number of
            pixels everywhere, so the views can be compared directly.

        Same scale off
            Each box takes the shape of its own slice and all three share
            one height. Panels are no longer comparable, but each view is as
            large as the space allows. This matters on a tall scan, where a
            common window shrinks the axial view into a narrow box.

        In both cases the axes aspect keeps the geometry true; a slice is
        never stretched to fill its panel.
        """
        g = self._gray
        if not HAS_MPL or g is None:
            return
        fig_w, fig_h = (float(v) for v in self._tri_fig.get_size_inches())
        if fig_w <= 0 or fig_h <= 0:
            return
        m = TRI_MARGINS
        avail_w = (m['right'] - m['left']) - 2 * m['gap']
        avail_h = m['top'] - m['bottom']
        if avail_w <= 0 or avail_h <= 0:
            return

        nx, ny, nz = (int(v) for v in g.shape[:3])
        dx, dy, dz = self._voxel_sizes()
        size_mm = {
            'X': (ny * dy, nz * dz),
            'Y': (nx * dx, nz * dz),
            'Z': (nx * dx, ny * dy),
        }

        if self._tri_same_scale():
            w_mm, h_mm = self._tri_view_window()
            shapes = {axis: (w_mm, h_mm) for axis in ('X', 'Y', 'Z')}
        else:
            shapes = size_mm

        # Figure-fraction height per unit width, for a box of each panel
        # shape.
        ks = {}
        for axis in ('X', 'Y', 'Z'):
            w_mm, h_mm = shapes[axis]
            k = (h_mm / w_mm) * (fig_w / fig_h) if w_mm > 0 else 1.0
            ks[axis] = k if (math.isfinite(k) and k > 0) else 1.0

        # One height for all three; each width follows from its own shape.
        height = min(avail_h, avail_w / sum(1.0 / k for k in ks.values()))
        widths = {axis: height / ks[axis] for axis in ('X', 'Y', 'Z')}
        total_w = sum(widths.values())
        if height <= 0 or total_w <= 0:
            return

        x = m['left'] + (avail_w - total_w) / 2.0
        y = m['bottom'] + (avail_h - height) / 2.0
        for axis in ('X', 'Y', 'Z'):
            width = widths[axis]
            try:
                self._panel_ax_obj(axis).set_position([x, y, width, height])
            except Exception:
                return
            x += width + m['gap']

    def _tri_same_scale(self) -> bool:
        check = getattr(self, '_tri_same_scale_check', None)
        return True if check is None else bool(check.isChecked())

    def _fit_tri_panels(self):
        """Frame each panel's slice inside its box, undistorted.

        The view is widened in whichever direction is short of the box
        shape, so the limits already satisfy the panel aspect and the spare
        room becomes padding around a centred slice. Getting this exactly
        right is what keeps matplotlib quiet: on the next draw it would
        otherwise widen a limit itself to honour the aspect and warn that it
        is ignoring the fixed limits, once per panel per frame.

        The box is read back from the axes rather than remembered from the
        layout pass. During a window resize the layout can bail out on a
        transient zero-sized figure, and a remembered box then belongs to
        the previous size while the figure has already moved on.
        """
        g = self._gray
        if not HAS_MPL or g is None:
            return
        nx, ny, nz = (int(v) for v in g.shape[:3])
        counts = {'X': (ny, nz), 'Y': (nx, nz), 'Z': (nx, ny)}
        same_scale = self._tri_same_scale()
        win_w_mm, win_h_mm = self._tri_view_window()
        fig_w, fig_h = (float(v) for v in self._tri_fig.get_size_inches())
        if fig_w <= 0 or fig_h <= 0:
            return

        for axis in ('X', 'Y', 'Z'):
            ax_obj = self._panel_ax_obj(axis)
            if ax_obj is None:
                continue
            h_mm, v_mm = self._panel_spacings(axis)
            n_h, n_v = counts[axis]
            if h_mm <= 0 or v_mm <= 0:
                continue

            if same_scale:
                span_h = win_w_mm / h_mm
                span_v = win_h_mm / v_mm
            else:
                span_h, span_v = float(n_h), float(n_v)
            span_h = max(span_h, 1.0)
            span_v = max(span_v, 1.0)

            # The box the layout gave this panel, read back from the axes so
            # the limits always match the box actually in force.
            box = ax_obj.get_position()
            box_w_px = box.width * fig_w
            box_h_px = box.height * fig_h
            aspect = v_mm / h_mm
            if box_w_px > 0 and box_h_px > 0 and aspect > 0:
                # Undistorted requires box_h/box_w == aspect*span_v/span_h.
                need_h = span_v * aspect * box_w_px / box_h_px
                if need_h > span_h:
                    span_h = need_h
                else:
                    span_v = span_h * (box_h_px / box_w_px) / aspect

            cx = (n_h - 1) / 2.0
            cy = (n_v - 1) / 2.0
            try:
                ax_obj.set_xlim(cx - span_h / 2.0, cx + span_h / 2.0)
                ax_obj.set_ylim(cy - span_v / 2.0, cy + span_v / 2.0)
            except Exception:
                continue
        self._tri_bg_valid = False

    def _on_tri_scale_mode_changed(self, _checked=False):
        """Re-frame the panels after the scale mode is switched."""
        if self._gray is None:
            return
        self._layout_tri_panels()
        self._fit_tri_panels()
        self._tri_canvas.draw_idle()

    def _add_tri_scale_bar(self, ax_obj, axis, extent):
        """Draw a millimetre scale bar snapped to a 1-2-5 step.

        The length follows the panel visible width, so when the three views
        share one scale they also share one bar length.
        """
        h_mm, _ = self._panel_spacings(axis)
        span = abs(float(extent[1]) - float(extent[0]))
        try:
            lo, hi = ax_obj.get_xlim()
            view_span = abs(float(hi) - float(lo))
        except Exception:
            view_span = span
        target = max(view_span, span) * h_mm * 0.25
        if h_mm <= 0 or span <= 0 or not math.isfinite(target) or target <= 0:
            return
        exponent = math.floor(math.log10(target))
        length_mm = 10.0 ** exponent
        for mult in (1.0, 2.0, 5.0):
            candidate = mult * 10.0 ** exponent
            if candidate <= target:
                length_mm = candidate
        y_span = abs(float(extent[3]) - float(extent[2]))
        x0 = float(extent[0]) + span * 0.04
        y0 = float(extent[2]) + y_span * 0.05
        ax_obj.plot([x0, x0 + length_mm / h_mm], [y0, y0],
                    color=TEXT, lw=2.0, solid_capstyle='butt')
        label = (f"{length_mm:g} mm" if length_mm >= 1.0
                 else f"{length_mm * 1000.0:g} µm")
        ax_obj.text(x0 + (length_mm / h_mm) / 2.0, y0 + y_span * 0.015,
                    label, color=TEXT, fontsize=6, ha='center', va='bottom')

    def _tri_voxel_at(self, axis, xdata, ydata):
        """Panel coordinates to a clamped (ix, iy, iz) voxel index."""
        g = self._gray
        if g is None or xdata is None or ydata is None:
            return None
        ex, ey = int(round(float(xdata))), int(round(float(ydata)))
        xi, yi, zi = self._tri_idx['X'], self._tri_idx['Y'], self._tri_idx['Z']
        if axis == 'Z':
            ix, iy, iz = ex, ey, zi
        elif axis == 'Y':
            ix, iy, iz = ex, yi, ey
        else:
            ix, iy, iz = xi, ex, ey
        return (max(0, min(ix, g.shape[0] - 1)),
                max(0, min(iy, g.shape[1] - 1)),
                max(0, min(iz, g.shape[2] - 1)))

    def _tri_sample_values(self, ix, iy, iz):
        """Grey value at a voxel, or ``None`` once sampling proves slow.

        A compressed proxy may decode a whole plane per read, so live
        sampling disables itself rather than stalling the pointer.
        """
        if not getattr(self, '_tri_probe_ok', True) or self._gray is None:
            return None
        try:
            t0 = time.perf_counter()
            raw = float(self._gray[ix, iy, iz])
            if time.perf_counter() - t0 > TRI_SAMPLE_BUDGET:
                self._tri_probe_ok = False
        except Exception:
            return None
        return raw

    def _tri_probe_text(self, axis, ix, iy, iz):
        """Status-line text for one voxel: index, position and values."""
        name = {'X': 'Sagittal', 'Y': 'Coronal', 'Z': 'Axial'}[axis]
        dx, dy, dz = self._voxel_sizes()
        text = (f"{name}  voxel ({ix}, {iy}, {iz})  "
                f"mm ({ix * dx:.3f}, {iy * dy:.3f}, {iz * dz:.3f})")
        raw = self._tri_sample_values(ix, iy, iz)
        if raw is not None:
            text += f"  value={raw:.4g}"
        return text

    def _on_tri_hover(self, event):
        """Refresh the readout and guide crosshair as the pointer moves."""
        if self._gray is None or self._roi_drag_axis is not None:
            return
        ax_map = {self._ax_sag: 'X', self._ax_cor: 'Y', self._ax_axi: 'Z'}
        axis = ax_map.get(event.inaxes)
        if axis is None:
            self._hide_tri_cursor()
            return
        now = time.perf_counter()
        if now - getattr(self, '_tri_hover_t', 0.0) < TRI_HOVER_INTERVAL:
            return
        self._tri_hover_t = now
        vox = self._tri_voxel_at(axis, event.xdata, event.ydata)
        if vox is None:
            return
        self._probe_var.set(self._tri_probe_text(axis, *vox))
        self._move_tri_cursor(axis, event.xdata, event.ydata)

    def _on_tri_hover_leave(self, _event):
        self._hide_tri_cursor()

    def _tri_cursor_lines(self, axis):
        """Guide lines for a panel, created on first use."""
        ax_obj = self._panel_ax_obj(axis)
        pair = getattr(self, '_tri_cursor', {}).get(axis)
        if pair is None or pair[0].axes is not ax_obj:
            pair = (ax_obj.axhline(0.0, color=TEAL, lw=0.7, ls=':', alpha=0.9),
                    ax_obj.axvline(0.0, color=TEAL, lw=0.7, ls=':', alpha=0.9))
            for art in pair:
                art.set_visible(False)
                art.set_animated(True)
            self._tri_cursor[axis] = pair
        return pair

    def _tri_hover_artists(self, axis):
        """Animated guide artists for the blit compositor."""
        return getattr(self, '_tri_cursor', {}).get(axis, ())

    def _move_tri_cursor(self, axis, xdata, ydata):
        """Position the guide lines and blit only the panels affected."""
        if not self._tri_bg_valid:
            return
        try:
            h_line, v_line = self._tri_cursor_lines(axis)
            h_line.set_ydata([ydata, ydata])
            v_line.set_xdata([xdata, xdata])
            h_line.set_visible(True)
            v_line.set_visible(True)
            panels = {axis}
            prev = self._tri_cursor_axis
            if prev is not None and prev != axis:
                for art in self._tri_cursor.get(prev, ()):
                    art.set_visible(False)
                panels.add(prev)
            self._tri_cursor_axis = axis
            self._blit_panels(set(), only=panels)
        except Exception:
            self._tri_bg_valid = False

    def _hide_tri_cursor(self):
        prev = getattr(self, '_tri_cursor_axis', None)
        if prev is None:
            return
        self._tri_cursor_axis = None
        for art in getattr(self, '_tri_cursor', {}).get(prev, ()):
            art.set_visible(False)
        if self._tri_bg_valid:
            try:
                self._blit_panels(set(), only={prev})
            except Exception:
                self._tri_bg_valid = False

    # Per-axis PNG export

    def _export_slice(self, axis):
        if self._gray is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save slice as PNG", "",
            "PNG (*.png);;All files (*.*)",
        )
        if not path:
            return
        g = self._gray
        idx = self._tri_idx[axis]
        cmap = self._cmap_var.get()
        ww, wc = self._ww, self._wc
        if ww is None or wc is None:
            ww, wc = auto_window(g)
        if axis == 'X':
            sl = g[idx, :, :].T
        elif axis == 'Y':
            sl = g[:, idx, :].T
        else:
            sl = g[:, :, idx].T
        sl_win = apply_window(sl.astype(np.float32, copy=False), ww, wc)
        # Figure plus Agg canvas directly rather than pyplot: deps.py sets
        # the global backend to TkAgg and switching it while Qt owns the main
        # loop raises ImportError.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(6, 6), facecolor='white')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        # Physical aspect, so exported figures are not stretched.
        h_mm, v_mm = self._panel_spacings(axis)
        aspect = (v_mm / h_mm) if h_mm > 0 and v_mm > 0 else 'equal'
        ax.imshow(sl_win, cmap=cmap, origin='lower', vmin=0, vmax=1,
                  aspect=aspect)
        ax.axis('off')
        self._add_tri_scale_bar(
            ax, axis,
            (-0.5, sl_win.shape[1] - 0.5, -0.5, sl_win.shape[0] - 0.5),
        )
        self._draw_measurements_on(ax, axis)
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        self._append_log(f"  Exported {axis} slice → {Path(path).name}", 'ok')

    def _draw_measurements_on(self, ax, axis):
        """Re-render distance, profile and ROI overlays for `axis` onto `ax`.

        The live overlay artists belong to the on-screen canvas while PNG
        export builds a new Figure, so measurements are replayed from their
        stored coordinates.
        """
        # Distance overlay
        pts = getattr(self, '_dist_points', {}).get(axis, [])
        if len(pts) >= 1:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs, ys, marker='o', markersize=5, linestyle='-' if len(pts) == 2 else 'None',
                color=ACCENT, markerfacecolor=ACCENT, markeredgecolor='white',
                markeredgewidth=0.8, lw=1.6, alpha=0.95,
            )
            if len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                h_mm, v_mm = self._panel_spacings(axis)
                dist_mm = float(np.hypot((x1 - x0) * h_mm, (y1 - y0) * v_mm))
                ax.text(
                    (x0 + x1) / 2.0, (y0 + y1) / 2.0 + 2.0, f"{dist_mm:.2f} mm",
                    color=ACCENT, fontsize=8, fontweight='bold', ha='center',
                    va='bottom',
                    bbox=dict(
                        facecolor=BG, edgecolor=ACCENT,
                        boxstyle='round,pad=0.2', alpha=0.85,
                    ),
                )

        # Profile line overlay
        p_pts = getattr(self, '_profile_points', {}).get(axis, [])
        if len(p_pts) >= 1:
            ax.plot(
                [p[0] for p in p_pts], [p[1] for p in p_pts],
                marker='s', markersize=5,
                linestyle='-' if len(p_pts) == 2 else 'None',
                color=TEAL, markerfacecolor=TEAL, markeredgecolor='white',
                markeredgewidth=0.8, lw=1.6, alpha=0.95,
            )

        # ROI overlay
        bounds = getattr(self, '_roi_bounds', {}).get(axis)
        if bounds is not None:
            from matplotlib.patches import Rectangle
            x0, y0, x1, y1 = bounds
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            ax.add_patch(Rectangle(
                (rx, ry), rw, rh,
                linewidth=1.4, edgecolor=ACCENT, facecolor=ACCENT, alpha=0.15,
            ))
            # Recompute stats so the PNG label matches the live readout
            g = self._gray
            h0, h1 = sorted((int(round(x0)), int(round(x1))))
            v0, v1 = sorted((int(round(y0)), int(round(y1))))
            h0 = max(0, h0); v0 = max(0, v0)
            try:
                if axis == 'Z':
                    h1 = min(h1, g.shape[0] - 1)
                    v1 = min(v1, g.shape[1] - 1)
                    block = np.asarray(
                        g[h0:h1 + 1, v0:v1 + 1, self._tri_idx['Z']], dtype=np.float32
                    )
                elif axis == 'Y':
                    h1 = min(h1, g.shape[0] - 1)
                    v1 = min(v1, g.shape[2] - 1)
                    block = np.asarray(
                        g[h0:h1 + 1, self._tri_idx['Y'], v0:v1 + 1], dtype=np.float32
                    )
                else:
                    h1 = min(h1, g.shape[1] - 1)
                    v1 = min(v1, g.shape[2] - 1)
                    block = np.asarray(
                        g[self._tri_idx['X'], h0:h1 + 1, v0:v1 + 1], dtype=np.float32
                    )
            except Exception:
                block = None
            if block is not None and block.size > 0:
                h_mm, v_mm = self._panel_spacings(axis)
                area_mm2 = (h1 - h0 + 1) * (v1 - v0 + 1) * h_mm * v_mm
                ax.text(
                    rx + rw / 2.0, ry + rh + 1.5,
                    f"μ={float(block.mean()):.1f} σ={float(block.std()):.1f}\n"
                    f"{area_mm2:.1f} mm²",
                    color=ACCENT, fontsize=7, ha='center', va='bottom',
                    fontweight='bold',
                    bbox=dict(
                        facecolor=BG, edgecolor=ACCENT,
                        boxstyle='round,pad=0.2', alpha=0.85,
                    ),
                )


class _QLabelVar:
    """QLabel wrapper that exposes ``get()`` / ``set(value)``."""

    def __init__(self, label: QLabel):
        self._lbl = label

    def get(self) -> str:
        return self._lbl.text()

    def set(self, value) -> None:
        self._lbl.setText(str(value))
