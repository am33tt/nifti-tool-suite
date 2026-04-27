"""Tri-planar (sagittal / coronal / axial) viewer.

See ``ui/triplanar_tab.py`` for the performance rationale. The matplotlib
draw path is identical; only the slider / layout widgets change.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIntValidator
from PyQt6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QRadioButton, QSizePolicy, QSlider,
    QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, AXIS_COLOR, BG, BORDER, ERR, PANEL2, SLIDER_DEBOUNCE_MS, TEAL,
    TEXT, TEXT_DIM,
)
from ..core.windowing import apply_window, auto_window
from ..deps import HAS_MPL, np
from .widgets import styled_btn


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

        self._tri_fig.set_tight_layout({'pad': 0.4, 'w_pad': 1.8})
        self._tri_canvas = FigureCanvasTkAgg(self._tri_fig)
        root.addWidget(self._tri_canvas, 1)
        self._tri_canvas.mpl_connect('button_press_event', self._on_tri_press)
        self._tri_canvas.mpl_connect('motion_notify_event', self._on_tri_motion)
        self._tri_canvas.mpl_connect('button_release_event', self._on_tri_release)

        self._build_tri_measure_bar(parent, root)

        tb_frame = QWidget(parent)
        tb_lay = QHBoxLayout(tb_frame)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(NavigationToolbar2Tk(self._tri_canvas, tb_frame))
        tb_lay.addStretch(1)
        root.addWidget(tb_frame)

        # slider bar
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
            idx_edit.setToolTip("Type a slice index and press Enter to jump.")
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

        # export buttons
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

        # Debounce state
        self._tri_redraw_pending = None
        self._tri_debounce_timer = QTimer(self)
        self._tri_debounce_timer.setSingleShot(True)
        self._tri_debounce_timer.timeout.connect(self._refresh_triplanar)

        # Tk-compat shim for ``_tri_idx_vars['X'].set(...)`` usage elsewhere.
        self._tri_idx_vars = {
            ax: _QLabelVar(lbl) for ax, lbl in self._tri_idx_widgets.items()
        }

    # slider / click handlers

    def _on_tri_drag(self, axis, val):
        idx = int(val)
        self._tri_idx[axis] = idx
        w = self._tri_idx_widgets[axis]
        w.blockSignals(True)
        w.setText(str(idx))
        w.blockSignals(False)
        if self._gray is None:
            return
        # Debounce
        self._tri_debounce_timer.start(SLIDER_DEBOUNCE_MS)

    def _on_tri_idx_entered(self, axis, edit):
        """User typed a slice number and committed (Enter / focus out).

        Clamp to the slider range and jump. Setting the slider value re-emits
        valueChanged, which refreshes the canvas through the debounce timer.
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
            if self._hu_vol is not None:
                probe += f"  HU={float(self._hu_vol[ix, iy, iz]):.1f}"
            if self._E_map is not None:
                probe += f"  E={float(self._E_map[ix, iy, iz]):.1f} MPa"
            self._probe_var.set(probe)
        except Exception:
            pass

    # measurement: toolbar + helpers

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

        # per-axis artist caches
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
            'profile': "Profile: click two points — a line plot will open.",
        }
        self._tri_measure_status.setText(hints.get(tool, ""))

    def _panel_spacings(self, axis):
        """Return (h_mm, v_mm) physical spacing of the panel `axis`.

        Panels display transposed slices:
            'X' (sagittal) → (h, v) = (y, z)
            'Y' (coronal)  → (h, v) = (x, z)
            'Z' (axial)    → (h, v) = (x, y)
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

    # distance tool

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

    # profile tool

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

        hu_vals = None
        if self._hu_vol is not None:
            try:
                if axis == 'Z':
                    hu_slice = np.asarray(self._hu_vol[:, :, zi], dtype=np.float32)
                elif axis == 'Y':
                    hu_slice = np.asarray(self._hu_vol[:, yi, :], dtype=np.float32)
                else:
                    hu_slice = np.asarray(self._hu_vol[xi, :, :], dtype=np.float32)
                if ndimage is not None:
                    hu_vals = ndimage.map_coordinates(
                        hu_slice, [h_t, v_t], order=1, mode='nearest',
                    )
                else:
                    hu_vals = hu_slice[hi, vi]
            except Exception:
                hu_vals = None

        d_mm = np.linspace(0.0, length_mm, n, dtype=np.float32)
        return d_mm, raw_vals, hu_vals, n

    def _open_profile_plot(self, axis, pts):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
        )
        from PyQt6.QtWidgets import QDialog, QVBoxLayout as _QVBoxLayout

        d_mm, raw, hu, n = self._sample_profile(axis, pts)
        panel_name = {'X': 'Sagittal', 'Y': 'Coronal', 'Z': 'Axial'}[axis]

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Intensity profile — {panel_name}")
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
        ax.plot(d_mm, raw, color=TEAL, lw=1.4, label='raw')
        ax.set_xlabel("distance (mm)", color=TEXT_DIM, fontsize=9)
        ax.set_ylabel("raw intensity", color=TEAL, fontsize=9)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color(BORDER)

        if hu is not None:
            ax2 = ax.twinx()
            ax2.plot(d_mm, hu, color=ACCENT, lw=1.2, linestyle='--', label='HU')
            ax2.set_ylabel("HU", color=ACCENT, fontsize=9)
            ax2.tick_params(colors=TEXT_DIM, labelsize=8)
            for side in ('top',):
                ax2.spines[side].set_visible(False)
            ax2.spines['right'].set_color(BORDER)

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
            self._tri_measure_status.setText("ROI too small — drag a larger rectangle.")
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
        if self._hu_vol is not None:
            try:
                if axis == 'Z':
                    hu_blk = np.asarray(
                        self._hu_vol[h0:h1 + 1, v0:v1 + 1, zi], dtype=np.float32
                    )
                elif axis == 'Y':
                    hu_blk = np.asarray(
                        self._hu_vol[h0:h1 + 1, yi, v0:v1 + 1], dtype=np.float32
                    )
                else:
                    hu_blk = np.asarray(
                        self._hu_vol[xi, h0:h1 + 1, v0:v1 + 1], dtype=np.float32
                    )
                extra = f"  HU μ={float(hu_blk.mean()):.1f}"
            except Exception:
                pass

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

    # refresh (hot path)

    def _refresh_triplanar(self):
        self._tri_redraw_pending = None
        if not HAS_MPL or self._gray is None:
            return

        # First volume loaded — retire the welcome banner.
        if getattr(self, '_tri_welcome', None) is not None:
            self._tri_welcome.set_visible(False)
            self._tri_welcome_sub.set_visible(False)

        g = self._gray
        cmap = self._cmap_var.get()
        xi = max(0, min(self._tri_idx['X'], g.shape[0] - 1))
        yi = max(0, min(self._tri_idx['Y'], g.shape[1] - 1))
        zi = max(0, min(self._tri_idx['Z'], g.shape[2] - 1))

        ww, wc = self._ww, self._wc
        if ww is None or wc is None:
            ww, wc = auto_window(g)

        cache = self._slice_cache
        sl_X = cache.get('X', xi, ww, wc)
        sl_Y = cache.get('Y', yi, ww, wc)
        sl_Z = cache.get('Z', zi, ww, wc)

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
        # For each panel, which slider drives the H-line and which drives the
        # V-line? Colors come from AXIS_COLOR so a line's hue always matches
        # its slider in the bottom bar.
        crosshair_axes = {
            'X': ('Z', 'Y'),  # sagittal: horiz = Z-slider, vert = Y-slider
            'Y': ('Z', 'X'),  # coronal
            'Z': ('Y', 'X'),  # axial
        }

        for ax_obj, axis, idx, sl_win, title in panels:
            im = self._tri_im[axis]
            if im is None or im.get_array().shape != sl_win.shape:
                ax_obj.clear()
                ax_obj.set_facecolor(PANEL2)
                # Subdued voxel-index ticks so the user can read positions.
                # Set once on creation; hot path never touches tick state.
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
                    sl_win, cmap=cmap, origin='lower', vmin=0, vmax=1,
                    aspect='equal', interpolation='nearest',
                )
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
            else:
                im.set_data(sl_win)
                im.set_cmap(cmap)
                self._tri_title[axis].set_text(title)
                h_pos, v_pos = crosshair_pos[axis]
                self._tri_hline[axis].set_ydata([h_pos, h_pos])
                self._tri_vline[axis].set_xdata([v_pos, v_pos])
                lft, rgt, bot, top_ = compass_spec[axis]
                c = self._tri_compass[axis]
                c['L'].set_text(lft); c['R'].set_text(rgt)
                c['B'].set_text(bot); c['T'].set_text(top_)

        self._tri_canvas.draw_idle()

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
        self._tri_im = {'X': None, 'Y': None, 'Z': None}
        self._tri_title.clear()
        self._tri_hline.clear()
        self._tri_vline.clear()
        self._tri_compass.clear()
        # Measurement artists live on the axes that are about to be cleared;
        # drop our dict references so we don't hold onto dead objects.
        for axis in ('X', 'Y', 'Z'):
            self._dist_artists[axis] = []
            self._dist_points[axis] = []
            self._roi_rect_artist[axis] = None
            self._roi_bounds[axis] = None
            self._profile_artists[axis] = []
            self._profile_points[axis] = []
        if hasattr(self, '_tri_measure_status'):
            self._tri_measure_status.setText("")

    # per-axis PNG export

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
        # Use Figure + Agg canvas directly rather than pyplot: deps.py sets
        # the global backend to TkAgg for the legacy tk UI, and switching
        # backends while Qt owns the main loop raises ImportError.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(6, 6), facecolor='white')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.imshow(sl_win, cmap=cmap, origin='lower', vmin=0, vmax=1)
        ax.axis('off')
        self._draw_measurements_on(ax, axis)
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        self._append_log(f"  Exported {axis} slice → {Path(path).name}", 'ok')

    def _draw_measurements_on(self, ax, axis):
        """Re-render any distance / ROI overlays for `axis` onto `ax`.

        The live overlay artists belong to the on-screen canvas; PNG export
        builds a fresh Figure, so we replay the measurements from stored
        coordinates.
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
