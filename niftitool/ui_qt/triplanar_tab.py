"""Tri-planar (sagittal / coronal / axial) viewer — PyQt6 port.

See ``ui/triplanar_tab.py`` for the performance rationale.  Everything
about the matplotlib draw path is identical; only the slider / layout
widgets change.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QSlider, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, ERR, PANEL2, SLIDER_DEBOUNCE_MS, TEXT, TEXT_DIM,
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

        # Persistent matplotlib artists (created once, updated each frame).
        self._tri_im:      dict = {'X': None, 'Y': None, 'Z': None}
        self._tri_title:   dict = {}
        self._tri_hline:   dict = {}
        self._tri_vline:   dict = {}
        self._tri_compass: dict = {}

        self._tri_fig.tight_layout(pad=0.4)
        self._tri_canvas = FigureCanvasTkAgg(self._tri_fig)
        root.addWidget(self._tri_canvas, 1)
        self._tri_canvas.mpl_connect('button_press_event', self._on_tri_click)

        tb_frame = QWidget(parent)
        tb_lay = QHBoxLayout(tb_frame)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(NavigationToolbar2Tk(self._tri_canvas, tb_frame))
        tb_lay.addStretch(1)
        root.addWidget(tb_frame)

        # ── slider bar ──
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
            col_frame = QWidget(slider_bar)
            col_lay = QVBoxLayout(col_frame)
            col_lay.setContentsMargins(0, 0, 0, 0)
            col_lay.setSpacing(2)

            title_lbl = QLabel(label, col_frame)
            title_lbl.setFont(QFont("Segoe UI", 9))
            title_lbl.setStyleSheet(
                f"color: {ACCENT}; background-color: {PANEL2};"
            )
            col_lay.addWidget(title_lbl)

            row_w = QWidget(col_frame)
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(4)
            sl = QSlider(Qt.Orientation.Horizontal, row_w)
            sl.setMinimum(0); sl.setMaximum(100)
            row_lay.addWidget(sl, 1)
            idx_lbl = QLabel("--", row_w)
            idx_lbl.setFont(QFont("Consolas", 10))
            idx_lbl.setStyleSheet(
                f"color: {ACCENT}; background-color: {PANEL2};"
            )
            idx_lbl.setFixedWidth(40)
            row_lay.addWidget(idx_lbl)
            col_lay.addWidget(row_w)

            sb_grid.addWidget(col_frame, 0, col)
            sb_grid.setColumnStretch(col, 1)

            sl.valueChanged.connect(
                lambda val, a=ax: self._on_tri_drag(a, val)
            )
            self._tri_sliders[ax] = sl
            self._tri_idx_widgets[ax] = idx_lbl

        root.addWidget(slider_bar)

        # ── export buttons ──
        export_bar = QWidget(parent)
        eb_lay = QHBoxLayout(export_bar)
        eb_lay.setContentsMargins(6, 2, 6, 2)
        for axis in ('Z', 'Y', 'X'):
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

    # ── slider / click handlers ──────────────────────────────────────────────

    def _on_tri_drag(self, axis, val):
        idx = int(val)
        self._tri_idx[axis] = idx
        self._tri_idx_widgets[axis].setText(str(idx))
        if self._gray is None:
            return
        # Debounce
        self._tri_debounce_timer.start(SLIDER_DEBOUNCE_MS)

    def _on_tri_click(self, event):
        if self._gray is None or event.inaxes is None:
            return
        ax_map = {self._ax_sag: 'X', self._ax_cor: 'Y', self._ax_axi: 'Z'}
        if event.inaxes not in ax_map:
            return
        axis = ax_map[event.inaxes]
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

    # ── refresh (hot path) ───────────────────────────────────────────────────

    def _refresh_triplanar(self):
        self._tri_redraw_pending = None
        if not HAS_MPL or self._gray is None:
            return

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

        for ax_obj, axis, idx, sl_win, title in panels:
            im = self._tri_im[axis]
            if im is None or im.get_array().shape != sl_win.shape:
                ax_obj.clear()
                ax_obj.set_facecolor(PANEL2)
                ax_obj.axis('off')
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
                self._tri_hline[axis] = ax_obj.axhline(
                    h_pos, color='#00CFFF', lw=0.9, alpha=0.75, linestyle='--',
                )
                self._tri_vline[axis] = ax_obj.axvline(
                    v_pos, color='#FFD700', lw=0.9, alpha=0.75, linestyle='--',
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

    # ── per-axis PNG export ──────────────────────────────────────────────────

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
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.imshow(sl_win, cmap=cmap, origin='lower', vmin=0, vmax=1)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self._append_log(f"  Exported {axis} slice → {Path(path).name}", 'ok')


class _QLabelVar:
    """QLabel wrapper that exposes ``get()`` / ``set(value)``."""

    def __init__(self, label: QLabel):
        self._lbl = label

    def get(self) -> str:
        return self._lbl.text()

    def set(self, value) -> None:
        self._lbl.setText(str(value))
