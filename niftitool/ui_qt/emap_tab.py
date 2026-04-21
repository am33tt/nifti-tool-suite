"""E-Map viewer tab (PyQt6 port)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QSlider, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, E_CMAPS, ERR, PANEL2, SLIDER_DEBOUNCE_MS, TEAL,
    TEXT, TEXT_DIM,
)
from ..core.windowing import apply_window, auto_window
from ..deps import HAS_MPL, mcolors, np
from .widgets import styled_btn, styled_entry


class _WidgetVar:
    def __init__(self, w):
        self._w = w

    def get(self):
        if isinstance(self._w, QLineEdit):
            return self._w.text()
        if isinstance(self._w, QCheckBox):
            return bool(self._w.isChecked())
        if isinstance(self._w, QComboBox):
            return self._w.currentText()
        return None

    def set(self, value):
        if isinstance(self._w, QLineEdit):
            self._w.setText(str(value))
        elif isinstance(self._w, QCheckBox):
            self._w.setChecked(bool(value))
        elif isinstance(self._w, QComboBox):
            idx = self._w.findText(str(value))
            if idx >= 0:
                self._w.setCurrentIndex(idx)


class _QLabelVar:
    def __init__(self, label):
        self._lbl = label

    def get(self):
        return self._lbl.text()

    def set(self, value):
        self._lbl.setText(str(value))


class EmapTabMixin:
    """Adds the E-Map Viewer tab to :class:`NiftiApp`."""

    def _build_emap_tab(self, parent):
        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not HAS_MPL:
            lbl = QLabel("matplotlib not installed", parent)
            lbl.setStyleSheet(f"color: {ERR}; background-color: {BG};")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(lbl)
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvasTkAgg,
            NavigationToolbar2QT as NavigationToolbar2Tk,
        )

        ctrl = QWidget(parent)
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(8, 4, 8, 4)
        ctrl_lay.setSpacing(4)

        lbl = QLabel("E colormap:", ctrl)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        ctrl_lay.addWidget(lbl)

        self._ecmap_widget = QComboBox(ctrl)
        self._ecmap_widget.addItems(E_CMAPS)
        self._ecmap_widget.setCurrentText("viridis")
        self._ecmap_widget.setFixedWidth(120)
        ctrl_lay.addWidget(self._ecmap_widget)
        self._ecmap_var = _WidgetVar(self._ecmap_widget)
        self._ecmap_widget.currentTextChanged.connect(
            lambda _t: self._refresh_emap_viewer()
        )

        self._emap_overlay_widget = QCheckBox("Overlay on CT", ctrl)
        self._emap_overlay_widget.setChecked(False)
        self._emap_overlay_widget.setFont(QFont("Segoe UI", 9))
        self._emap_overlay_widget.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        self._emap_overlay_widget.toggled.connect(
            lambda _c: self._refresh_emap_viewer()
        )
        ctrl_lay.addWidget(self._emap_overlay_widget)
        self._emap_overlay = _WidgetVar(self._emap_overlay_widget)

        alpha_lbl = QLabel("Alpha:", ctrl)
        alpha_lbl.setFont(QFont("Segoe UI", 9))
        alpha_lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        ctrl_lay.addWidget(alpha_lbl)

        self._emap_alpha_widget = styled_entry(ctrl, width=5)
        self._emap_alpha_widget.setText("0.6")
        ctrl_lay.addWidget(self._emap_alpha_widget)
        self._emap_alpha_var = _WidgetVar(self._emap_alpha_widget)

        ctrl_lay.addWidget(styled_btn(ctrl, "Refresh",
                                      self._refresh_emap_viewer, small=True))
        ctrl_lay.addStretch(1)
        root.addWidget(ctrl)

        self._emap_fig = Figure(facecolor=BG)
        self._emap_axes = self._emap_fig.subplots(1, 3)
        for ax in self._emap_axes:
            ax.set_facecolor(PANEL2)
            ax.axis('off')
        self._emap_fig.tight_layout(pad=0.4)
        self._emap_canvas = FigureCanvasTkAgg(self._emap_fig)
        root.addWidget(self._emap_canvas, 1)

        tb = QWidget(parent)
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(NavigationToolbar2Tk(self._emap_canvas, tb))
        tb_lay.addStretch(1)
        root.addWidget(tb)

        # Slider bar
        slider_bar = QFrame(parent)
        slider_bar.setStyleSheet(f"background-color: {PANEL2};")
        sgrid = QGridLayout(slider_bar)
        sgrid.setContentsMargins(8, 4, 8, 4)
        sgrid.setHorizontalSpacing(16)

        self._emap_sliders: dict = {}
        self._emap_idx_widgets: dict = {}
        self._emap_idx = {'X': 0, 'Y': 0, 'Z': 0}
        axis_info = [
            ('X', "Sagittal (X)"),
            ('Y', "Coronal  (Y)"),
            ('Z', "Axial    (Z)"),
        ]
        for col, (ax, label) in enumerate(axis_info):
            cf = QWidget(slider_bar)
            cf_lay = QVBoxLayout(cf)
            cf_lay.setContentsMargins(0, 0, 0, 0); cf_lay.setSpacing(2)
            title = QLabel(label, cf)
            title.setFont(QFont("Segoe UI", 9))
            title.setStyleSheet(f"color: {TEAL}; background-color: {PANEL2};")
            cf_lay.addWidget(title)

            row_w = QWidget(cf)
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(0, 0, 0, 0); row_lay.setSpacing(4)
            sl = QSlider(Qt.Orientation.Horizontal, row_w)
            sl.setMinimum(0); sl.setMaximum(100)
            row_lay.addWidget(sl, 1)
            idx_lbl = QLabel("--", row_w)
            idx_lbl.setFont(QFont("Consolas", 10))
            idx_lbl.setStyleSheet(f"color: {TEAL}; background-color: {PANEL2};")
            idx_lbl.setFixedWidth(40)
            row_lay.addWidget(idx_lbl)
            cf_lay.addWidget(row_w)

            sgrid.addWidget(cf, 0, col)
            sgrid.setColumnStretch(col, 1)

            sl.valueChanged.connect(
                lambda val, a=ax: self._on_emap_drag(a, val)
            )
            self._emap_sliders[ax] = sl
            self._emap_idx_widgets[ax] = idx_lbl

        root.addWidget(slider_bar)

        self._emap_idx_vars = {
            ax: _QLabelVar(lbl) for ax, lbl in self._emap_idx_widgets.items()
        }

        # Per-slice histogram
        self._emap_slice_fig = Figure(figsize=(4, 1.8), facecolor=BG)
        self._emap_slice_ax = self._emap_slice_fig.add_subplot(111)
        self._emap_slice_ax.set_facecolor(PANEL2)
        self._emap_slice_canvas = FigureCanvasTkAgg(self._emap_slice_fig)
        self._emap_slice_canvas.setFixedHeight(170)
        root.addWidget(self._emap_slice_canvas)

        # Debounce
        self._emap_redraw_pending = None
        self._emap_debounce_timer = QTimer(self)
        self._emap_debounce_timer.setSingleShot(True)
        self._emap_debounce_timer.timeout.connect(self._refresh_emap_viewer)

    # ── interaction ──────────────────────────────────────────────────────────

    def _on_emap_drag(self, axis, val):
        idx = int(val)
        self._emap_idx[axis] = idx
        self._emap_idx_widgets[axis].setText(str(idx))
        if self._E_map is None:
            return
        self._emap_debounce_timer.start(SLIDER_DEBOUNCE_MS)

    def _update_emap_sliders(self):
        if self._E_map is None:
            return
        for ax, dim in zip(('X', 'Y', 'Z'), self._E_map.shape[:3]):
            mid = dim // 2
            sl = self._emap_sliders[ax]
            sl.blockSignals(True)
            sl.setMinimum(0); sl.setMaximum(dim - 1); sl.setValue(mid)
            sl.blockSignals(False)
            self._emap_idx[ax] = mid
            self._emap_idx_widgets[ax].setText(str(mid))

    # ── refresh ──────────────────────────────────────────────────────────────

    def _refresh_emap_viewer(self):
        self._emap_redraw_pending = None
        if not HAS_MPL or self._E_map is None:
            return
        E = self._E_map
        cmap = self._ecmap_var.get()
        xi = max(0, min(self._emap_idx['X'], E.shape[0] - 1))
        yi = max(0, min(self._emap_idx['Y'], E.shape[1] - 1))
        zi = max(0, min(self._emap_idx['Z'], E.shape[2] - 1))
        E_min, E_max = float(E.min()), float(E.max())
        norm = mcolors.Normalize(vmin=E_min, vmax=E_max)

        slices_E = {'X': E[xi, :, :].T, 'Y': E[:, yi, :].T, 'Z': E[:, :, zi].T}
        titles = {
            'X': f"Sagittal X={xi}",
            'Y': f"Coronal  Y={yi}",
            'Z': f"Axial    Z={zi}",
        }

        overlay = self._emap_overlay.get()
        try:
            alpha = float(self._emap_alpha_var.get())
        except Exception:
            alpha = 0.6

        self._emap_fig.clear()
        axes = self._emap_fig.subplots(1, 3)

        for ax_obj, axis in zip(axes, ('X', 'Y', 'Z')):
            ax_obj.set_facecolor(PANEL2)
            ax_obj.axis('off')
            sl_E = slices_E[axis]
            if overlay and self._gray is not None:
                g = self._gray
                if axis == 'X':
                    sl_g = g[xi, :, :].T
                elif axis == 'Y':
                    sl_g = g[:, yi, :].T
                else:
                    sl_g = g[:, :, zi].T
                ww, wc = auto_window(sl_g)
                sl_g_w = apply_window(sl_g.astype(np.float32, copy=False), ww, wc)
                ax_obj.imshow(
                    sl_g_w, cmap='gray', origin='lower', vmin=0, vmax=1,
                    aspect='equal', interpolation='nearest',
                )
                ax_obj.imshow(
                    sl_E, cmap=cmap, norm=norm, origin='lower',
                    aspect='equal', interpolation='nearest', alpha=alpha,
                )
            else:
                im = ax_obj.imshow(
                    sl_E, cmap=cmap, norm=norm, origin='lower',
                    aspect='equal', interpolation='nearest',
                )
                self._emap_fig.colorbar(
                    im, ax=ax_obj, shrink=0.7, label='E [MPa]',
                    fraction=0.046, pad=0.04,
                )
            ax_obj.set_title(titles[axis], color=TEXT_DIM, fontsize=8, pad=3)

        self._emap_fig.tight_layout(pad=0.4)
        self._emap_canvas.draw_idle()

        # Per-slice histogram (axial)
        self._emap_slice_ax.clear()
        self._emap_slice_ax.set_facecolor(PANEL2)
        sl_flat = E[:, :, zi].ravel()
        self._emap_slice_ax.hist(sl_flat, bins=64, color=TEAL, alpha=0.8, linewidth=0)
        self._emap_slice_ax.set_xlabel(
            f"E [MPa]  axial Z={zi}", color=TEXT_DIM, fontsize=8,
        )
        self._emap_slice_ax.set_ylabel("count", color=TEXT_DIM, fontsize=8)
        self._emap_slice_ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in self._emap_slice_ax.spines.values():
            sp.set_color(BORDER)
        self._emap_slice_fig.tight_layout(pad=0.4)
        self._emap_slice_canvas.draw_idle()
