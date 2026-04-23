"""Histogram tab: intensity distribution and per-slice mean curve."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, ERR, PANEL2, TEAL, TEXT, TEXT_DIM,
)
from ..deps import HAS_MPL
from .widgets import styled_btn, styled_entry


class _WidgetVar:
    def __init__(self, w):
        self._w = w

    def get(self):
        if isinstance(self._w, QLineEdit):
            return self._w.text()
        if isinstance(self._w, QCheckBox):
            return bool(self._w.isChecked())
        return None

    def set(self, value):
        if isinstance(self._w, QLineEdit):
            self._w.setText(str(value))
        elif isinstance(self._w, QCheckBox):
            self._w.setChecked(bool(value))


class HistogramTabMixin:
    """Adds the Histogram tab to :class:`NiftiApp`."""

    def _build_histogram_tab(self, parent):
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

        self._hist_fig = Figure(facecolor=BG)
        self._hist_ax1, self._hist_ax2 = self._hist_fig.subplots(1, 2)
        for ax in (self._hist_ax1, self._hist_ax2):
            ax.set_facecolor(PANEL2)
        self._hist_canvas = FigureCanvasTkAgg(self._hist_fig)
        root.addWidget(self._hist_canvas, 1)

        tb = QWidget(parent)
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(NavigationToolbar2Tk(self._hist_canvas, tb))
        tb_lay.addStretch(1)
        root.addWidget(tb)

        opts = QWidget(parent)
        opts_lay = QHBoxLayout(opts)
        opts_lay.setContentsMargins(8, 4, 8, 4)
        opts_lay.setSpacing(4)

        lbl = QLabel("Bins:", opts)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        opts_lay.addWidget(lbl)

        self._hist_bins_widget = styled_entry(opts, width=5)
        self._hist_bins_widget.setText("256")
        opts_lay.addWidget(self._hist_bins_widget)
        self._hist_bins_var = _WidgetVar(self._hist_bins_widget)

        self._hist_show_hu_widget = QCheckBox("Show HU axis", opts)
        self._hist_show_hu_widget.setChecked(True)
        self._hist_show_hu_widget.setFont(QFont("Segoe UI", 9))
        self._hist_show_hu_widget.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        opts_lay.addWidget(self._hist_show_hu_widget)
        self._hist_show_hu = _WidgetVar(self._hist_show_hu_widget)

        self._hist_shade_phases_widget = QCheckBox("Shade phases", opts)
        self._hist_shade_phases_widget.setChecked(True)
        self._hist_shade_phases_widget.setFont(QFont("Segoe UI", 9))
        self._hist_shade_phases_widget.setStyleSheet(
            f"color: {TEXT}; background-color: transparent;"
        )
        opts_lay.addWidget(self._hist_shade_phases_widget)
        self._hist_shade_phases = _WidgetVar(self._hist_shade_phases_widget)

        opts_lay.addWidget(styled_btn(opts, "Re-run", self._do_histogram, small=True))
        opts_lay.addStretch(1)
        root.addWidget(opts)

    # drawing

    def _draw_histogram(self, counts, edges, zidx, means, mn, mx):
        if not HAS_MPL:
            return
        ax1, ax2 = self._hist_ax1, self._hist_ax2
        ax1.clear(); ax2.clear()
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL2)
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(BORDER)

        centers = (edges[:-1] + edges[1:]) / 2
        ax1.bar(centers, counts, width=(edges[1] - edges[0]),
                color=ACCENT, alpha=0.8, linewidth=0)

        if self._hist_shade_phases.get():
            try:
                vt = float(self._void_thresh_var.get())
                ax1.axvspan(mn, vt, alpha=0.12, color='blue', label='Void')
                ax1.axvline(vt, color='blue', lw=1, linestyle='--', alpha=0.7)
                agg_str = self._agg_thresh_var.get().strip()
                if agg_str:
                    at = float(agg_str)
                    ax1.axvspan(vt, at, alpha=0.10, color='green', label='Matrix')
                    ax1.axvspan(at, mx, alpha=0.10, color='orange', label='Aggregate')
                    ax1.axvline(at, color='orange', lw=1, linestyle='--', alpha=0.7)
                ax1.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER)
            except Exception:
                pass

        ax1.set_xlabel("Intensity (raw)", color=TEXT_DIM, fontsize=8)
        ax1.set_ylabel("Voxel count", color=TEXT_DIM, fontsize=8)
        ax1.set_title("Intensity Histogram", color=TEXT, fontsize=9)

        if self._hist_show_hu.get() and self._hu_cal:
            m_ = self._hu_cal.get('m', 1.0)
            c_ = self._hu_cal.get('c', 0.0)
            ax1_top = ax1.twiny()
            ax1_top.set_xlim(ax1.get_xlim())
            raw_ticks = ax1.get_xticks()
            hu_ticks = [m_ * r + c_ for r in raw_ticks]
            ax1_top.set_xticks(raw_ticks)
            ax1_top.set_xticklabels(
                [f"{h:.0f}" for h in hu_ticks], fontsize=7, color=TEAL,
            )
            ax1_top.set_xlabel("HU (calibrated)", color=TEAL, fontsize=8)

        ax2.plot(zidx, means, color=ACCENT, lw=1.5)
        ax2.fill_between(zidx, means, alpha=0.2, color=ACCENT)
        ax2.set_xlabel("Z slice index", color=TEXT_DIM, fontsize=8)
        ax2.set_ylabel("Mean intensity", color=TEXT_DIM, fontsize=8)
        ax2.set_title("Mean Intensity per Z Slice", color=TEXT, fontsize=9)

        self._hist_fig.tight_layout(pad=1.2)
        self._hist_canvas.draw()
