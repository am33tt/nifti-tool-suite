"""Histogram tab: intensity distribution + per-slice mean curve."""

from __future__ import annotations

import tkinter as tk

from ..config import ACCENT, BG, BORDER, ENTRY_BG, ERR, FONT_SMALL, FONT_UI, PANEL2, TEAL, TEXT, TEXT_DIM
from ..deps import HAS_MPL
from .widgets import styled_btn, styled_entry


class HistogramTabMixin:
    """Adds the Histogram tab to :class:`NiftiApp`."""

    def _build_histogram_tab(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="matplotlib not installed",
                     bg=BG, fg=ERR, font=FONT_UI).pack(expand=True)
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk,
        )

        self._hist_fig = Figure(facecolor=BG)
        self._hist_ax1, self._hist_ax2 = self._hist_fig.subplots(1, 2)
        for ax in (self._hist_ax1, self._hist_ax2):
            ax.set_facecolor(PANEL2)
        self._hist_canvas = FigureCanvasTkAgg(self._hist_fig, master=parent)
        self._hist_canvas.get_tk_widget().pack(fill='both', expand=True)
        tb = tk.Frame(parent, bg=BG); tb.pack(fill='x')
        NavigationToolbar2Tk(self._hist_canvas, tb)

        opts = tk.Frame(parent, bg=BG, pady=4); opts.pack(fill='x', padx=8)
        tk.Label(opts, text="Bins:", bg=BG, fg=TEXT_DIM, font=FONT_SMALL).pack(side='left')
        self._hist_bins_var = tk.StringVar(value="256")
        styled_entry(opts, textvariable=self._hist_bins_var, width=5).pack(side='left', padx=4)
        self._hist_show_hu = tk.BooleanVar(value=True)
        tk.Checkbutton(
            opts, text="Show HU axis", variable=self._hist_show_hu,
            bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
            activebackground=BG, font=FONT_SMALL,
        ).pack(side='left', padx=8)
        self._hist_shade_phases = tk.BooleanVar(value=True)
        tk.Checkbutton(
            opts, text="Shade phases", variable=self._hist_shade_phases,
            bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
            activebackground=BG, font=FONT_SMALL,
        ).pack(side='left', padx=4)
        styled_btn(opts, "Re-run", self._do_histogram, small=True).pack(side='left', padx=8)

    # ── drawing ──────────────────────────────────────────────────────────────

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
        ax1.set_ylabel("Voxel count",    color=TEXT_DIM, fontsize=8)
        ax1.set_title("Intensity Histogram", color=TEXT, fontsize=9)

        if self._hist_show_hu.get() and self._hu_cal:
            m_ = self._hu_cal.get('m', 1.0)
            c_ = self._hu_cal.get('c', 0.0)
            ax1_top = ax1.twiny()
            ax1_top.set_xlim(ax1.get_xlim())
            raw_ticks = ax1.get_xticks()
            hu_ticks  = [m_ * r + c_ for r in raw_ticks]
            ax1_top.set_xticks(raw_ticks)
            ax1_top.set_xticklabels(
                [f"{h:.0f}" for h in hu_ticks], fontsize=7, color=TEAL,
            )
            ax1_top.set_xlabel("HU (calibrated)", color=TEAL, fontsize=8)

        ax2.plot(zidx, means, color=ACCENT, lw=1.5)
        ax2.fill_between(zidx, means, alpha=0.2, color=ACCENT)
        ax2.set_xlabel("Z slice index",  color=TEXT_DIM, fontsize=8)
        ax2.set_ylabel("Mean intensity", color=TEXT_DIM, fontsize=8)
        ax2.set_title("Mean Intensity per Z Slice", color=TEXT, fontsize=9)

        self._hist_fig.tight_layout(pad=1.2)
        self._hist_canvas.draw()
