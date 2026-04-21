"""E-Map viewer tab.

Three-panel tri-planar viewer over the Young's modulus field, plus an
optional overlay on the underlying CT intensity and a per-axial-slice E
histogram.  Like the main tri-planar tab, the image artists are created
once and updated with :meth:`~matplotlib.image.AxesImage.set_data` so
slider drags stay smooth on large volumes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..config import (
    ACCENT, BG, BORDER, E_CMAPS, ENTRY_BG, ERR, FONT_MONO,
    FONT_SMALL, FONT_UI, PANEL2, SLIDER_DEBOUNCE_MS, TEAL, TEXT, TEXT_DIM,
)
from ..core.windowing import apply_window, auto_window
from ..deps import HAS_MPL, mcolors, np
from .widgets import styled_btn, styled_entry


class EmapTabMixin:
    """Adds the E-Map Viewer tab to :class:`NiftiApp`."""

    def _build_emap_tab(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="matplotlib not installed",
                     bg=BG, fg=ERR, font=FONT_UI).pack(expand=True)
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk,
        )

        ctrl = tk.Frame(parent, bg=BG, pady=4); ctrl.pack(fill='x', padx=8)
        tk.Label(ctrl, text="E colormap:", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left')
        self._ecmap_var = tk.StringVar(value="viridis")
        ecm_cb = ttk.Combobox(
            ctrl, textvariable=self._ecmap_var,
            values=E_CMAPS, state='readonly', width=12,
        )
        ecm_cb.pack(side='left', padx=4)
        ecm_cb.bind('<<ComboboxSelected>>', lambda e: self._refresh_emap_viewer())

        self._emap_overlay = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ctrl, text="Overlay on CT", variable=self._emap_overlay,
            bg=BG, fg=TEXT, selectcolor=ENTRY_BG,
            activebackground=BG, font=FONT_SMALL,
            command=self._refresh_emap_viewer,
        ).pack(side='left', padx=8)
        tk.Label(ctrl, text="Alpha:", bg=BG, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left')
        self._emap_alpha_var = tk.StringVar(value="0.6")
        styled_entry(ctrl, textvariable=self._emap_alpha_var, width=5).pack(side='left', padx=4)
        styled_btn(ctrl, "Refresh", self._refresh_emap_viewer, small=True).pack(side='left', padx=8)

        self._emap_fig = Figure(facecolor=BG)
        self._emap_axes = self._emap_fig.subplots(1, 3)
        for ax in self._emap_axes:
            ax.set_facecolor(PANEL2)
            ax.axis('off')
        self._emap_fig.tight_layout(pad=0.4)
        self._emap_canvas = FigureCanvasTkAgg(self._emap_fig, master=parent)
        self._emap_canvas.get_tk_widget().pack(fill='both', expand=True)
        tb = tk.Frame(parent, bg=BG); tb.pack(fill='x')
        NavigationToolbar2Tk(self._emap_canvas, tb)

        # Slider bar
        slider_bar = tk.Frame(parent, bg=PANEL2, pady=4); slider_bar.pack(fill='x')
        self._emap_sliders:  dict = {}
        self._emap_idx_vars: dict = {}
        self._emap_idx = {'X': 0, 'Y': 0, 'Z': 0}
        axis_info = [
            ('X', "Sagittal (X)"),
            ('Y', "Coronal  (Y)"),
            ('Z', "Axial    (Z)"),
        ]
        for col, (ax, label) in enumerate(axis_info):
            col_frame = tk.Frame(slider_bar, bg=PANEL2)
            col_frame.grid(row=0, column=col, sticky='ew', padx=8, pady=2)
            slider_bar.columnconfigure(col, weight=1)
            tk.Label(col_frame, text=label, bg=PANEL2, fg=TEAL,
                     font=FONT_SMALL).pack(anchor='w')
            row_f = tk.Frame(col_frame, bg=PANEL2); row_f.pack(fill='x')
            idx_var = tk.StringVar(value="--")
            self._emap_idx_vars[ax] = idx_var
            sl = tk.Scale(
                row_f, from_=0, to=100, orient='horizontal',
                bg=PANEL2, fg=TEXT, troughcolor=BORDER,
                highlightthickness=0, bd=0, showvalue=False,
                command=lambda val, a=ax: self._on_emap_drag(a, val),
            )
            sl.pack(side='left', fill='x', expand=True)
            tk.Label(row_f, textvariable=idx_var, bg=PANEL2, fg=TEAL,
                     font=FONT_MONO, width=5).pack(side='left')
            self._emap_sliders[ax] = sl

        # Per-slice histogram
        self._emap_slice_fig = Figure(figsize=(4, 1.8), facecolor=BG)
        self._emap_slice_ax  = self._emap_slice_fig.add_subplot(111)
        self._emap_slice_ax.set_facecolor(PANEL2)
        self._emap_slice_canvas = FigureCanvasTkAgg(self._emap_slice_fig, master=parent)
        self._emap_slice_canvas.get_tk_widget().pack(fill='x', padx=8, pady=4)

        # Debounce
        self._emap_redraw_pending = None

    # ── interaction ──────────────────────────────────────────────────────────

    def _on_emap_drag(self, axis, val):
        idx = int(float(val))
        self._emap_idx[axis] = idx
        self._emap_idx_vars[axis].set(str(idx))
        if self._E_map is None:
            return
        if self._emap_redraw_pending is not None:
            try:
                self.after_cancel(self._emap_redraw_pending)
            except Exception:
                pass
        self._emap_redraw_pending = self.after(
            SLIDER_DEBOUNCE_MS, self._refresh_emap_viewer,
        )

    def _update_emap_sliders(self):
        if self._E_map is None:
            return
        for ax, dim in zip(('X', 'Y', 'Z'), self._E_map.shape[:3]):
            mid = dim // 2
            self._emap_sliders[ax].config(from_=0, to=dim - 1)
            self._emap_sliders[ax].set(mid)
            self._emap_idx[ax] = mid
            self._emap_idx_vars[ax].set(str(mid))

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
        titles   = {
            'X': f"Sagittal X={xi}",
            'Y': f"Coronal  Y={yi}",
            'Z': f"Axial    Z={zi}",
        }

        overlay = self._emap_overlay.get()
        try:
            alpha = float(self._emap_alpha_var.get())
        except Exception:
            alpha = 0.6

        # Full rebuild here: overlay toggles change the layer stack and colorbars,
        # so a simple set_data() doesn't cover all cases. Still fast enough for
        # the E-Map refresh cadence (after Compute, and on slider drag).
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
