"""Tri-planar (sagittal / coronal / axial) viewer.

Performance notes
-----------------
This mixin is intentionally written to minimise per-frame work:

* The three :class:`~matplotlib.image.AxesImage` handles are created
  once (``imshow``) and subsequent frames update them via
  :meth:`AxesImage.set_data` — no ``ax.clear()`` → no rebuild of ticks,
  spines, compass text, colorbar, legend, etc.
* Windowed slices come from :class:`SliceCache` so repeated visits to the
  same slice index with the same ``(ww, wc)`` are free.
* Slider ``command`` callbacks schedule a redraw through a single
  :meth:`tk.after` timer and cancel any pending redraw first — so rapid
  drags coalesce into a single frame.
* ``tight_layout`` is run once on build, not every frame.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from ..config import (
    ACCENT, BG, BORDER, ERR, FONT_MONO, FONT_SMALL, FONT_UI, PANEL2,
    SLIDER_DEBOUNCE_MS, TEXT, TEXT_DIM,
)
from ..core.windowing import apply_window, auto_window
from ..deps import HAS_MPL, np
from .widgets import styled_btn


class TriplanarMixin:
    """Adds the Tri-Planar tab to :class:`NiftiApp`.

    Required state on ``self`` (set by :class:`NiftiApp.__init__`):
      ``_gray``, ``_hu_vol``, ``_E_map``, ``_ww``, ``_wc``,
      ``_axis_labels``, ``_slice_cache``.
    """

    # ── build ────────────────────────────────────────────────────────────────

    def _build_triplanar(self, parent):
        if not HAS_MPL:
            tk.Label(
                parent, text="matplotlib not installed",
                bg=BG, fg=ERR, font=FONT_UI,
            ).pack(expand=True)
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk,
        )

        self._tri_fig = Figure(facecolor=BG)
        axes = self._tri_fig.subplots(1, 3)
        self._ax_sag, self._ax_cor, self._ax_axi = axes
        for ax in axes:
            ax.set_facecolor(PANEL2)
            ax.axis('off')

        # Persistent matplotlib artists (created once, updated each frame)
        self._tri_im:       dict = {'X': None, 'Y': None, 'Z': None}
        self._tri_title:    dict = {}
        self._tri_hline:    dict = {}
        self._tri_vline:    dict = {}
        self._tri_compass:  dict = {}

        self._tri_fig.tight_layout(pad=0.4)
        self._tri_canvas = FigureCanvasTkAgg(self._tri_fig, master=parent)
        self._tri_canvas.get_tk_widget().pack(fill='both', expand=True)
        self._tri_canvas.mpl_connect('button_press_event', self._on_tri_click)

        tb_frame = tk.Frame(parent, bg=BG)
        tb_frame.pack(fill='x')
        NavigationToolbar2Tk(self._tri_canvas, tb_frame)

        # ── slider bar ──
        slider_bar = tk.Frame(parent, bg=PANEL2, pady=4)
        slider_bar.pack(fill='x')
        self._tri_sliders:  dict = {}
        self._tri_idx_vars: dict = {}
        self._tri_idx = {'X': 0, 'Y': 0, 'Z': 0}
        axis_info = [
            ('X', "Sagittal (X)"),
            ('Y', "Coronal  (Y)"),
            ('Z', "Axial    (Z)"),
        ]
        for col, (ax, label) in enumerate(axis_info):
            col_frame = tk.Frame(slider_bar, bg=PANEL2)
            col_frame.grid(row=0, column=col, sticky='ew', padx=8, pady=2)
            slider_bar.columnconfigure(col, weight=1)
            tk.Label(col_frame, text=label, bg=PANEL2, fg=ACCENT,
                     font=FONT_SMALL).pack(anchor='w')
            row_f = tk.Frame(col_frame, bg=PANEL2)
            row_f.pack(fill='x')
            idx_var = tk.StringVar(value="--")
            self._tri_idx_vars[ax] = idx_var
            sl = tk.Scale(
                row_f, from_=0, to=100, orient='horizontal',
                bg=PANEL2, fg=TEXT, troughcolor=BORDER,
                highlightthickness=0, bd=0, showvalue=False,
                command=lambda val, a=ax: self._on_tri_drag(a, val),
            )
            sl.pack(side='left', fill='x', expand=True)
            tk.Label(row_f, textvariable=idx_var, bg=PANEL2, fg=ACCENT,
                     font=FONT_MONO, width=5).pack(side='left')
            self._tri_sliders[ax] = sl

        # ── export buttons ──
        export_bar = tk.Frame(parent, bg=BG, pady=2)
        export_bar.pack(fill='x')
        for axis in ('Z', 'Y', 'X'):
            name = {'Z': 'Axial', 'Y': 'Coronal', 'X': 'Sagittal'}[axis]
            styled_btn(
                export_bar, f"Export {name} PNG",
                lambda a=axis: self._export_slice(a),
                small=True,
            ).pack(side='left', padx=4)

        # Debounce state
        self._tri_redraw_pending = None

    # ── slider / click handlers ──────────────────────────────────────────────

    def _on_tri_drag(self, axis, val):
        idx = int(float(val))
        self._tri_idx[axis] = idx
        self._tri_idx_vars[axis].set(str(idx))
        if self._gray is None:
            return
        # Debounce: collapse a burst of slider events into at most one redraw
        # per SLIDER_DEBOUNCE_MS window.
        if self._tri_redraw_pending is not None:
            try:
                self.after_cancel(self._tri_redraw_pending)
            except Exception:
                pass
        self._tri_redraw_pending = self.after(
            SLIDER_DEBOUNCE_MS, self._refresh_triplanar,
        )

    def _on_tri_click(self, event):
        if self._gray is None or event.inaxes is None:
            return
        ax_map = {self._ax_sag: 'X', self._ax_cor: 'Y', self._ax_axi: 'Z'}
        if event.inaxes not in ax_map:
            return
        axis = ax_map[event.inaxes]
        ex, ey = int(event.xdata or 0), int(event.ydata or 0)
        g  = self._gray
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

        # Pull windowed slices from the cache — this is the whole point of it.
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
                # First frame for this axis, or shape changed (new volume).
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
                                     ha='left',  fontweight='bold'),
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
                # Hot path: just update the image buffer and text.
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
            self._tri_sliders[ax].config(from_=0, to=dim - 1)
            self._tri_sliders[ax].set(mid)
            self._tri_idx[ax] = mid
            self._tri_idx_vars[ax].set(str(mid))

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
            messagebox.showwarning("No data", "Load a file first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save slice as PNG", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
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
