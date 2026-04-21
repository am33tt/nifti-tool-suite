"""Main application window.

:class:`NiftiApp` is the thin composition layer — it owns the shared
volume state and wires the individual tab mixins together.  Each mixin
lives in its own file so behaviour for a given tab can be edited (and
bugs tracked) in isolation.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..config import (
    ACCENT, BG, BORDER, FONT_MONO, FONT_SMALL, FONT_TITLE, PANEL, PANEL2,
    TEXT, TEXT_DIM,
)
from ..core.slice_cache import SliceCache
from ..deps import missing_report
from ..utils import process_rss_mb
from .actions import ActionsMixin
from .controls import ControlsMixin
from .emap_tab import EmapTabMixin
from .export_tab import ExportTabMixin
from .histogram_tab import HistogramTabMixin
from .log_tab import LogTabMixin
from .triplanar_tab import TriplanarMixin
from .view3d_tab import View3DMixin
from .widgets import StageProgressBar, styled_btn


class NiftiApp(
    ControlsMixin,
    TriplanarMixin,
    View3DMixin,
    HistogramTabMixin,
    EmapTabMixin,
    ExportTabMixin,
    LogTabMixin,
    ActionsMixin,
    tk.Tk,
):
    """Top-level tkinter window.

    All mutable volume state lives on ``self`` so every mixin can reach
    it directly (load → ``_img`` / ``_gray``; calibrate → ``_hu_vol``;
    compute E-Map → ``_E_map`` / ``_labels`` / ``_E_stats``; etc.).
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("NIfTI Tool Suite  v6  —  SimReady")
        self.configure(bg=BG)
        self.geometry("1600x1000")
        self.minsize(1200, 750)

        # ── volume state ─────────────────────────────────────────────────────
        self._img         = None
        self._gray        = None
        self._path        = None
        self._busy        = False
        self._axis_labels = {'x': ('X', '-'), 'y': ('Y', '-'), 'z': ('Z', '-')}
        self._ww          = None
        self._wc          = None

        # ── material state ───────────────────────────────────────────────────
        self._hu_vol   = None
        self._E_map    = None
        self._labels   = None
        self._hu_cal:  dict = {}
        self._E_stats: dict = {}
        self._porosity = 0.0

        # ── perf aids ────────────────────────────────────────────────────────
        self._slice_cache = SliceCache()

        # ── build ────────────────────────────────────────────────────────────
        self._build_ui()
        self._check_deps()
        self._setup_dnd()
        self.bind_all('<Control-o>', lambda e: self._open_file())
        self._start_ram_monitor()

    # ── tab switching ────────────────────────────────────────────────────────

    def _on_tab_changed(self, _event):
        """Fire deferred work when the user switches tabs."""
        try:
            current = self._nb.select()
        except Exception:
            return
        if current == str(self._viewer3d_tab):
            fn = getattr(self, 'maybe_auto_render_3d', None)
            if callable(fn):
                fn()

    # ── drag-and-drop ────────────────────────────────────────────────────────

    def _setup_dnd(self):
        try:
            import tkinterdnd2  # noqa: F401
            self.drop_target_register('DND_Files')
            self.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        path = event.data.strip().strip('{}')
        if path:
            self._load_path(path)

    # ── dependency check ─────────────────────────────────────────────────────

    def _check_deps(self):
        missing = missing_report()
        if missing:
            self._append_log(
                f"  Missing packages: {', '.join(missing)}\n"
                f"  Install: pip install nibabel numpy scipy matplotlib psutil scikit-image\n",
                'warn',
            )

    # ── RAM monitor ──────────────────────────────────────────────────────────

    def _start_ram_monitor(self):
        self._update_ram()

    def _update_ram(self):
        mb = process_rss_mb()
        if mb is not None:
            self._ram_var.set(f"RAM  {mb:.0f} MiB")
        else:
            self._ram_var.set("RAM  — (pip install psutil)")
        self.after(2000, self._update_ram)

    # ── UI skeleton ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=BG, pady=8); top.pack(fill='x', padx=14)
        tk.Label(top, text="NIfTI Tool Suite", bg=BG, fg=TEXT,
                 font=FONT_TITLE).pack(side='left')
        tk.Label(top, text="v6  SimReady  |  Ctrl+O to open",
                 bg=BG, fg=TEXT_DIM, font=FONT_SMALL).pack(
                     side='left', padx=12, pady=2,
                 )
        btn_row = tk.Frame(top, bg=BG); btn_row.pack(side='right')
        styled_btn(btn_row, "Open NIfTI...", self._open_file,
                   accent=True).pack(side='left', padx=4)
        styled_btn(btn_row, "Gunzip .gz...", self._do_gunzip).pack(side='left')

        # File info bar
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x')
        self._info_var = tk.StringVar(
            value="No file loaded  ─  open a .nii or .nii.gz file",
        )
        inf = tk.Frame(self, bg=PANEL2, pady=5); inf.pack(fill='x')
        tk.Label(inf, textvariable=self._info_var, bg=PANEL2, fg=TEXT_DIM,
                 font=FONT_MONO, anchor='w').pack(side='left', padx=10)

        # Main paned
        paned = tk.PanedWindow(self, orient='horizontal', bg=BG,
                               sashwidth=6, sashrelief='flat', sashpad=2)
        paned.pack(fill='both', expand=True)

        ctrl_outer = tk.Frame(paned, bg=BG, width=380)
        paned.add(ctrl_outer, minsize=320)

        ctrl_canvas = tk.Canvas(ctrl_outer, bg=BG, bd=0, highlightthickness=0)
        ctrl_scroll = ttk.Scrollbar(
            ctrl_outer, orient='vertical', command=ctrl_canvas.yview,
        )
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_scroll.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        self._ctrl_frame = tk.Frame(ctrl_canvas, bg=BG)
        self._ctrl_win_id = ctrl_canvas.create_window(
            (0, 0), window=self._ctrl_frame, anchor='nw',
        )
        self._ctrl_frame.bind(
            '<Configure>',
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')),
        )
        ctrl_canvas.bind(
            '<Configure>',
            lambda e: ctrl_canvas.itemconfig(self._ctrl_win_id, width=e.width),
        )
        ctrl_canvas.bind(
            '<MouseWheel>',
            lambda e: ctrl_canvas.yview_scroll(-1 * (e.delta // 120), 'units'),
        )
        self._ctrl_canvas = ctrl_canvas

        self._build_controls(self._ctrl_frame)

        # Right notebook
        right = tk.Frame(paned, bg=BG)
        paned.add(right, minsize=700)

        style = ttk.Style()
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure(
            'TNotebook.Tab',
            background=PANEL2, foreground=TEXT_DIM,
            padding=[10, 4], font=FONT_SMALL,
        )
        style.map(
            'TNotebook.Tab',
            background=[('selected', PANEL), ('active', BORDER)],
            foreground=[('selected', ACCENT)],
        )

        nb = ttk.Notebook(right); nb.pack(fill='both', expand=True)

        self._triplanar_tab = tk.Frame(nb, bg=BG)
        self._viewer3d_tab  = tk.Frame(nb, bg=BG)
        self._histogram_tab = tk.Frame(nb, bg=BG)
        self._emap_tab      = tk.Frame(nb, bg=BG)
        self._export_tab    = tk.Frame(nb, bg=BG)
        self._log_tab       = tk.Frame(nb, bg=BG)

        nb.add(self._triplanar_tab, text="  Tri-Planar  ")
        nb.add(self._viewer3d_tab,  text="  3-D View  ")
        nb.add(self._histogram_tab, text="  Histogram  ")
        nb.add(self._emap_tab,      text="  E-Map Viewer  ")
        nb.add(self._export_tab,    text="  Sim Export  ")
        nb.add(self._log_tab,       text="  Log  ")

        self._nb = nb

        self._build_triplanar(self._triplanar_tab)
        self._build_3d_view(self._viewer3d_tab)
        # Auto-render the 3-D view the first time the user looks at it.
        nb.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        self._build_histogram_tab(self._histogram_tab)
        self._build_emap_tab(self._emap_tab)
        self._build_export_tab(self._export_tab)
        self._build_log(self._log_tab)

        # Status bar
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x', side='bottom')
        status_bar = tk.Frame(self, bg=PANEL2, pady=4); status_bar.pack(fill='x', side='bottom')
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=PANEL2, fg=TEXT_DIM, font=FONT_SMALL, anchor='w').pack(
                     side='left', padx=10,
                 )
        self._probe_var = tk.StringVar(value="")
        tk.Label(status_bar, textvariable=self._probe_var,
                 bg=PANEL2, fg=ACCENT, font=FONT_MONO, anchor='e').pack(
                     side='left', padx=14,
                 )
        self._ram_var = tk.StringVar(value="RAM  --")
        tk.Label(status_bar, textvariable=self._ram_var,
                 bg=PANEL2, fg=TEXT_DIM, font=FONT_SMALL, anchor='e').pack(
                     side='right', padx=16,
                 )
        tk.Frame(status_bar, bg=BORDER, width=1).pack(side='right', fill='y', pady=2)
        self._prog = StageProgressBar(status_bar)
