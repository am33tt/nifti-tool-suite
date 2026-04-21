"""Simulation Export tab — buttons + live ct.hpp preview."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext

from ..config import ACCENT, BG, BORDER, FONT_HEAD, FONT_SMALL, PANEL, TEAL, TEXT, TEXT_DIM
from .widgets import styled_btn


class ExportTabMixin:
    """Adds the Sim Export tab to :class:`NiftiApp`."""

    def _build_export_tab(self, parent):
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill='both', expand=True, padx=16, pady=12)

        tk.Label(outer, text="Simulation Export", bg=BG, fg=TEXT,
                 font=FONT_HEAD).pack(anchor='w', pady=(0, 8))
        tk.Label(outer, text="Run  ▶ Compute E-Map  first in the controls panel.",
                 bg=BG, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w')

        tk.Frame(outer, bg=BORDER, height=1).pack(fill='x', pady=8)

        for label, tip, cmd in [
            ("Export E-Map as NIfTI (.nii.gz)",
             "Float32 NIfTI — direct input to FCM / FEM pipeline",
             self._export_emap_nifti),
            ("Export E-Map as raw float32 binary",
             "C-order binary + JSON header sidecar (e.g. for custom solvers)",
             self._export_emap_raw),
            ("Export Phase Labels as NIfTI",
             "uint8 label map: 0=void, 1=matrix, 2=aggregate",
             self._export_labels_nifti),
            ("Export Phase Statistics CSV",
             "Per-phase volume fraction, HU stats, E stats",
             self._export_stats_csv),
            ("Generate ct.hpp snippet",
             "Copy-paste C++ for AMfracture / Finite Cell Method pipeline",
             self._export_ct_hpp),
        ]:
            btn_f = tk.Frame(
                outer, bg=PANEL, pady=8, padx=12,
                highlightthickness=1, highlightbackground=BORDER,
            )
            btn_f.pack(fill='x', pady=4)
            styled_btn(btn_f, label, cmd, teal=True).pack(side='left')
            tk.Label(btn_f, text=tip, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(side='left', padx=12)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill='x', pady=8)

        # ct.hpp preview
        tk.Label(outer, text="ct.hpp snippet preview:", bg=BG, fg=TEXT,
                 font=FONT_HEAD).pack(anchor='w', pady=(0, 4))
        self._cpp_text = scrolledtext.ScrolledText(
            outer, bg="#1E1E1E", fg="#D4D4D4", insertbackground=ACCENT,
            font=("Consolas", 9), relief='flat', wrap='none', height=18,
        )
        self._cpp_text.pack(fill='both', expand=True)
        self._cpp_text.insert(
            'end',
            "// Compute the E-Map first, then click 'Generate ct.hpp snippet'.\n",
        )

        btn_row = tk.Frame(outer, bg=BG, pady=4); btn_row.pack(fill='x')
        styled_btn(btn_row, "Copy to clipboard", self._copy_cpp, small=True).pack(side='left')
        styled_btn(btn_row, "Save as ct.hpp...", self._save_cpp, small=True).pack(side='left', padx=6)
