"""Left-hand controls panel.

Everything the user adjusts before running an action lives here:
viewer settings, intensity stats, HU calibration, material model,
reorientation, angle rotation, crop ROI.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from ..config import (
    ACCENT, BG, BORDER, CMAPS, CT_PRESETS, ENTRY_BG, FONT_CODE, FONT_HEAD,
    FONT_MONO, FONT_SMALL, FONT_UI, MATERIAL_PRESETS, PANEL, TEAL, TEXT, TEXT_DIM,
)
from .widgets import (
    CollapsibleSection, IntRangeRow, styled_btn, styled_entry,
)


class ControlsMixin:
    """Adds ``_build_controls`` and all its helpers to :class:`NiftiApp`."""

    def _build_controls(self, parent):
        # ── File info ─────────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "📋 Metadata / File Info")
        sec.pack(fill='x', padx=10, pady=3)
        styled_btn(sec.content, "Read Full Metadata", self._do_metadata,
                   small=True).pack(padx=10, pady=8, anchor='w')

        # ── Viewer ───────────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "🎨 Viewer Controls")
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)

        for label, var_name, default, values in [
            ("Colormap",  "_cmap_var",   "gray",        CMAPS),
            ("CT Preset", "_preset_var", "-- preset --", list(CT_PRESETS.keys())),
        ]:
            row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=10, anchor='w').pack(side='left')
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            cb = ttk.Combobox(row, textvariable=var, values=values,
                              state='readonly', width=14)
            cb.pack(side='left')
            if var_name == "_cmap_var":
                cb.bind('<<ComboboxSelected>>', lambda e: self._refresh_triplanar())
            else:
                cb.bind('<<ComboboxSelected>>', self._apply_preset)

        for label, var_name in [("Window W", "_ww_var"), ("Window C", "_wc_var")]:
            row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=10, anchor='w').pack(side='left')
            var = tk.StringVar(value="auto")
            setattr(self, var_name, var)
            styled_entry(row, textvariable=var, width=8).pack(side='left')

        btn_row = tk.Frame(inner, bg=PANEL); btn_row.pack(fill='x', pady=4)
        styled_btn(btn_row, "Apply W/C", self._apply_window_manual,
                   small=True).pack(side='left', padx=(0, 4))
        styled_btn(btn_row, "Reset Auto", self._reset_window,
                   small=True).pack(side='left')

        # ── Intensity statistics ─────────────────────────────────────────────
        sec = CollapsibleSection(parent, "📊 Intensity Statistics")
        sec.pack(fill='x', padx=10, pady=3)
        inner2 = tk.Frame(sec.content, bg=PANEL); inner2.pack(fill='x', padx=10, pady=4)
        styled_btn(inner2, "Compute Histogram", self._do_histogram,
                   small=True).pack(anchor='w', pady=(0, 4))
        self._stats_frame = tk.Frame(inner2, bg=PANEL)
        self._stats_frame.pack(fill='x')
        self._stat_labels: dict = {}
        for key in ("Min", "Max", "Mean", "Std", "p1", "p99"):
            row = tk.Frame(self._stats_frame, bg=PANEL); row.pack(fill='x', pady=1)
            tk.Label(row, text=f"{key:>4}:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=5, anchor='e').pack(side='left')
            lv = tk.StringVar(value="--")
            tk.Label(row, textvariable=lv, bg=PANEL, fg=ACCENT,
                     font=FONT_MONO, anchor='w').pack(side='left', padx=4)
            self._stat_labels[key] = lv

        # ── HU calibration ───────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "🔬 HU Calibration")
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)
        tk.Label(inner, text="Two-point linear: HU = m·raw + c",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w', pady=(0, 6))

        def _cal_row(parent, label, var_name, default, tip=""):
            r = tk.Frame(parent, bg=PANEL); r.pack(fill='x', pady=2)
            tk.Label(r, text=label, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=16, anchor='w').pack(side='left')
            v = tk.StringVar(value=str(default))
            setattr(self, var_name, v)
            styled_entry(r, textvariable=v, width=9).pack(side='left')
            if tip:
                tk.Label(r, text=tip, bg=PANEL, fg=TEXT_DIM,
                         font=FONT_SMALL).pack(side='left', padx=4)

        _cal_row(inner, "Air intensity", "_cal_air_int", "0",     "raw px = air")
        _cal_row(inner, "Ref intensity", "_cal_ref_int", "199",   "raw px = ref material")
        _cal_row(inner, "Air HU",        "_cal_air_hu",  "-1000", "HU of air")
        _cal_row(inner, "Ref HU",        "_cal_ref_hu",  "0",     "HU of ref (water=0)")

        tk.Label(inner, text="(Use histogram peak for air; known material for ref)",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL, wraplength=300,
                 justify='left').pack(anchor='w', pady=(4, 2))

        self._cal_result_var = tk.StringVar(value="Not calibrated")
        tk.Label(inner, textvariable=self._cal_result_var,
                 bg=PANEL, fg=TEAL, font=FONT_MONO).pack(anchor='w', pady=2)
        styled_btn(inner, "Apply Calibration", self._do_calibrate,
                   teal=True, small=True).pack(anchor='w', pady=4)

        # ── Material mapping ─────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "⚙️  Material Mapping  →  E [MPa]")
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)

        pr_row = tk.Frame(inner, bg=PANEL); pr_row.pack(fill='x', pady=2)
        tk.Label(pr_row, text="Preset", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=10, anchor='w').pack(side='left')
        self._mat_preset_var = tk.StringVar(value="AM Concrete (default)")
        pr_cb = ttk.Combobox(
            pr_row, textvariable=self._mat_preset_var,
            values=list(MATERIAL_PRESETS.keys()),
            state='readonly', width=24,
        )
        pr_cb.pack(side='left')
        pr_cb.bind('<<ComboboxSelected>>', self._apply_material_preset)

        mod_row = tk.Frame(inner, bg=PANEL); mod_row.pack(fill='x', pady=2)
        tk.Label(mod_row, text="Model", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=10, anchor='w').pack(side='left')
        self._model_var = tk.StringVar(value="bilinear")
        for m in ("linear", "power", "bilinear", "table"):
            tk.Radiobutton(
                mod_row, text=m, variable=self._model_var, value=m,
                bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
                activebackground=PANEL, font=FONT_SMALL,
                command=self._update_model_panel,
            ).pack(side='left', padx=3)

        self._model_params_frame = tk.Frame(inner, bg=PANEL)
        self._model_params_frame.pack(fill='x', pady=4)
        self._model_param_vars: dict = {}
        self._update_model_panel()

        vt_row = tk.Frame(inner, bg=PANEL); vt_row.pack(fill='x', pady=2)
        tk.Label(vt_row, text="Void thresh", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=12, anchor='w').pack(side='left')
        self._void_thresh_var = tk.StringVar(value="-500")
        styled_entry(vt_row, textvariable=self._void_thresh_var, width=8).pack(side='left')
        tk.Label(vt_row, text="HU", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left', padx=4)

        ag_row = tk.Frame(inner, bg=PANEL); ag_row.pack(fill='x', pady=2)
        tk.Label(ag_row, text="Aggreg. thresh", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=12, anchor='w').pack(side='left')
        self._agg_thresh_var = tk.StringVar(value="")
        styled_entry(ag_row, textvariable=self._agg_thresh_var, width=8).pack(side='left')
        tk.Label(ag_row, text="HU  (blank = 2-phase)", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left', padx=4)

        self._mat_note_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._mat_note_var,
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL,
                 wraplength=300, justify='left').pack(anchor='w', pady=(4, 2))

        styled_btn(inner, "▶  Compute E-Map", self._do_compute_emap,
                   accent=True, small=True).pack(anchor='w', pady=6)

        # E stats
        self._e_stats_frame = tk.Frame(inner, bg=PANEL)
        self._e_stats_frame.pack(fill='x', pady=2)
        self._e_stat_labels: dict = {}
        for key in ("E_min", "E_max", "E_mean", "E_std", "porosity"):
            row = tk.Frame(self._e_stats_frame, bg=PANEL); row.pack(fill='x', pady=1)
            disp = key.replace("_", " ")
            tk.Label(row, text=f"{disp:>10}:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=12, anchor='e').pack(side='left')
            lv = tk.StringVar(value="--")
            tk.Label(row, textvariable=lv, bg=PANEL, fg=TEAL,
                     font=FONT_MONO, anchor='w').pack(side='left', padx=4)
            self._e_stat_labels[key] = lv

        styled_btn(inner, "Load Preset JSON...", self._load_preset_json,
                   small=True).pack(anchor='w', pady=(8, 0))

        self._apply_material_preset()

        # ── Reorientation ────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "🔄 Reorientation  (no resample)", start_open=False)
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)
        row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
        tk.Label(row, text="Target", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        self._reorient_target = tk.StringVar(value="RAS")
        for preset in ("RAS", "LPS", "LAS"):
            styled_btn(row, preset, lambda p=preset: self._reorient_target.set(p),
                       small=True).pack(side='left', padx=2)
        row2 = tk.Frame(inner, bg=PANEL); row2.pack(fill='x', pady=2)
        tk.Label(row2, text="Custom", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        styled_entry(row2, textvariable=self._reorient_target, width=8).pack(side='left')
        tk.Label(row2, text="e.g. RAS", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side='left', padx=6)
        self._reorient_save = tk.BooleanVar(value=True)
        tk.Checkbutton(
            inner, text="Choose save path", variable=self._reorient_save,
            bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
            activebackground=PANEL, font=FONT_SMALL,
        ).pack(anchor='w')
        styled_btn(sec.content, "Run Reorientation", self._do_reorient,
                   small=True).pack(padx=10, pady=(2, 8), anchor='w')

        # ── Angle Rotation ───────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "↩  Angle Rotation  (resample)", start_open=False)
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)
        row = tk.Frame(inner, bg=PANEL); row.pack(fill='x', pady=2)
        tk.Label(row, text="Axis", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        self._rot_axis = tk.StringVar(value="z")
        for ax in ('x', 'y', 'z'):
            tk.Radiobutton(
                row, text=ax.upper(), variable=self._rot_axis, value=ax,
                bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
                activebackground=PANEL, font=FONT_UI,
            ).pack(side='left', padx=4)
        row2 = tk.Frame(inner, bg=PANEL); row2.pack(fill='x', pady=2)
        tk.Label(row2, text="Angle °", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, width=7, anchor='w').pack(side='left')
        self._rot_angle = tk.StringVar(value="45")
        styled_entry(row2, textvariable=self._rot_angle, width=7).pack(side='left', padx=4)
        for deg in (90, 180, 270):
            styled_btn(row2, str(deg), lambda d=deg: self._rot_angle.set(str(d)),
                       small=True).pack(side='left', padx=2)
        self._rot_save = tk.BooleanVar(value=True)
        tk.Checkbutton(
            inner, text="Choose save path", variable=self._rot_save,
            bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
            activebackground=PANEL, font=FONT_SMALL,
        ).pack(anchor='w')
        styled_btn(sec.content, "Run Rotation", self._do_rotate,
                   small=True).pack(padx=10, pady=(2, 8), anchor='w')

        # ── Crop ─────────────────────────────────────────────────────────────
        sec = CollapsibleSection(parent, "✂️  Crop  (ROI selection)", start_open=False)
        sec.pack(fill='x', padx=10, pady=3)
        inner = tk.Frame(sec.content, bg=PANEL); inner.pack(fill='x', padx=10, pady=6)
        self._crop_rows: dict = {}
        for axis in ('X', 'Y', 'Z'):
            r = IntRangeRow(inner, axis); r.pack(fill='x', pady=2)
            self._crop_rows[axis] = r
        btn_row2 = tk.Frame(inner, bg=PANEL); btn_row2.pack(fill='x', pady=4)
        styled_btn(btn_row2, "Auto-fill from shape", self._autofill_crop,
                   small=True).pack(side='left')
        self._crop_save = tk.BooleanVar(value=True)
        tk.Checkbutton(
            inner, text="Choose save path", variable=self._crop_save,
            bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG,
            activebackground=PANEL, font=FONT_SMALL,
        ).pack(anchor='w', pady=2)
        styled_btn(sec.content, "Run Crop", self._do_crop,
                   small=True).pack(padx=10, pady=(2, 8), anchor='w')

        tk.Frame(parent, bg=BG, height=20).pack()

    # ── Model params panel ───────────────────────────────────────────────────

    def _update_model_panel(self):
        for w in self._model_params_frame.winfo_children():
            w.destroy()
        self._model_param_vars.clear()
        model = self._model_var.get()
        f = self._model_params_frame

        def param_row(lbl_text, key, default, unit=""):
            r = tk.Frame(f, bg=PANEL); r.pack(fill='x', pady=2)
            tk.Label(r, text=lbl_text, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=14, anchor='w').pack(side='left')
            v = tk.StringVar(value=str(default))
            self._model_param_vars[key] = v
            styled_entry(r, textvariable=v, width=10).pack(side='left')
            if unit:
                tk.Label(r, text=unit, bg=PANEL, fg=TEXT_DIM,
                         font=FONT_SMALL).pack(side='left', padx=4)

        if model == "linear":
            tk.Label(f, text="E = a·HU + b  [MPa]", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(anchor='w')
            param_row("a  (slope)",     "a",      "20.0",  "MPa/HU")
            param_row("b  (intercept)", "b",      "0.0",   "MPa")
            param_row("E_void",         "E_void", "0.001", "MPa")

        elif model == "power":
            tk.Label(f, text="E = a · HU^b  [MPa]", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(anchor='w')
            param_row("a  (scale)",    "a",            "0.09",  "MPa")
            param_row("b  (exponent)", "b",            "1.92")
            param_row("HU min clamp",  "hu_min_clamp", "1.0",   "HU")
            param_row("E_void",        "E_void",       "0.001", "MPa")

        elif model == "bilinear":
            tk.Label(f, text="Piecewise: void | solid", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).pack(anchor='w')
            param_row("HU threshold", "hu_thresh", "-200",    "HU")
            param_row("E_void",       "E_void",    "0.001",   "MPa")
            param_row("E_solid",      "E_solid",   "30000.0", "MPa")

        elif model == "table":
            tk.Label(f, text="Piecewise-linear table\nEnter HU values:",
                     bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w')
            self._table_text = scrolledtext.ScrolledText(
                f, bg=ENTRY_BG, fg=TEXT, font=FONT_CODE,
                width=32, height=6, relief='flat',
            )
            self._table_text.pack(fill='x', pady=4)
            self._table_text.insert(
                'end',
                "# HU_value  E_MPa\n"
                "-1000       0.001\n"
                " -500       0.001\n"
                "    0    5000.0\n"
                "  500   25000.0\n"
                " 1000   40000.0\n",
            )
            tk.Label(f, text="(Lines starting with # are comments)",
                     bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor='w')

    def _get_model_params(self):
        model = self._model_var.get()
        vars_ = self._model_param_vars
        try:
            if model == "linear":
                return {
                    "a": float(vars_["a"].get()),
                    "b": float(vars_["b"].get()),
                    "E_void": float(vars_["E_void"].get()),
                }
            if model == "power":
                return {
                    "a": float(vars_["a"].get()),
                    "b": float(vars_["b"].get()),
                    "hu_min_clamp": float(vars_["hu_min_clamp"].get()),
                    "E_void": float(vars_["E_void"].get()),
                }
            if model == "bilinear":
                return {
                    "hu_thresh": float(vars_["hu_thresh"].get()),
                    "E_void": float(vars_["E_void"].get()),
                    "E_solid": float(vars_["E_solid"].get()),
                }
            if model == "table":
                lines = self._table_text.get('1.0', 'end').strip().splitlines()
                hu_vals: list[float] = []
                E_vals:  list[float] = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    hu_vals.append(float(parts[0]))
                    E_vals.append(float(parts[1]))
                return {"hu": hu_vals, "E": E_vals}
        except Exception as ex:
            raise ValueError(f"Bad parameter input: {ex}")

    def _apply_material_preset(self, event=None):
        name = self._mat_preset_var.get()
        preset = MATERIAL_PRESETS.get(name)
        if not preset:
            return
        self._model_var.set(preset.get("model", "linear"))
        self._void_thresh_var.set(str(preset.get("void_thresh", "-500")))
        self._mat_note_var.set(preset.get("notes", ""))
        self._update_model_panel()
        params = preset.get("params", {})
        for k, v in params.items():
            if k in self._model_param_vars:
                self._model_param_vars[k].set(str(v))
        self._cal_air_int.set(str(preset.get("hu_air", "0")))
        self._cal_ref_int.set(str(preset.get("hu_ref", "199")))

    def _load_preset_json(self):
        path = filedialog.askopenfilename(
            title="Load material preset JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            MATERIAL_PRESETS.update(data)
            self._append_log(
                f"  Loaded {len(data)} preset(s) from {Path(path).name}", 'ok',
            )
        except Exception as ex:
            messagebox.showerror("JSON Error", str(ex))

    # ── Windowing helpers ────────────────────────────────────────────────────

    def _apply_preset(self, event=None):
        name = self._preset_var.get()
        val = CT_PRESETS.get(name)
        if val is None:
            return
        ww, wc = val
        self._ww_var.set(str(ww)); self._wc_var.set(str(wc))
        self._ww = float(ww);      self._wc = float(wc)
        self._refresh_triplanar()

    def _apply_window_manual(self):
        try:
            self._ww = float(self._ww_var.get())
            self._wc = float(self._wc_var.get())
        except ValueError:
            self._ww = self._wc = None
            self._ww_var.set("auto"); self._wc_var.set("auto")
        self._refresh_triplanar()

    def _reset_window(self):
        self._ww = self._wc = None
        self._ww_var.set("auto"); self._wc_var.set("auto")
        self._preset_var.set("-- preset --")
        self._refresh_triplanar()

    # ── Crop autofill ────────────────────────────────────────────────────────

    def _autofill_crop(self):
        if self._img is None:
            return
        for axis, dim in zip(('X', 'Y', 'Z'), self._img.shape[:3]):
            self._crop_rows[axis].start_var.set("0")
            self._crop_rows[axis].end_var.set(str(dim))
