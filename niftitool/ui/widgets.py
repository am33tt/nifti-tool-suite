"""Reusable tkinter widgets (styled button, collapsible section, progress).

Nothing here knows about the application state — all widgets are pure
presentation components.
"""

from __future__ import annotations

import tkinter as tk

from ..config import (
    ACCENT, BG, BORDER, BTN_BG, BTN_HOV, ERR, FONT_HEAD, FONT_MONO,
    FONT_SMALL, FONT_UI, PANEL, PANEL2, TEAL, TEXT, TEXT_DIM,
    ENTRY_BG,
)


# ── atoms ─────────────────────────────────────────────────────────────────────

def styled_btn(parent, text, cmd, *, accent=False, danger=False, small=False,
               teal=False, **kw):
    if teal:
        bg, fg = TEAL, "#FFFFFF"
    elif accent:
        bg, fg = ACCENT, "#FFFFFF"
    elif danger:
        bg, fg = ERR, "#FFFFFF"
    else:
        bg, fg = BTN_BG, TEXT
    font = FONT_SMALL if small else FONT_UI
    b = tk.Button(
        parent, text=text, command=cmd,
        bg=bg, fg=fg, activebackground=BTN_HOV,
        activeforeground=TEXT, relief='flat', bd=0,
        padx=10, pady=4 if small else 6,
        cursor='hand2', font=font, **kw,
    )
    hover_bg = (
        "#C96A1A" if accent
        else ERR if danger
        else TEAL if teal
        else BTN_HOV
    )
    b.bind('<Enter>', lambda e: b.config(bg=hover_bg))
    b.bind('<Leave>', lambda e: b.config(bg=bg))
    return b


def styled_entry(parent, textvariable=None, width=12):
    return tk.Entry(
        parent, textvariable=textvariable, width=width,
        bg=ENTRY_BG, fg=TEXT, insertbackground=ACCENT,
        relief='flat', bd=0, highlightthickness=1,
        highlightbackground=BORDER, highlightcolor=ACCENT,
        font=FONT_MONO,
    )


def lbl(parent, text, *, dim=False, head=False, **kw):
    fg   = TEXT_DIM if dim else TEXT
    font = FONT_HEAD if head else FONT_UI
    return tk.Label(parent, text=text, bg=PANEL, fg=fg, font=font, **kw)


# ── composite widgets ────────────────────────────────────────────────────────

class CollapsibleSection(tk.Frame):
    """Click-to-toggle panel with an accent-coloured header."""

    def __init__(self, parent, title, start_open=True, **kw):
        super().__init__(parent, bg=BG, pady=3, **kw)
        self._open  = tk.BooleanVar(value=start_open)
        self._title = title
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill='x')
        self._toggle_btn = tk.Label(
            hdr, text=self._header_text(), bg=BG, fg=ACCENT,
            font=FONT_HEAD, cursor='hand2', anchor='w',
        )
        self._toggle_btn.pack(fill='x')
        self._toggle_btn.bind('<Button-1>', lambda e: self.toggle())
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x', pady=(0, 3))
        self.content = tk.Frame(
            self, bg=PANEL, bd=0,
            highlightthickness=1, highlightbackground=BORDER,
        )
        if start_open:
            self.content.pack(fill='x')

    def _header_text(self):
        return f" {'▼' if self._open.get() else '►'} {self._title}"

    def toggle(self):
        self._open.set(not self._open.get())
        self._toggle_btn.config(text=self._header_text())
        if self._open.get():
            self.content.pack(fill='x')
        else:
            self.content.forget()

    def open(self):
        if not self._open.get():
            self.toggle()


class IntRangeRow(tk.Frame):
    """Axis-labelled start/end integer entry row (used by the crop panel)."""

    def __init__(self, parent, axis_label, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.start_var = tk.StringVar(value="0")
        self.end_var   = tk.StringVar(value="0")
        tk.Label(
            self, text=axis_label, bg=PANEL, fg=ACCENT,
            font=("Consolas", 10, "bold"), width=3,
        ).pack(side='left', padx=(8, 4))
        tk.Label(self, text="start", bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side='left')
        styled_entry(self, textvariable=self.start_var, width=6).pack(side='left', padx=(2, 8))
        tk.Label(self, text="end", bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side='left')
        styled_entry(self, textvariable=self.end_var, width=6).pack(side='left', padx=(2, 4))


class StageProgressBar(tk.Frame):
    """Six-stage load indicator that doubles as an indeterminate spinner."""

    STAGES = [
        ("Opening file",      0.10),
        ("Decoding voxels",   0.40),
        ("Greyscale convert", 0.65),
        ("Updating sliders",  0.80),
        ("Computing stats",   0.92),
        ("Ready",             1.00),
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=PANEL2, **kw)
        self._stage_idx = -1
        self._mode      = 'hidden'
        self._spin_pos  = 0.0
        self._label = tk.Label(
            self, text="", bg=PANEL2, fg=TEXT_DIM,
            font=FONT_SMALL, width=18, anchor='w',
        )
        self._label.pack(side='left', padx=(6, 2))
        self._canvas = tk.Canvas(
            self, height=14, width=160, bg=PANEL2,
            bd=0, highlightthickness=0,
        )
        self._canvas.pack(side='left', padx=(0, 6))
        self._canvas.bind('<Configure>', self._redraw)

    # ── public API ───────────────────────────────────────────────────────────

    def begin_staged(self):
        self._stage_idx = 0
        self._mode = 'stage'
        self.pack(side='right', padx=4)
        self._draw_stage()

    def advance(self):
        if self._stage_idx < len(self.STAGES) - 1:
            self._stage_idx += 1
        self._draw_stage()

    def done(self):
        self._stage_idx = len(self.STAGES) - 1
        self._draw_stage()
        self.after(900, self._hide)

    def start(self, interval=12):
        self._mode = 'spinner'
        self._spin_pos = 0.0
        self.pack(side='right', padx=4)
        self._tick_spinner()

    def stop(self):
        self._hide()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _hide(self):
        self._mode = 'hidden'
        self._label.config(text="")
        self.pack_forget()

    def _draw_stage(self):
        if self._stage_idx < 0 or self._stage_idx >= len(self.STAGES):
            return
        name, frac = self.STAGES[self._stage_idx]
        self._label.config(text=name)
        self._draw_bar(frac)

    def _draw_bar(self, frac):
        c = self._canvas
        w = c.winfo_width() or 160
        h = 14
        c.delete('all')
        r = 5
        c.create_rectangle(r, 2, w - r, h - 2, fill=BORDER, outline='', width=0)
        fill_w = max(0, int(frac * w))
        if fill_w > r:
            c.create_rectangle(r, 2, fill_w, h - 2, fill=ACCENT, outline='', width=0)
        c.create_text(
            w // 2, h // 2, text=f"{int(frac * 100)}%",
            fill=TEXT if frac > 0.5 else TEXT_DIM, font=FONT_SMALL,
        )

    def _redraw(self, event=None):
        if self._mode == 'stage':
            self._draw_stage()
        elif self._mode == 'spinner':
            self._draw_bar(self._spin_pos)

    def _tick_spinner(self):
        if self._mode != 'spinner':
            return
        self._spin_pos = (self._spin_pos + 0.05) % 1.05
        self._draw_bar(min(self._spin_pos, 1.0))
        self.after(60, self._tick_spinner)
