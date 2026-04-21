"""Log pane + thread-safe logging helpers."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext

from ..config import ACCENT, ACCENT2, BG, ENTRY_BG, ERR, FONT_HEAD, FONT_MONO, TEAL, TEXT, TEXT_DIM, WARN
from .widgets import styled_btn


class LogTabMixin:
    """Log pane + :meth:`_append_log`, :meth:`_set_status`, etc."""

    def _build_log(self, parent):
        top_row = tk.Frame(parent, bg=BG); top_row.pack(fill='x', padx=6, pady=(6, 0))
        tk.Label(top_row, text="Output / Log", bg=BG, fg=TEXT,
                 font=FONT_HEAD).pack(side='left')
        styled_btn(top_row, "Clear", self._clear_log, small=True).pack(side='right')
        self._log_widget = scrolledtext.ScrolledText(
            parent, bg=ENTRY_BG, fg=TEXT, insertbackground=ACCENT,
            font=FONT_MONO, relief='flat', wrap='word', state='disabled',
        )
        self._log_widget.pack(fill='both', expand=True, padx=6, pady=6)
        self._log_widget.tag_config('accent', foreground=ACCENT)
        self._log_widget.tag_config('ok',     foreground=ACCENT2)
        self._log_widget.tag_config('warn',   foreground=WARN)
        self._log_widget.tag_config('err',    foreground=ERR)
        self._log_widget.tag_config('dim',    foreground=TEXT_DIM)
        self._log_widget.tag_config('teal',   foreground=TEAL)

    # ── thread-safe public API ───────────────────────────────────────────────

    def _clear_log(self):
        self._log_widget.config(state='normal')
        self._log_widget.delete('1.0', 'end')
        self._log_widget.config(state='disabled')

    def _append_log(self, text: str, tag: str = ''):
        self.after(0, self._do_append_log, text, tag)

    def _do_append_log(self, text: str, tag: str):
        self._log_widget.config(state='normal')
        self._log_widget.insert('end', text + "\n", tag)
        self._log_widget.see('end')
        self._log_widget.config(state='disabled')

    def _log_sep(self, title: str = ''):
        line = f"── {title} {'─' * max(0, 50 - len(title))}"
        self._append_log("", '')
        self._append_log(line, 'accent')

    def _set_status(self, msg: str, busy: bool = False):
        self.after(0, self._do_set_status, msg, busy)

    def _do_set_status(self, msg: str, busy: bool):
        self._status_var.set(msg)
        if busy:
            self._prog.start()
        else:
            self._prog.stop()
