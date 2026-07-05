"""Log pane + thread-safe logging helpers."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, ACCENT2, BG, ENTRY_BG, ERR, TEAL, TEXT, TEXT_DIM, WARN,
)
from .widgets import styled_btn


_TAG_COLORS = {
    'accent': ACCENT,
    'ok':     ACCENT2,
    'warn':   WARN,
    'err':    ERR,
    'dim':    TEXT_DIM,
    'teal':   TEAL,
    '':       TEXT,
}


class LogTabMixin:
    """Log pane + :meth:`_append_log`, :meth:`_set_status`, etc."""

    def _build_log(self, parent):
        root = QVBoxLayout(parent)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        top_row = QWidget(parent)
        tr_lay = QHBoxLayout(top_row)
        tr_lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Output / Log", top_row)
        title.setFont(QFont("Segoe UI Semibold", 11))
        title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        tr_lay.addWidget(title)
        tr_lay.addStretch(1)
        tr_lay.addWidget(styled_btn(top_row, "Clear",
                                    self._clear_log, small=True))
        root.addWidget(top_row)

        self._log_widget = QPlainTextEdit(parent)
        self._log_widget.setReadOnly(True)
        self._log_widget.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {ENTRY_BG}; color: {TEXT}; "
            f"border: none; }}"
        )
        self._log_widget.setFont(QFont("Consolas", 10))
        self._log_widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        root.addWidget(self._log_widget, 1)

    # thread-safe public API

    def _clear_log(self):
        self._log_widget.clear()

    def _append_log(self, text: str, tag: str = ''):
        """Thread-safe.  Emits a signal that's connected to
        :meth:`_do_append_log` on the GUI thread in ``app.py``.
        """
        # ``_post_log_signal`` is defined on NiftiApp.
        try:
            self._post_log_signal.emit(text, tag)
        except Exception:
            # Fallback for the rare case we're called before signals are wired.
            self._do_append_log(text, tag)

    def _do_append_log(self, text: str, tag: str):
        color = _TAG_COLORS.get(tag, TEXT)
        cur = self._log_widget.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cur.setCharFormat(fmt)
        cur.insertText(text + "\n")
        self._log_widget.setTextCursor(cur)
        self._log_widget.ensureCursorVisible()

    def _log_sep(self, title: str = ''):
        line = f"── {title} {'─' * max(0, 50 - len(title))}"
        self._append_log("", '')
        self._append_log(line, 'accent')

    def _set_status(self, msg: str, busy: bool = False):
        """Thread-safe status-bar update."""
        try:
            self._post_status_signal.emit(msg, bool(busy))
        except Exception:
            self._do_set_status(msg, busy)

    def _do_set_status(self, msg: str, busy: bool):
        self._status_var.set(msg)
        stop_btn = getattr(self, '_stop_btn', None)
        if stop_btn is not None:
            stop_btn.setVisible(bool(busy))
        if busy:
            self._prog.start()
        else:
            self._prog.stop()
