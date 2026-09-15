"""Log pane + thread-safe logging helpers."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, ACCENT2, BG, ENTRY_BG, ERR, TEAL, TEXT, TEXT_DIM, WARN,
)
from ..core import accel
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

        gpu_lbl = QLabel("GPU:", top_row)
        gpu_lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        tr_lay.addWidget(gpu_lbl)

        self._gpu_combo = QComboBox(top_row)
        for mode in accel.MODES:
            self._gpu_combo.addItem(mode)
        self._gpu_combo.setCurrentText(accel.get_mode())
        self._gpu_combo.currentTextChanged.connect(self._on_gpu_mode_changed)
        tr_lay.addWidget(self._gpu_combo)

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

    def _on_gpu_mode_changed(self, mode: str):
        """Switch the acceleration backend and record it in the log.

        Every run that follows is logged, so a session transcript states
        which backend produced its numbers.
        """
        accel.set_mode(mode)
        self._append_log(f"  {accel.gpu_status_line()}", 'teal')

    # thread-safe public API

    def _clear_log(self):
        self._log_widget.clear()

    def _append_log(self, text: str, tag: str = ''):
        """Emit a signal handled by :meth:`_do_append_log` on the GUI thread."""
        try:
            self._post_log_signal.emit(text, tag)
        except Exception:
            # Fallback for calls made before the signals are wired.
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
        # The Stop button follows the task registry in app.py, not this flag,
        # so it stays up for the whole task rather than for one message.
        if busy or self.tasks_running():
            self._prog.start()
        else:
            self._prog.stop()
