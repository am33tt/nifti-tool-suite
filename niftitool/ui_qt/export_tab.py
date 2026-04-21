"""Simulation Export tab — PyQt6 port."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget,
)

from ..config import ACCENT, BG, BORDER, PANEL, TEAL, TEXT, TEXT_DIM
from .widgets import hline, styled_btn


class _PlainTextShim:
    """Tkinter-Text ``.get('1.0','end')`` / ``.insert('end',...)`` shim."""

    def __init__(self, w: QPlainTextEdit):
        self._w = w

    def get(self, _start='1.0', _end='end'):
        return self._w.toPlainText()

    def delete(self, _start='1.0', _end='end'):
        self._w.clear()

    def insert(self, _where, text):
        cur = self._w.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(text)

    def config(self, **kw):
        # Accept state='normal'|'disabled' like tk's ScrolledText.
        state = kw.get('state')
        if state is not None:
            self._w.setReadOnly(state == 'disabled')


class ExportTabMixin:
    """Adds the Sim Export tab to :class:`NiftiApp`."""

    def _build_export_tab(self, parent):
        outer = QWidget(parent)
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(16, 12, 16, 12)
        outer_lay.setSpacing(4)

        root_lay = QVBoxLayout(parent)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.addWidget(outer)

        title = QLabel("Simulation Export", outer)
        title.setFont(QFont("Segoe UI Semibold", 11))
        title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        outer_lay.addWidget(title)

        sub = QLabel("Run  ▶ Compute E-Map  first in the controls panel.", outer)
        sub.setFont(QFont("Segoe UI", 9))
        sub.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        outer_lay.addWidget(sub)

        outer_lay.addWidget(hline(outer))

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
            btn_f = QFrame(outer)
            btn_f.setStyleSheet(
                f"QFrame {{ background-color: {PANEL}; border: 1px solid {BORDER}; }}"
            )
            bf_lay = QHBoxLayout(btn_f)
            bf_lay.setContentsMargins(12, 8, 12, 8)
            bf_lay.setSpacing(12)
            bf_lay.addWidget(styled_btn(btn_f, label, cmd, teal=True))
            tip_lbl = QLabel(tip, btn_f)
            tip_lbl.setFont(QFont("Segoe UI", 9))
            tip_lbl.setStyleSheet(
                f"color: {TEXT_DIM}; background-color: {PANEL};"
            )
            bf_lay.addWidget(tip_lbl)
            bf_lay.addStretch(1)
            outer_lay.addWidget(btn_f)

        outer_lay.addWidget(hline(outer))

        # ct.hpp preview
        preview_title = QLabel("ct.hpp snippet preview:", outer)
        preview_title.setFont(QFont("Segoe UI Semibold", 11))
        preview_title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        outer_lay.addWidget(preview_title)

        self._cpp_text_widget = QPlainTextEdit(outer)
        self._cpp_text_widget.setStyleSheet(
            "QPlainTextEdit { background-color: #1E1E1E; color: #D4D4D4; "
            "border: none; }"
        )
        self._cpp_text_widget.setFont(QFont("Consolas", 9))
        self._cpp_text_widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._cpp_text_widget.setMinimumHeight(260)
        self._cpp_text_widget.setPlainText(
            "// Compute the E-Map first, then click 'Generate ct.hpp snippet'.\n"
        )
        outer_lay.addWidget(self._cpp_text_widget, 1)

        self._cpp_text = _PlainTextShim(self._cpp_text_widget)

        btn_row = QWidget(outer)
        br_lay = QHBoxLayout(btn_row)
        br_lay.setContentsMargins(0, 4, 0, 0)
        br_lay.setSpacing(6)
        br_lay.addWidget(styled_btn(btn_row, "Copy to clipboard",
                                    self._copy_cpp, small=True))
        br_lay.addWidget(styled_btn(btn_row, "Save as ct.hpp...",
                                    self._save_cpp, small=True))
        br_lay.addStretch(1)
        outer_lay.addWidget(btn_row)
