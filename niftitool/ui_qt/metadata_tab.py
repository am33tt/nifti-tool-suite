"""Metadata viewer tab — shows a rendered version of the log-pane
metadata report as a scrollable column of titled cards."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from ..config import ACCENT, BG, BORDER, PANEL, PANEL2, TEAL, TEXT, TEXT_DIM


class MetadataTabMixin:
    """Adds the Metadata viewer tab to :class:`NiftiApp`."""

    def _build_metadata_tab(self, parent):
        root = QVBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Scroll host — metadata easily overflows on large files.
        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background-color: {BG}; border: none; }}")
        root.addWidget(scroll, 1)

        host = QWidget(scroll)
        host.setStyleSheet(f"background-color: {BG};")
        self._meta_host_lay = QVBoxLayout(host)
        self._meta_host_lay.setContentsMargins(14, 14, 14, 14)
        self._meta_host_lay.setSpacing(12)
        scroll.setWidget(host)

        # Placeholder shown until the user triggers "View metadata".
        self._meta_placeholder = QLabel(
            "Load a NIfTI file and click “View metadata” to see a "
            "structured report here.",
            host,
        )
        self._meta_placeholder.setWordWrap(True)
        self._meta_placeholder.setFont(QFont("Segoe UI", 10))
        self._meta_placeholder.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: transparent;"
        )
        self._meta_host_lay.addWidget(self._meta_placeholder)
        self._meta_host_lay.addStretch(1)

    def show_metadata_sections(self, sections: list):
        """Populate the tab from a structured-metadata bundle produced by
        :func:`niftitool.core.metadata.collect_metadata`."""
        lay = self._meta_host_lay

        # Drop everything that's there (placeholder, old cards, trailing stretch).
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        title = QLabel(
            getattr(self, '_path', None).name
            if getattr(self, '_path', None) is not None
            else "Metadata",
        )
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT}; background-color: transparent;")
        lay.addWidget(title)

        for section in sections:
            lay.addWidget(self._build_meta_card(section))
        lay.addStretch(1)

    def _build_meta_card(self, section: dict) -> QWidget:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {PANEL}; "
            f"border: 1px solid {BORDER}; border-radius: 6px; }}"
        )
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        outer = QVBoxLayout(card)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        header = QLabel(section.get('title', ''))
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setStyleSheet(
            f"color: {ACCENT}; background-color: transparent; "
            f"border: none; letter-spacing: 0.5px;"
        )
        outer.addWidget(header)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setStyleSheet(f"color: {BORDER}; background-color: {BORDER};")
        rule.setFixedHeight(1)
        outer.addWidget(rule)

        if 'rows' in section:
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
            for r, (k, v) in enumerate(section['rows']):
                k_lbl = QLabel(str(k))
                k_lbl.setFont(QFont("Segoe UI", 9))
                k_lbl.setStyleSheet(
                    f"color: {TEXT_DIM}; background-color: transparent; border: none;"
                )
                k_lbl.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
                )
                v_lbl = QLabel(str(v))
                v_lbl.setFont(QFont("Consolas", 9))
                v_lbl.setStyleSheet(
                    f"color: {TEXT}; background-color: transparent; border: none;"
                )
                v_lbl.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                v_lbl.setWordWrap(True)
                grid.addWidget(k_lbl, r, 0)
                grid.addWidget(v_lbl, r, 1)
            outer.addLayout(grid)
        elif 'matrix' in section:
            outer.addWidget(self._build_matrix_table(section['matrix']))

        return card

    def _build_matrix_table(self, matrix) -> QWidget:
        """Render the 4x4 affine as a monospace grid."""
        host = QFrame()
        host.setStyleSheet(
            f"background-color: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px;"
        )
        grid = QGridLayout(host)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(2)
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                cell = QLabel(f"{v:>10.4f}")
                cell.setFont(QFont("Consolas", 10))
                color = TEAL if (r < 3 and c < 3) else TEXT
                cell.setStyleSheet(
                    f"color: {color}; background-color: transparent; border: none;"
                )
                cell.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                grid.addWidget(cell, r, c)
        wrap = QWidget()
        wlay = QHBoxLayout(wrap)
        wlay.setContentsMargins(0, 0, 0, 0)
        wlay.addWidget(host)
        wlay.addStretch(1)
        return wrap
