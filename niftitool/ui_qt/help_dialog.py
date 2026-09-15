"""The Help window.

A searchable topic list beside a reading pane, which is what a user
expects "Help" to open and what a hover tooltip could never be: it stays
open while you work, it can be read without holding the mouse still, it
can be searched, and the text can be selected and copied.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSplitter,
    QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, ENTRY_BG, PANEL, TEXT, TEXT_DIM,
)
from .help_content import SECTIONS, WELCOME, matches

#: Role holding a topic's rendered body on its tree item.
_BODY_ROLE = Qt.ItemDataRole.UserRole


class HelpDialog(QDialog):
    """Non-modal help browser.

    Non-modal on purpose: the point of moving this out of tooltips is that
    you can leave it open beside the controls it describes and change a
    setting while reading about it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NIfTI Tool Suite - Help")
        self.setModal(False)
        self.resize(940, 620)
        self.setSizeGripEnabled(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter, 1)

        # left: search + topic tree
        left = QWidget(splitter)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)

        self._search = QLineEdit(left)
        self._search.setPlaceholderText("Search help...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        left_lay.addWidget(self._search)

        self._tree = QTreeWidget(left)
        self._tree.setHeaderHidden(True)
        self._tree.setStyleSheet(
            f"QTreeWidget {{ background-color: {ENTRY_BG}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
            f"QTreeWidget::item:selected {{ background-color: {ACCENT}; "
            f"color: #FFFFFF; }}"
        )
        self._tree.currentItemChanged.connect(self._show_current)
        left_lay.addWidget(self._tree, 1)

        self._count = QLabel("", left)
        self._count.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {BG};")
        left_lay.addWidget(self._count)

        # right: reading pane
        self._view = QTextBrowser(splitter)
        self._view.setOpenExternalLinks(False)
        self._view.setStyleSheet(
            f"QTextBrowser {{ background-color: {ENTRY_BG}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; padding: 10px; }}"
        )
        self._view.document().setDefaultStyleSheet(
            f"h2 {{ color: {ACCENT}; }}"
            f"dt {{ margin-top: 6px; }}"
            f"dd {{ margin-left: 18px; margin-bottom: 4px; }}"
            f"code {{ background-color: {PANEL}; }}"
        )
        self._view.setFont(QFont("Segoe UI", 10))

        splitter.addWidget(left)
        splitter.addWidget(self._view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 640])

        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        row.addWidget(close)
        root.addLayout(row)

        self._populate("")
        self._view.setHtml(f"<h2>Help</h2>{WELCOME}")

    # population and filtering

    def _populate(self, query: str) -> int:
        """Rebuild the tree, keeping only topics matching *query*."""
        self._tree.clear()
        shown = 0
        for section in SECTIONS:
            topics = [t for t in section.topics if matches(t, query)]
            if not topics:
                continue
            parent = QTreeWidgetItem(self._tree, [section.title])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            for topic in topics:
                child = QTreeWidgetItem(parent, [topic.title])
                child.setData(0, _BODY_ROLE,
                              f"<h2>{topic.title}</h2>{topic.body}")
                shown += 1
            parent.setExpanded(True)
        return shown

    def _apply_filter(self, query: str):
        shown = self._populate(query)
        if not query.strip():
            self._count.setText("")
        elif shown:
            self._count.setText(
                f"{shown} topic{'s' if shown != 1 else ''} match")
            # Land on the first hit, so a search is one keystroke from an
            # answer rather than a filtered list you still have to click.
            first = self._first_topic_item()
            if first is not None:
                self._tree.setCurrentItem(first)
        else:
            self._count.setText("No topics match")
            self._view.setHtml(
                f"<h2>No results</h2><p>Nothing matches "
                f"<b>{query}</b>.</p>")

    def _first_topic_item(self):
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            if parent.childCount():
                return parent.child(0)
        return None

    def _show_current(self, current, _previous):
        if current is None:
            return
        body = current.data(0, _BODY_ROLE)
        if body:
            self._view.setHtml(body)
            self._view.verticalScrollBar().setValue(0)

    # public API

    def show_topic(self, title: str) -> bool:
        """Select the topic whose title equals *title*. True if it existed."""
        self._search.clear()
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.text(0) == title:
                    self._tree.setCurrentItem(child)
                    return True
        return False

    def show_section(self, title: str) -> bool:
        """Select the first topic of the section named *title*."""
        self._search.clear()
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            if parent.text(0) == title and parent.childCount():
                self._tree.setCurrentItem(parent.child(0))
                return True
        return False

    def focus_search(self):
        self._search.setFocus()
        self._search.selectAll()
