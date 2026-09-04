"""Reusable PyQt6 widgets (styled button, collapsible section, progress).

The widgets hold no application state; styling is applied inline through
stylesheets.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, BTN_BG, BTN_HOV, ERR, PANEL, PANEL2, TEAL, TEXT, TEXT_DIM,
    ENTRY_BG,
)


# matplotlib toolbar

def tint_icon(icon: QIcon, color: str) -> QIcon:
    """Recolour an icon to *color*, keeping its shape.

    The icon is used as its own mask through ``SourceIn``, so the original
    colour does not matter; black and white glyphs both come out in the
    requested colour.
    """
    sizes = icon.availableSizes()
    if not sizes:
        return icon
    largest = max(sizes, key=lambda s: s.width() * s.height())
    source = icon.pixmap(largest)
    if source.isNull():
        return icon

    tinted = QPixmap(source.size())
    tinted.setDevicePixelRatio(source.devicePixelRatio())
    tinted.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tinted)
    try:
        painter.drawPixmap(0, 0, source)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(color))
    finally:
        painter.end()
    return QIcon(tinted)


def style_nav_toolbar(toolbar, *, color: str = TEXT, background: str = PANEL2):
    """Make a matplotlib navigation toolbar legible on the light panels.

    Matplotlib picks the toolbar icon colour once, at construction, from the
    widget palette: a dark palette background makes it refill every icon with
    the palette foreground, which is white. This application paints its
    panels through a stylesheet and never touches the palette, so under a
    dark system theme the toolbar shows white icons on a cream panel.

    The palette is therefore corrected, so later icons are left alone, and
    every icon already on the toolbar is repainted in the application text
    colour.
    """
    palette = toolbar.palette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base,
                 QPalette.ColorRole.Button):
        palette.setColor(role, QColor(background))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.Text):
        palette.setColor(role, QColor(color))
    toolbar.setPalette(palette)

    for action in toolbar.actions():
        icon = action.icon()
        if icon is not None and not icon.isNull():
            action.setIcon(tint_icon(icon, color))

    toolbar.setStyleSheet(
        f"QToolBar {{ background-color: {background}; border: none;"
        f" padding: 2px; spacing: 2px; }}"
        f"QToolButton {{ background-color: transparent; color: {color};"
        f" border: 1px solid transparent; border-radius: 3px; padding: 3px; }}"
        f"QToolButton:hover {{ background-color: {BTN_HOV}; }}"
        f"QToolButton:checked {{ background-color: {BTN_HOV};"
        f" border: 1px solid {BORDER}; }}"
        f"QToolButton:pressed {{ background-color: {BORDER}; }}"
        f"QLabel {{ color: {color}; background-color: transparent; }}"
    )
    return toolbar


# atoms

def _btn_stylesheet(bg: str, fg: str, hover_bg: str) -> str:
    return (
        f"QPushButton {{ background-color: {bg}; color: {fg}; "
        f"border: none; padding: 6px 10px; }}"
        f"QPushButton:hover {{ background-color: {hover_bg}; color: {TEXT}; }}"
        f"QPushButton:disabled {{ background-color: {BTN_BG}; color: {TEXT_DIM}; }}"
    )


def styled_btn(parent, text, cmd, *, accent=False, danger=False, small=False,
               teal=False, **_kw):
    """Return a themed :class:`QPushButton`.

    ``parent`` sets ownership only; callers add the button to a layout.
    """
    if teal:
        bg, fg = TEAL, "#FFFFFF"
        hover = TEAL
    elif accent:
        bg, fg = ACCENT, "#FFFFFF"
        hover = "#C96A1A"
    elif danger:
        bg, fg = ERR, "#FFFFFF"
        hover = ERR
    else:
        bg, fg = BTN_BG, TEXT
        hover = BTN_HOV

    b = QPushButton(text, parent)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(_btn_stylesheet(bg, fg, hover))
    font = QFont("Segoe UI", 9 if small else 10)
    b.setFont(font)
    if cmd is not None:
        b.clicked.connect(lambda _checked=False, fn=cmd: fn())
    return b


def styled_entry(parent=None, *, width=None, placeholder=""):
    """Return a themed :class:`QLineEdit`."""
    e = QLineEdit(parent) if parent is not None else QLineEdit()
    e.setStyleSheet(
        f"QLineEdit {{ background-color: {ENTRY_BG}; color: {TEXT}; "
        f"border: 1px solid {BORDER}; padding: 2px 4px; }}"
        f"QLineEdit:focus {{ border: 1px solid {ACCENT}; }}"
    )
    e.setFont(QFont("Consolas", 10))
    if placeholder:
        e.setPlaceholderText(placeholder)
    if width is not None:
        # Approximate character-to-pixel conversion.
        e.setFixedWidth(int(width * 8))
    return e


def lbl(parent, text, *, dim=False, head=False, bg=PANEL, **_kw):
    fg = TEXT_DIM if dim else TEXT
    font_size = 10
    font_bold = False
    if head:
        font_bold = True
        font_size = 11
    q = QLabel(text, parent) if parent is not None else QLabel(text)
    font = QFont("Segoe UI Semibold" if head else "Segoe UI", font_size)
    font.setBold(font_bold)
    q.setFont(font)
    q.setStyleSheet(f"color: {fg}; background-color: {bg};")
    return q


# composite widgets

class CollapsibleSection(QWidget):
    """Click-to-toggle panel with an accent-coloured header."""

    def __init__(self, parent, title, start_open=True):
        super().__init__(parent)
        self._title = title
        self._open = start_open

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 3, 0, 3)
        self._root.setSpacing(0)

        self._toggle_btn = QToolButton(self)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(start_open)
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._toggle_btn.setAutoRaise(True)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setText(self._header_text())
        self._toggle_btn.setStyleSheet(
            f"QToolButton {{ color: {ACCENT}; background-color: {BG}; "
            f"border: none; font: bold 11pt 'Segoe UI Semibold'; "
            f"padding: 2px 6px; text-align: left; }}"
        )
        self._toggle_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle_btn.clicked.connect(self._on_toggled)
        self._root.addWidget(self._toggle_btn)

        divider = QFrame(self)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {BORDER};")
        self._root.addWidget(divider)

        # Content widget - callers add to .content_layout
        self.content = QFrame(self)
        self.content.setStyleSheet(
            f"QFrame {{ background-color: {PANEL}; border: 1px solid {BORDER}; }}"
        )
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        self.content_layout.setSpacing(2)
        self._root.addWidget(self.content)

        self.content.setVisible(start_open)

    def _header_text(self):
        return f" {'▼' if self._open else '►'} {self._title}"

    def _on_toggled(self):
        self._open = self._toggle_btn.isChecked()
        self._toggle_btn.setText(self._header_text())
        self.content.setVisible(self._open)

    def toggle(self):
        self._toggle_btn.setChecked(not self._toggle_btn.isChecked())
        self._on_toggled()

    def open(self):
        if not self._open:
            self.toggle()


class IntRangeRow(QWidget):
    """Axis-labelled start/end integer entry row (used by the crop panel)."""

    def __init__(self, parent, axis_label, default_lo=0, default_hi=0):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 4, 2)
        lay.setSpacing(4)
        self.setStyleSheet(f"background-color: {PANEL};")

        ax = QLabel(axis_label, self)
        ax.setStyleSheet(f"color: {ACCENT}; background-color: {PANEL};")
        ax.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        ax.setFixedWidth(20)
        lay.addWidget(ax)

        s_lbl = QLabel("start", self)
        s_lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {PANEL};")
        s_lbl.setFont(QFont("Segoe UI", 9))
        lay.addWidget(s_lbl)

        self.lo = styled_entry(self, width=6)
        self.lo.setText(str(default_lo))
        lay.addWidget(self.lo)

        e_lbl = QLabel("end", self)
        e_lbl.setStyleSheet(f"color: {TEXT_DIM}; background-color: {PANEL};")
        e_lbl.setFont(QFont("Segoe UI", 9))
        lay.addWidget(e_lbl)

        self.hi = styled_entry(self, width=6)
        self.hi.setText(str(default_hi))
        lay.addWidget(self.hi)

        lay.addStretch(1)

        # ``.get()`` and ``.set()`` shims for the call sites in actions.py.
        self.start_var = _LineEditVar(self.lo)
        self.end_var = _LineEditVar(self.hi)

    def values(self):
        return int(self.lo.text()), int(self.hi.text())


class _LineEditVar:
    """Duck-typed ``tk.StringVar`` - ``.get()`` / ``.set()`` on a QLineEdit."""

    def __init__(self, line_edit: QLineEdit):
        self._w = line_edit

    def get(self) -> str:
        return self._w.text()

    def set(self, value) -> None:
        self._w.setText(str(value))


class StageProgressBar(QWidget):
    """Six-stage load indicator that doubles as an indeterminate spinner."""

    STAGES = [
        ("Opening file",      0.10),
        ("Decoding voxels",   0.40),
        ("Greyscale convert", 0.65),
        ("Updating sliders",  0.80),
        ("Computing stats",   0.92),
        ("Ready",             1.00),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stage_idx = -1
        self._mode = 'hidden'
        self._spin_pos = 0.0
        self._label_text = ""
        self.setFixedHeight(18)
        self.setMinimumWidth(260)
        self.setStyleSheet(f"background-color: {PANEL2};")
        self.setVisible(False)

        from PyQt6.QtCore import QTimer
        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._tick_spinner)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide)

    # public API

    def begin_staged(self):
        self._stage_idx = 0
        self._mode = 'stage'
        self.setVisible(True)
        self._update_from_stage()

    def advance(self):
        if self._stage_idx < len(self.STAGES) - 1:
            self._stage_idx += 1
        self._update_from_stage()

    def done(self):
        self._stage_idx = len(self.STAGES) - 1
        self._update_from_stage()
        self._hide_timer.start(900)

    def start(self, interval=60):
        self._mode = 'spinner'
        self._spin_pos = 0.0
        self.setVisible(True)
        self._spin_timer.start(interval)
        self.update()

    def stop(self):
        self._spin_timer.stop()
        self._hide()

    def reset(self):
        self._stage_idx = -1
        self._mode = 'hidden'
        self._spin_pos = 0.0
        self._label_text = ""
        self._hide()

    def set_stage(self, i: int):
        self._stage_idx = max(-1, min(i, len(self.STAGES) - 1))
        self._mode = 'stage'
        self.setVisible(True)
        self._update_from_stage()

    # drawing

    def sizeHint(self) -> QSize:
        return QSize(260, 18)

    def _hide(self):
        self._mode = 'hidden'
        self._label_text = ""
        self.setVisible(False)
        self.update()

    def _update_from_stage(self):
        if self._stage_idx < 0 or self._stage_idx >= len(self.STAGES):
            return
        name, _frac = self.STAGES[self._stage_idx]
        self._label_text = name
        self.update()

    def _tick_spinner(self):
        if self._mode != 'spinner':
            return
        self._spin_pos = (self._spin_pos + 0.05) % 1.05
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor(PANEL2))

        label_w = 100
        if self._label_text:
            p.setPen(QColor(TEXT_DIM))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(6, 0, label_w, h, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                       self._label_text)

        bar_x = label_w + 6
        bar_w = max(10, w - bar_x - 8)
        bar_y = 3
        bar_h = h - 6

        if self._mode == 'stage':
            if self._stage_idx < 0 or self._stage_idx >= len(self.STAGES):
                frac = 0.0
            else:
                frac = self.STAGES[self._stage_idx][1]
        elif self._mode == 'spinner':
            frac = min(self._spin_pos, 1.0)
        else:
            frac = 0.0

        # One cell per stage, filled up to the current fraction.
        n = len(self.STAGES)
        cell_w = bar_w / n
        for i in range(n):
            x = int(bar_x + i * cell_w)
            cw = int((bar_x + (i + 1) * cell_w)) - x - 1
            p.fillRect(x, bar_y, cw, bar_h, QColor(BORDER))
            covered_frac = max(0.0, min(1.0, frac * n - i))
            if covered_frac > 0:
                fw = int(cw * covered_frac)
                if fw > 0:
                    p.fillRect(x, bar_y, fw, bar_h, QColor(ACCENT))

        p.setPen(QColor(TEXT if frac > 0.5 else TEXT_DIM))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(bar_x, bar_y, bar_w, bar_h,
                   int(Qt.AlignmentFlag.AlignCenter),
                   f"{int(frac * 100)}%")
        p.end()


# helpers

def hline(parent=None, color=None):
    f = QFrame(parent)
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background-color: {color or BORDER}; border: none;")
    return f


def vline(parent=None, color=None):
    f = QFrame(parent)
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFixedWidth(1)
    f.setStyleSheet(f"background-color: {color or BORDER}; border: none;")
    return f
