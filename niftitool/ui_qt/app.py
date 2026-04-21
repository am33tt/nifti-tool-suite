"""Main application window (PyQt6 port of the tkinter ``app.py``).

:class:`NiftiApp` is the thin composition layer — it owns the shared
volume state and wires the individual tab mixins together.  Each mixin
lives in its own file so behaviour for a given tab can be edited (and
bugs tracked) in isolation.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QScrollArea, QSplitter,
    QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, BTN_BG, BTN_HOV, ENTRY_BG, PANEL, PANEL2, TEXT, TEXT_DIM,
)
from ..core.slice_cache import SliceCache
from ..deps import missing_report
from ..utils import process_rss_mb
from .actions import ActionsMixin
from .controls import ControlsMixin
from .emap_tab import EmapTabMixin
from .export_tab import ExportTabMixin
from .histogram_tab import HistogramTabMixin
from .log_tab import LogTabMixin
from .triplanar_tab import TriplanarMixin
from .view3d_tab import View3DMixin
from .widgets import StageProgressBar, hline, styled_btn


def _build_qss() -> str:
    """Application-wide stylesheet assembled from config.py theme colours.

    Applied at the ``QApplication`` level so it reaches top-level dialogs
    (``QMessageBox``, ``QFileDialog``, etc.) which are not descendants of
    the main window and would otherwise fall back to the OS palette —
    on Windows 11 that means dark mode, which makes our light-themed
    dialogs unreadable.
    """
    return f"""
    QWidget {{
        color: {TEXT}; background-color: {BG};
        font-family: 'Segoe UI'; font-size: 10pt;
    }}
    QMainWindow, QDialog, QMessageBox {{ background-color: {BG}; color: {TEXT}; }}
    QLabel {{ color: {TEXT}; background-color: transparent; }}
    QScrollArea {{ background-color: {BG}; border: none; }}
    QScrollArea > QWidget > QWidget {{ background-color: {BG}; }}
    QSplitter::handle {{ background-color: {BORDER}; }}
    QSplitter::handle:horizontal {{ width: 6px; }}

    QTabWidget::pane {{ border: 1px solid {BORDER}; background-color: {BG}; }}
    QTabBar::tab {{
        background: {PANEL2}; color: {TEXT_DIM};
        padding: 4px 10px; border: 1px solid {BORDER};
        border-bottom: none;
    }}
    QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; }}
    QTabBar::tab:hover {{ background: {BORDER}; }}

    QPlainTextEdit, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {ENTRY_BG}; color: {TEXT};
        border: 1px solid {BORDER}; selection-background-color: {ACCENT};
        selection-color: #FFFFFF;
    }}

    QComboBox {{
        background-color: {ENTRY_BG}; color: {TEXT};
        border: 1px solid {BORDER}; padding: 2px 4px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {ENTRY_BG}; color: {TEXT};
        selection-background-color: {ACCENT};
        selection-color: #FFFFFF;
    }}

    QRadioButton, QCheckBox {{ background-color: transparent; color: {TEXT}; }}

    QSlider::groove:horizontal {{
        height: 6px; background: {BORDER}; border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT}; width: 12px; margin: -4px 0;
        border-radius: 6px;
    }}

    /* Dialog buttons (QMessageBox Yes/No, QFileDialog Open/Cancel, etc.) */
    QPushButton {{
        background-color: {BTN_BG}; color: {TEXT};
        border: 1px solid {BORDER}; padding: 4px 12px; min-width: 60px;
    }}
    QPushButton:hover {{ background-color: {BTN_HOV}; }}
    QPushButton:pressed {{ background-color: {BORDER}; }}
    QPushButton:default {{ border: 1px solid {ACCENT}; }}
    QPushButton:disabled {{ color: {TEXT_DIM}; background-color: {PANEL2}; }}

    /* QMessageBox icon + text area */
    QMessageBox QLabel {{ color: {TEXT}; background-color: transparent; }}

    /* QFileDialog internals */
    QDialog QListView, QDialog QTreeView, QDialog QTableView {{
        background-color: {ENTRY_BG}; color: {TEXT};
        alternate-background-color: {PANEL};
        selection-background-color: {ACCENT}; selection-color: #FFFFFF;
    }}
    QHeaderView::section {{
        background-color: {PANEL2}; color: {TEXT};
        border: 1px solid {BORDER}; padding: 2px 6px;
    }}
    QMenu {{
        background-color: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER};
    }}
    QMenu::item:selected {{ background-color: {ACCENT}; color: #FFFFFF; }}
    QToolTip {{
        background-color: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER};
    }}
    """


class NiftiApp(
    ControlsMixin,
    TriplanarMixin,
    View3DMixin,
    HistogramTabMixin,
    EmapTabMixin,
    ExportTabMixin,
    LogTabMixin,
    ActionsMixin,
    QMainWindow,
):
    """Top-level Qt window.

    All mutable volume state lives on ``self`` so every mixin can reach
    it directly (load → ``_img`` / ``_gray``; calibrate → ``_hu_vol``;
    compute E-Map → ``_E_map`` / ``_labels`` / ``_E_stats``; etc.).
    """

    # Thread-safe signal used by :meth:`_append_log` / :meth:`_set_status`
    # to marshal messages from background threads onto the GUI thread.
    _post_log_signal = pyqtSignal(str, str)
    _post_status_signal = pyqtSignal(str, bool)
    _post_call_signal = pyqtSignal(object)  # carries a zero-arg callable

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NIfTI Tool Suite  v6  —  SimReady")
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 750)

        # Apply QSS at application level so top-level dialogs (QMessageBox,
        # QFileDialog) pick it up too — otherwise Windows dark mode leaks in.
        from PyQt6.QtWidgets import QApplication
        qapp = QApplication.instance()
        if qapp is not None:
            qapp.setStyleSheet(_build_qss())
        else:
            self.setStyleSheet(_build_qss())

        # ── volume state ─────────────────────────────────────────────────────
        self._img = None
        self._gray = None
        self._path = None
        self._busy = False
        self._axis_labels = {'x': ('X', '-'), 'y': ('Y', '-'), 'z': ('Z', '-')}
        self._ww = None
        self._wc = None

        # ── material state ───────────────────────────────────────────────────
        self._hu_vol = None
        self._E_map = None
        self._labels = None
        self._hu_cal: dict = {}
        self._E_stats: dict = {}
        self._porosity = 0.0

        # ── perf aids ────────────────────────────────────────────────────────
        self._slice_cache = SliceCache()

        # ── build ────────────────────────────────────────────────────────────
        self._build_ui()
        self._connect_thread_signals()
        self._check_deps()
        self._setup_dnd()

        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self._open_file)

        self._start_ram_monitor()

    # ── thread-safe plumbing ────────────────────────────────────────────────

    def _connect_thread_signals(self):
        self._post_log_signal.connect(self._do_append_log)
        self._post_status_signal.connect(self._do_set_status)
        self._post_call_signal.connect(self._do_call)

    def _do_call(self, fn):
        try:
            fn()
        except Exception:
            pass

    def after(self, ms: int, fn, *args):
        """Tkinter compatibility shim — schedule ``fn(*args)`` on the GUI
        thread after ``ms`` milliseconds.  All existing ``actions.py``
        call sites continue to work unchanged.
        """
        if args:
            call = lambda f=fn, a=args: f(*a)
        else:
            call = fn
        if ms <= 0:
            # Marshal via signal for thread safety (callable may come from
            # a worker thread).
            self._post_call_signal.emit(call)
            return None
        QTimer.singleShot(int(ms), call)
        return None

    def after_cancel(self, _token):
        """No-op: we don't rely on cancelling, debounce logic uses fresh
        timers each time."""
        return None

    # ── tab switching ────────────────────────────────────────────────────────

    def _on_tab_changed(self, index: int):
        """Fire deferred work when the user switches tabs."""
        try:
            current = self._nb.widget(index)
        except Exception:
            return
        if current is self._viewer3d_tab:
            fn = getattr(self, 'maybe_auto_render_3d', None)
            if callable(fn):
                fn()

    # ── drag-and-drop ────────────────────────────────────────────────────────

    def _setup_dnd(self):
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self._load_path(path)

    # ── dependency check ─────────────────────────────────────────────────────

    def _check_deps(self):
        missing = missing_report()
        if missing:
            self._append_log(
                f"  Missing packages: {', '.join(missing)}\n"
                f"  Install: pip install nibabel numpy scipy matplotlib psutil scikit-image\n",
                'warn',
            )

    # ── RAM monitor ──────────────────────────────────────────────────────────

    def _start_ram_monitor(self):
        self._ram_timer = QTimer(self)
        self._ram_timer.timeout.connect(self._update_ram)
        self._ram_timer.start(2000)
        self._update_ram()

    def _update_ram(self):
        mb = process_rss_mb()
        if mb is not None:
            self._ram_label.setText(f"RAM  {mb:.0f} MiB")
        else:
            self._ram_label.setText("RAM  — (pip install psutil)")

    # ── clipboard helper (used by actions._copy_cpp) ────────────────────────

    def clipboard_clear(self):
        from PyQt6.QtWidgets import QApplication
        cb = QApplication.clipboard()
        if cb is not None:
            cb.clear()

    def clipboard_append(self, text: str):
        from PyQt6.QtWidgets import QApplication
        cb = QApplication.clipboard()
        if cb is not None:
            existing = cb.text() or ""
            cb.setText(existing + text)

    # ── UI skeleton ──────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ─────────────────────────────────────────────────────────
        top = QFrame(self)
        top.setStyleSheet(f"background-color: {BG};")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(14, 8, 14, 8)
        title = QLabel("NIfTI Tool Suite", top)
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        top_lay.addWidget(title)
        sub = QLabel("v6  SimReady  |  Ctrl+O to open", top)
        sub.setFont(QFont("Segoe UI", 9))
        sub.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        top_lay.addWidget(sub)
        top_lay.addStretch(1)
        open_btn = styled_btn(top, "Open NIfTI...", self._open_file, accent=True)
        top_lay.addWidget(open_btn)
        gz_btn = styled_btn(top, "Gunzip .gz...", self._do_gunzip)
        top_lay.addWidget(gz_btn)
        root.addWidget(top)

        # Separator
        root.addWidget(hline(self))

        # ── File info bar ───────────────────────────────────────────────────
        inf = QFrame(self)
        inf.setStyleSheet(f"background-color: {PANEL2};")
        inf_lay = QHBoxLayout(inf)
        inf_lay.setContentsMargins(10, 5, 10, 5)
        self._info_label = QLabel(
            "No file loaded  ─  open a .nii or .nii.gz file", inf,
        )
        self._info_label.setFont(QFont("Consolas", 10))
        self._info_label.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2};"
        )
        inf_lay.addWidget(self._info_label)
        inf_lay.addStretch(1)
        root.addWidget(inf)

        # ── Main splitter ───────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(6)

        # Left: scrollable control panel
        ctrl_outer = QWidget(splitter)
        ctrl_outer_lay = QVBoxLayout(ctrl_outer)
        ctrl_outer_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_outer_lay.setSpacing(0)

        ctrl_scroll = QScrollArea(ctrl_outer)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._ctrl_frame = QWidget()
        self._ctrl_frame.setStyleSheet(f"background-color: {BG};")
        self._ctrl_layout = QVBoxLayout(self._ctrl_frame)
        self._ctrl_layout.setContentsMargins(10, 8, 10, 8)
        self._ctrl_layout.setSpacing(3)

        ctrl_scroll.setWidget(self._ctrl_frame)
        ctrl_outer_lay.addWidget(ctrl_scroll)
        ctrl_outer.setMinimumWidth(320)
        splitter.addWidget(ctrl_outer)

        self._build_controls(self._ctrl_frame)
        # Final stretch so controls don't expand vertically to fill
        self._ctrl_layout.addStretch(1)

        # Right: notebook
        right = QWidget(splitter)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        nb = QTabWidget(right)
        right_lay.addWidget(nb)
        right.setMinimumWidth(700)
        splitter.addWidget(right)

        # Create empty tab container widgets; each build_* populates them.
        self._triplanar_tab = QWidget()
        self._viewer3d_tab = QWidget()
        self._histogram_tab = QWidget()
        self._emap_tab = QWidget()
        self._export_tab = QWidget()
        self._log_tab = QWidget()

        for w in (self._triplanar_tab, self._viewer3d_tab, self._histogram_tab,
                  self._emap_tab, self._export_tab, self._log_tab):
            w.setStyleSheet(f"background-color: {BG};")

        nb.addTab(self._triplanar_tab, "  Tri-Planar  ")
        nb.addTab(self._viewer3d_tab, "  3-D View  ")
        nb.addTab(self._histogram_tab, "  Histogram  ")
        nb.addTab(self._emap_tab, "  E-Map Viewer  ")
        nb.addTab(self._export_tab, "  Sim Export  ")
        nb.addTab(self._log_tab, "  Log  ")

        self._nb = nb

        self._build_triplanar(self._triplanar_tab)
        self._build_3d_view(self._viewer3d_tab)
        nb.currentChanged.connect(self._on_tab_changed)
        self._build_histogram_tab(self._histogram_tab)
        self._build_emap_tab(self._emap_tab)
        self._build_export_tab(self._export_tab)
        self._build_log(self._log_tab)

        splitter.setSizes([380, 1200])
        root.addWidget(splitter, 1)

        # ── Status bar ──────────────────────────────────────────────────────
        root.addWidget(hline(self))
        status_bar = QFrame(self)
        status_bar.setStyleSheet(f"background-color: {PANEL2};")
        sb_lay = QHBoxLayout(status_bar)
        sb_lay.setContentsMargins(10, 4, 16, 4)

        self._status_label = QLabel("Ready", status_bar)
        self._status_label.setFont(QFont("Segoe UI", 9))
        self._status_label.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2};"
        )
        sb_lay.addWidget(self._status_label)

        sb_lay.addSpacing(14)

        self._probe_label = QLabel("", status_bar)
        self._probe_label.setFont(QFont("Consolas", 10))
        self._probe_label.setStyleSheet(
            f"color: {ACCENT}; background-color: {PANEL2};"
        )
        sb_lay.addWidget(self._probe_label)

        sb_lay.addStretch(1)

        self._prog = StageProgressBar(status_bar)
        sb_lay.addWidget(self._prog)

        div = QFrame(status_bar)
        div.setFixedWidth(1)
        div.setStyleSheet(f"background-color: {BORDER};")
        sb_lay.addWidget(div)

        self._ram_label = QLabel("RAM  --", status_bar)
        self._ram_label.setFont(QFont("Segoe UI", 9))
        self._ram_label.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2};"
        )
        sb_lay.addWidget(self._ram_label)

        root.addWidget(status_bar)

    # ── convenience shims for Tk StringVar-style call sites ─────────────────

    # ``_info_var`` / ``_status_var`` / ``_probe_var`` / ``_ram_var`` used to
    # be ``tk.StringVar`` instances on which ``.set(...)`` was called from
    # many places in ``actions.py`` and the tabs.  Expose tiny .set()-able
    # shims backed by the QLabels so we don't have to rewrite those calls.

    @property
    def _info_var(self):
        return _LabelVar(self._info_label)

    @property
    def _status_var(self):
        return _LabelVar(self._status_label)

    @property
    def _probe_var(self):
        return _LabelVar(self._probe_label)

    @property
    def _ram_var(self):
        return _LabelVar(self._ram_label)


class _LabelVar:
    """Tiny shim that presents a QLabel with a ``tk.StringVar``-like API."""

    def __init__(self, label: QLabel):
        self._lbl = label

    def get(self) -> str:
        return self._lbl.text()

    def set(self, value) -> None:
        self._lbl.setText(str(value))
