"""Main application window.

NiftiApp owns the shared volume state and wires the tab mixins together.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QStatusBar,
    QTabWidget,
    QVBoxLayout, QWidget,
)

from ..config import (
    ACCENT, BG, BORDER, BTN_BG, BTN_HOV, ENTRY_BG, PANEL, PANEL2, TEXT, TEXT_DIM,
)
from ..core import accel
from ..core.slice_cache import SliceCache
from ..deps import missing_report
from ..utils import process_rss_mb
from .actions import ActionsMixin
from .beam_hardening_tab import BeamHardeningTabMixin
from .binarize_tab import BinarizeTabMixin
from .controls import ControlsMixin
from .export_tab import ExportTabMixin
from .histogram_tab import HistogramTabMixin
from .log_tab import LogTabMixin
from .metadata_tab import MetadataTabMixin
from .porosity_tab import PorosityTabMixin
from .sim_export_actions import SimExportActionsMixin
from .triplanar_tab import TriplanarMixin
from .view3d_tab import View3DMixin
from .widgets import StageProgressBar, hline, styled_btn


#: The tooltip's look, kept separate because it has to be applied twice:
#: once in the application stylesheet, for tooltips on widgets that have no
#: stylesheet of their own, and once directly onto the tooltip window by
#: :class:`_ToolTipStyler` for the rest. See that class for why.
TOOLTIP_QSS = f"""
    QToolTip {{
        background-color: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER};
        padding: 4px 6px;
    }}
"""


class _ToolTipSuppressor(QObject):
    """Stop hover tooltips appearing anywhere in the application.

    Every explanation that used to live in a tooltip is now in Help. A
    tooltip is a poor place for one: it is unreadable on a dark-themed
    desktop unless every widget stylesheet remembers to restyle it, it
    disappears the moment the pointer moves, several of these ran to a
    paragraph or more, and none of them could be searched, selected or
    copied.

    This filter swallows the tooltip event itself rather than relying on
    every ``setToolTip`` call having been removed, so a tooltip Qt raises
    on its own -- an elided item in a tree, a truncated header -- does not
    reintroduce one either.

    The ``QToolTip`` rule in the stylesheet is kept as a fallback: if this
    filter is ever taken out, tooltips come back styled rather than black.
    """

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.Type.ToolTip:
                return True                    # handled: show nothing
        except Exception:
            pass
        return False


def _build_qss() -> str:
    """Application-wide stylesheet assembled from config.py theme colours.

    Applied at the ``QApplication`` level so it also reaches top-level
    dialogs (``QMessageBox``, ``QFileDialog``), which are not descendants of
    the main window and would otherwise use the OS palette.
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
    QComboBox:editable {{
        background-color: {ENTRY_BG}; color: {TEXT};
    }}
    QComboBox:!editable, QComboBox::drop-down:editable {{
        background-color: {ENTRY_BG}; color: {TEXT};
    }}
    QComboBox QAbstractItemView {{
        background-color: {ENTRY_BG}; color: {TEXT};
        border: 1px solid {BORDER};
        outline: 0;
        selection-background-color: {ACCENT};
        selection-color: #FFFFFF;
    }}
    QComboBox QAbstractItemView::item {{
        background-color: {ENTRY_BG}; color: {TEXT};
        padding: 3px 6px;
    }}
    QComboBox QAbstractItemView::item:selected {{
        background-color: {ACCENT}; color: #FFFFFF;
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: {PANEL2}; color: {TEXT};
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
    QMenuBar {{
        background-color: {PANEL2}; color: {TEXT};
        border-bottom: 1px solid {BORDER};
    }}
    QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
    QMenuBar::item:selected {{ background-color: {BORDER}; }}
    QMenuBar::item:pressed {{ background-color: {ACCENT}; color: #FFFFFF; }}
    QMenu {{
        background-color: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER};
    }}
    QMenu::item {{ padding: 4px 24px 4px 20px; }}
    QMenu::item:selected {{ background-color: {ACCENT}; color: #FFFFFF; }}
    QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
    """ + TOOLTIP_QSS


def _apply_theme(qapp) -> None:
    """Give *qapp* this application's light theme, three ways over.

    The stylesheet alone is not enough for tooltips. A tooltip is a native
    top-level window whose colours come from the palette, and on Windows
    Qt follows the *system* colour scheme for those: with Windows in dark
    mode the tooltip is painted from the dark palette -- black on dark grey
    -- while every widget the stylesheet reaches stays light. That is why
    only the tooltip looked wrong.

    So all three are set, cheapest first, and each is harmless if another
    already did the job:

    1. the colour-scheme hint, so Qt stops pulling dark system colours in
       (Qt 6.8 and newer; ignored silently on older builds);
    2. the palette's tooltip roles, which is what a natively drawn tooltip
       actually reads;
    3. the stylesheet, which covers everything else.
    """
    from PyQt6.QtGui import QColor, QPalette

    # 1. Do not follow a dark Windows into a dark palette: this app has one
    #    theme, defined in config.py, and it is a light one.
    try:
        from PyQt6.QtCore import Qt
        hints = qapp.styleHints()
        if hasattr(hints, "setColorScheme"):
            hints.setColorScheme(Qt.ColorScheme.Light)
    except Exception:
        pass

    # 2. Palette roles. ToolTipBase and ToolTipText are the two a tooltip
    #    reads when the platform style draws it itself.
    try:
        palette = qapp.palette()
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
        palette.setColor(QPalette.ColorRole.Window, QColor(BG))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
        palette.setColor(QPalette.ColorRole.Base, QColor(ENTRY_BG))
        palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
        qapp.setPalette(palette)
    except Exception:
        pass

    # 3. The stylesheet, applied at application level so top-level dialogs
    #    pick it up too.
    qapp.setStyleSheet(_build_qss())

    # 4. And no hover tooltips at all: the explanations they carried are in
    #    Help now. Kept on the application so it is not garbage collected --
    #    an event filter is not owned by the object it filters.
    if getattr(qapp, "_niftitool_tooltip_filter", None) is None:
        suppressor = _ToolTipSuppressor(qapp)
        qapp.installEventFilter(suppressor)
        qapp._niftitool_tooltip_filter = suppressor


class NiftiApp(
    ControlsMixin,
    TriplanarMixin,
    View3DMixin,
    HistogramTabMixin,
    PorosityTabMixin,
    BeamHardeningTabMixin,
    BinarizeTabMixin,
    MetadataTabMixin,
    ExportTabMixin,
    LogTabMixin,
    ActionsMixin,
    SimExportActionsMixin,
    QMainWindow,
):
    """Top-level Qt window.

    All mutable volume state lives on ``self`` so every mixin can reach it:
    load sets ``_img`` and ``_gray``, the porosity analysis sets
    ``_void_result``, beam hardening sets ``_bh_fit`` and binarisation sets
    ``_bin_result``.
    """

    # Thread-safe signal used by :meth:`_append_log` and :meth:`_set_status`
    # to marshal messages from worker threads onto the GUI thread.
    _post_log_signal = pyqtSignal(str, str)
    _post_status_signal = pyqtSignal(str, bool)
    _post_call_signal = pyqtSignal(object)  # carries a zero-arg callable
    _post_delayed_signal = pyqtSignal(int, object)  # (ms, callable)
    _post_tasks_signal = pyqtSignal(int)  # number of running background tasks

    _STOP_LABEL = "■  Stop"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NIfTI Tool Suite  v6  -  SimReady")
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 750)

        # Application-level theme so top-level dialogs and tooltips pick it
        # up too. See _apply_theme for why the stylesheet is not enough.
        from PyQt6.QtWidgets import QApplication
        qapp = QApplication.instance()
        if qapp is not None:
            _apply_theme(qapp)
        else:
            self.setStyleSheet(_build_qss())

        # volume state
        self._img = None
        self._gray = None
        self._path = None
        self._busy = False
        self._axis_labels = {'x': ('X', '-'), 'y': ('Y', '-'), 'z': ('Z', '-')}
        self._ww = None
        self._wc = None
        self._auto_wwwc = None   # cached auto window, one per volume

        # measurement state
        self._porosity = 0.0

        # simulation / export state
        self._rotation_history: list = []
        self._bc_surface_exports: list = []
        self._sim_load_direction = "+Y"
        self._auto_void_threshold_info = None
        self._pore_threshold_info = None
        self._pore_threshold_result = None
        self._bh_fit = None
        self._bh_fit_info = None

        # perf aids
        self._slice_cache = SliceCache()

        # Background task registry. Every worker thread started through
        # _run_task is listed here while it runs, which is what shows the
        # status-bar Stop button. The Stop button sets _cancel_event and the
        # workers poll it at their own checkpoints.
        self._active_tasks: list[str] = []
        self._task_lock = threading.Lock()
        self._cancel_event = threading.Event()

        # build
        self._build_menubar()
        self._build_ui()
        self._connect_thread_signals()
        self._check_deps()
        self._setup_dnd()

        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self._open_file)
        QShortcut(QKeySequence("F1"), self).activated.connect(self.show_help)

        self._start_ram_monitor()

    # help

    def _build_menubar(self):
        """The menu bar. Help lives here rather than in hover tooltips.

        Every control used to carry a tooltip. They were unreadable on a
        dark-themed desktop, they vanished the moment the pointer moved,
        several were far too long to read that way, and none of them could
        be searched or copied. The same text now lives in
        :mod:`niftitool.ui_qt.help_content` and is shown in a window that
        stays open beside the controls it describes.
        """
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QMenuBar

        # Not self.menuBar(): a menu bar holding one menu wastes a whole
        # band across the top of the window. This one is built detached and
        # placed inside the application bar in _build_ui, so the menu, the
        # product name and the file actions share a single row.
        menubar = QMenuBar(self)
        menubar.setNativeMenuBar(False)
        self._menubar = menubar
        help_menu = menubar.addMenu("&Help")

        contents = QAction("&Contents", self)
        contents.setShortcut(QKeySequence("F1"))
        contents.triggered.connect(self.show_help)
        help_menu.addAction(contents)

        search = QAction("&Search help...", self)
        search.setShortcut(QKeySequence("Ctrl+F1"))
        search.triggered.connect(self.search_help)
        help_menu.addAction(search)

        help_menu.addSeparator()
        # Direct routes to the topics people look for by name rather than
        # by browsing, so the menu answers the question without a search.
        for label, section in (
            ("&Getting started", "Getting started"),
            ("&Volumes larger than RAM", "Getting started"),
            ("&GPU and reproducibility", "Performance and reproducibility"),
        ):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _checked=False, name=section: self.show_help(name))
            help_menu.addAction(action)

        help_menu.addSeparator()
        about = QAction("&About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def show_help(self, section: str | None = None):
        """Open the Help window, optionally on a named section."""
        from .help_dialog import HelpDialog

        if getattr(self, "_help_dialog", None) is None:
            self._help_dialog = HelpDialog(self)
        dialog = self._help_dialog
        if isinstance(section, str) and section:
            dialog.show_section(section)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def search_help(self):
        self.show_help().focus_search()

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox
        from ..core import accel

        box = QMessageBox(self)
        box.setWindowTitle("About")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("NIfTI Tool Suite")
        box.setInformativeText(
            "Preprocessing and inspection for industrial and research CT "
            "volumes.\n\n"
            f"{accel.gpu_status_line()}\n\n"
            "Press F1 for help."
        )
        box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    # thread-safe plumbing

    def _connect_thread_signals(self):
        self._post_log_signal.connect(self._do_append_log)
        self._post_status_signal.connect(self._do_set_status)
        self._post_call_signal.connect(self._do_call)
        self._post_tasks_signal.connect(self._do_tasks_changed)
        self._post_delayed_signal.connect(self._do_delayed_call)
        # Lets the cross-thread schedulers short-circuit during shutdown.
        # Late callbacks from daemon threads would otherwise re-enter QTimer
        # after the event dispatcher is gone.
        self._shutting_down = False

    def _do_call(self, fn):
        if self._shutting_down:
            return
        try:
            fn()
        except Exception:
            pass

    def _do_delayed_call(self, ms: int, fn):
        if self._shutting_down:
            return
        # Runs on the GUI thread (queued connection) so QTimer finds a live
        # event dispatcher.
        QTimer.singleShot(int(ms), lambda: self._do_call(fn))

    def after(self, ms: int, fn, *args):
        """Schedule ``fn(*args)`` on the GUI thread after ``ms`` ms."""
        if self._shutting_down:
            return None
        if args:
            call = lambda f=fn, a=args: f(*a)
        else:
            call = fn
        if ms <= 0:
            # Marshal via signal; the callable may come from a worker thread.
            self._post_call_signal.emit(call)
            return None
        # Route delayed calls through the main thread as well; singleShot
        # called from a worker thread has no event dispatcher.
        self._post_delayed_signal.emit(int(ms), call)
        return None

    def after_cancel(self, _token):
        """No-op: the debounce logic uses a fresh timer each time."""
        return None

    # tab switching

    # Left tool panel opened automatically for each visualisation tab.
    _TAB_TOOL_LINKS = {
        'triplanar': 'viewer',
        'view3d':    'viewer',
        'histogram': 'stats',
        'porosity':  'threshold',
        'beam_hardening': 'beam_hardening',
        'binarize':  'binarize',
        'metadata':  'metadata',
    }

    def _select_tool(self, key: str):
        """Open the named tool panel (and highlight its nav button)."""
        btn = self._tool_nav_buttons.get(key)
        idx = self._tool_stack_index.get(key)
        if btn is None or idx is None:
            return
        btn.setChecked(True)
        self._tool_stack.setCurrentIndex(idx)

    def _on_tab_changed(self, index: int):
        """Fire deferred work when the user switches tabs."""
        try:
            current = self._nb.widget(index)
        except Exception:
            return
        # Auto-open the linked tool panel.
        for name, widget in self._tab_by_name.items():
            if widget is current:
                key = self._TAB_TOOL_LINKS.get(name)
                if key:
                    self._select_tool(key)
                break
        if current is self._viewer3d_tab:
            fn = getattr(self, 'maybe_auto_render_3d', None)
            if callable(fn):
                fn()

    # background tasks and cooperative cancellation

    def _run_task(self, label: str, fn):
        """Run ``fn`` on a daemon thread and track it in the status bar.

        The Stop button is shown as soon as the first task starts and hidden
        when the last one finishes, so any long action can be interrupted.
        ``fn`` polls :meth:`cancel_requested` at its own checkpoints; a step
        already inside a library call finishes before the task gives up.
        """
        def _wrapped():
            try:
                fn()
            finally:
                self._task_finished(label)

        self._task_started(label)
        thread = threading.Thread(target=_wrapped, name=label, daemon=True)
        thread.start()
        return thread

    def _begin_cancellable(self):
        """Arm a fresh cancellation token.

        :meth:`_run_task` does this for the first task of a batch; the method
        stays for code that starts a thread of its own.
        """
        self._cancel_event.clear()

    def _task_started(self, label: str):
        with self._task_lock:
            if not self._active_tasks:
                self._cancel_event.clear()
            self._active_tasks.append(label)
            count = len(self._active_tasks)
        self._post_tasks_signal.emit(count)

    def _task_finished(self, label: str):
        with self._task_lock:
            if label in self._active_tasks:
                self._active_tasks.remove(label)
            count = len(self._active_tasks)
        self._post_tasks_signal.emit(count)
        if count == 0:
            self._report_gpu_fallbacks()

    def _report_gpu_fallbacks(self):
        """Say once which kernels could not use the GPU and why.

        A silent fallback leaves a run correct but slow with nothing to
        explain it. Reported only in 'on' mode, where the user asked for
        the GPU explicitly; in 'auto' a fallback is the expected behaviour
        and not worth interrupting the log for.
        """
        if accel.get_mode() != "on":
            return
        notes = accel.fallback_notes()
        new = {k: v for k, v in notes.items()
               if k not in getattr(self, '_gpu_notes_shown', set())}
        if not new:
            return
        self._gpu_notes_shown = set(notes)
        for kernel, reason in new.items():
            self._append_log(
                f"  GPU: {kernel} ran on the CPU ({reason}).", 'warn')

    def tasks_running(self) -> bool:
        with self._task_lock:
            return bool(self._active_tasks)

    def _do_tasks_changed(self, count: int):
        """Show or hide the Stop button as tasks come and go."""
        running = count > 0
        button = getattr(self, '_stop_btn', None)
        if button is not None:
            button.setVisible(running)
            if running and not self.cancel_requested():
                button.setEnabled(True)
                button.setText(self._STOP_LABEL)
        if running:
            self._prog.start()
        else:
            self._prog.stop()
            if self.cancel_requested():
                self._status_var.set("Stopped.")
            self._cancel_event.clear()
            if button is not None:
                button.setEnabled(True)
                button.setText(self._STOP_LABEL)

    def cancel_requested(self) -> bool:
        """True once the user has pressed Stop, until the last task ends."""
        return self._cancel_event.is_set()

    def _abort_if_stopped(self, what: str) -> bool:
        """True when Stop was pressed; logs it and clears the status bar.

        Call this at the point where a task is about to commit its result, so
        a cancelled run leaves the loaded volume and the analysis state as
        they were.
        """
        if not self.cancel_requested():
            return False
        self._append_log(f"  {what} cancelled.", 'warn')
        self._set_status("Stopped.", busy=False)
        return True

    def _request_stop(self):
        if not self._cancel_event.is_set():
            self._append_log(
                "  Stop requested, finishing the current step...", 'warn',
            )
        self._cancel_event.set()
        button = getattr(self, '_stop_btn', None)
        if button is not None:
            button.setEnabled(False)
            button.setText("■  Stopping")
        self._status_var.set("Stopping...")

    def _show_tab(self, name: str):
        """Jump the right-hand tab widget to the named tab."""
        tab = self._tab_by_name.get(name)
        if tab is None:
            return
        self._nb.setCurrentWidget(tab)

    # drag-and-drop

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

    # shutdown

    def closeEvent(self, event):
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Quit NIfTI Tool Suite",
            "Are you sure you want to close?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        # Stop in-flight worker threads from scheduling further callbacks.
        self._shutting_down = True

        # Stop QTimers before the event loop tears down.
        for name in ('_ram_timer', '_tri_debounce_timer', '_tri_sync_timer',
                     ):
            t = getattr(self, name, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass

        # QVTKRenderWindowInteractor owns a QTimer for its render loop.
        # Without Finalize() that timer can fire after the Qt event
        # dispatcher is gone.
        vtk_widget = getattr(self, '_vtk_widget', None)
        if vtk_widget is not None:
            try:
                iren = vtk_widget.GetRenderWindow().GetInteractor()
                if iren is not None:
                    iren.TerminateApp()
                vtk_widget.Finalize()
                vtk_widget.close()
            except Exception:
                pass

        super().closeEvent(event)

    # dependency check

    def _check_deps(self):
        missing = missing_report()
        if missing:
            self._append_log(
                f"  Missing packages: {', '.join(missing)}\n"
                f"  Install: pip install nibabel numpy scipy matplotlib psutil scikit-image\n",
                'warn',
            )
        # State the compute backend once at startup, so a session log says
        # what produced its numbers rather than what was configured.
        self._append_log(f"  {accel.gpu_status_line()}", 'teal')

    # RAM monitor

    def _start_ram_monitor(self):
        self._ram_timer = QTimer(self)
        self._ram_timer.timeout.connect(self._update_ram)
        self._ram_timer.start(2000)
        self._update_ram()

    def _update_ram(self):
        mb = process_rss_mb()
        text = f"RAM  {mb:.0f} MiB" if mb is not None else "RAM  - (pip install psutil)"
        # Video memory sits next to it: the 3-D viewer and the filter
        # kernels are budgeted against this number, not against RAM, so it
        # is the one to watch when a render is being downsampled.
        free_mb, total_mb, source = accel.video_memory_mb()
        if total_mb is not None:
            if free_mb is not None:
                text += f"   VRAM  {(total_mb - free_mb):.0f} / {total_mb:.0f} MiB"
            else:
                text += f"   VRAM  {total_mb:.0f} MiB ({source})"
        self._ram_label.setText(text)

    # clipboard helpers

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

    # UI skeleton

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # One application bar instead of three stacked strips. The window
        # used to open with four bands of chrome -- a menu bar holding a
        # single menu, a centred wordmark with the file actions pushed to
        # the far right, a file-info strip, and the tool nav -- roughly
        # 170 px before any data. Brand, menu and actions belong on one
        # row, and the file summary reads better beside the tool nav than
        # on a band of its own.
        appbar = QFrame(self)
        appbar.setStyleSheet(f"background-color: {BG};")
        bar_lay = QHBoxLayout(appbar)
        bar_lay.setContentsMargins(12, 4, 12, 4)
        bar_lay.setSpacing(8)

        from .. import LOGO_PATH
        logo_lbl = QLabel(appbar)
        logo_pm = QIcon(str(LOGO_PATH)).pixmap(22, 22)
        if not logo_pm.isNull():
            logo_lbl.setPixmap(logo_pm)
        logo_lbl.setStyleSheet(f"background-color: {BG}; border: none;")
        bar_lay.addWidget(logo_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        title = QLabel("NIfTI Tool Suite", appbar)
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        bar_lay.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)

        # The version is a detail, not a heading: same row, quieter. The
        # "Ctrl+O to open" hint that used to sit here is gone -- it labelled
        # a button standing right beside it. Help lists the shortcuts.
        sub = QLabel("v6 \u00b7 SimReady", appbar)
        sub.setFont(QFont("Segoe UI", 9))
        sub.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        bar_lay.addWidget(sub, 0, Qt.AlignmentFlag.AlignVCenter)

        menubar = getattr(self, "_menubar", None)
        if menubar is not None:
            menubar.setStyleSheet("QMenuBar { background: transparent; "
                                  "border: none; }")
            menubar.setSizePolicy(QSizePolicy.Policy.Maximum,
                                  QSizePolicy.Policy.Preferred)
            bar_lay.addSpacing(6)
            bar_lay.addWidget(menubar, 0, Qt.AlignmentFlag.AlignVCenter)

        # The file summary lives here, in the widest free space the window
        # has, rather than on a band of its own or crowded onto the end of
        # the tool nav -- at the 1200 px minimum width the nav row leaves
        # it about a hundred pixels, which elides away everything but the
        # first few characters. It elides from the middle, so the name and
        # the voxel size both survive whatever room is left.
        bar_lay.addSpacing(10)
        self._info_label = _ElidingLabel(
            "No file loaded  \u2500  open a .nii or .nii.gz file", appbar,
        )
        self._info_label.setFont(QFont("Consolas", 9))
        self._info_label.set_color(TEXT_DIM)
        self._info_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._info_label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                       QSizePolicy.Policy.Preferred)
        bar_lay.addWidget(self._info_label, 1)
        bar_lay.addSpacing(10)

        # Stop sits with the file actions rather than in the status bar.
        # It is the one control a user reaches for under time pressure, and
        # the status bar is the furthest corner of the window from where
        # they started the task. Hidden until a task is running, so it
        # costs nothing the rest of the time; _do_tasks_changed shows it.
        self._stop_btn = styled_btn(appbar, self._STOP_LABEL,
                                    self._request_stop, danger=True)
        self._stop_btn.setVisible(False)
        bar_lay.addWidget(self._stop_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        open_btn = styled_btn(appbar, "Open NIfTI...", self._open_file,
                              accent=True)
        bar_lay.addWidget(open_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        gz_btn = styled_btn(appbar, "Gunzip .gz...", self._do_gunzip)
        bar_lay.addWidget(gz_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        # setMenuWidget puts this in the window's menu-bar slot, above the
        # central widget, so it reads as one bar rather than as content.
        self.setMenuWidget(appbar)

        # Build all tool panels (populates self._tool_panels)
        self._build_controls()

        # Tool navigation bar: one exclusive toggle per tool, in display order.
        tool_order = [
            ("viewer",   "Viewer"),
            ("metadata", "Metadata"),
            ("stats",      "Stats"),
            ("threshold",  "Threshold"),
            ("binarize",   "Binarise"),
            ("beam_hardening", "Beam Hardening"),
            ("background", "Background"),
            ("reorient",   "Reorient"),
            ("rotate",   "Rotate"),
            ("crop",     "Crop"),
        ]

        nav = QFrame(self)
        nav.setStyleSheet(f"background-color: {PANEL2};")
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(10, 2, 10, 2)
        nav_lay.setSpacing(2)

        self._tool_nav_group = QButtonGroup(self)
        self._tool_nav_group.setExclusive(True)
        self._tool_nav_buttons: dict[str, QPushButton] = {}

        def _nav_btn_qss() -> str:
            return (
                f"QPushButton {{"
                f"  background-color: transparent; color: {TEXT_DIM};"
                f"  border: none; padding: 6px 14px; font-weight: 600;"
                f"  border-bottom: 2px solid transparent;"
                f"}}"
                f"QPushButton:hover {{ color: {TEXT}; }}"
                f"QPushButton:checked {{"
                f"  color: {ACCENT};"
                f"  border-bottom: 2px solid {ACCENT};"
                f"}}"
            )

        for key, label in tool_order:
            btn = QPushButton(label, nav)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(_nav_btn_qss())
            nav_lay.addWidget(btn)
            self._tool_nav_group.addButton(btn)
            self._tool_nav_buttons[key] = btn

        nav_lay.addStretch(1)
        root.addWidget(nav)
        root.addWidget(hline(self))

        # Vertical splitter: main area on top, log pane on the bottom
        v_split = QSplitter(Qt.Orientation.Vertical, self)
        v_split.setHandleWidth(6)
        v_split.setChildrenCollapsible(False)

        # Horizontal splitter inside: tool stack (left) | viz tabs (right)
        h_split = QSplitter(Qt.Orientation.Horizontal, v_split)
        h_split.setHandleWidth(6)

        # Tool panels in a scroll area so tall panels do not widen the
        # window.
        left = QWidget(h_split)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        left_scroll = QScrollArea(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Without this the panels grow past the pane and the descriptions
        # run off the edge instead of wrapping.
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._tool_stack = QStackedWidget()
        self._tool_stack.setStyleSheet(f"background-color: {BG};")
        self._tool_stack_index: dict[str, int] = {}
        for key, _label in tool_order:
            panel = self._tool_panels[key]
            idx = self._tool_stack.addWidget(panel)
            self._tool_stack_index[key] = idx

        left_scroll.setWidget(self._tool_stack)
        left_lay.addWidget(left_scroll)
        left.setMinimumWidth(320)
        h_split.addWidget(left)

        # Wire nav buttons → stack page
        for key, btn in self._tool_nav_buttons.items():
            btn.clicked.connect(
                lambda _chk, k=key: self._tool_stack.setCurrentIndex(
                    self._tool_stack_index[k]
                )
            )

        # Default selection
        self._tool_nav_buttons["viewer"].setChecked(True)
        self._tool_stack.setCurrentIndex(self._tool_stack_index["viewer"])

        # Right: visualisation notebook
        right = QWidget(h_split)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        nb = QTabWidget(right)
        right_lay.addWidget(nb)
        right.setMinimumWidth(700)
        h_split.addWidget(right)

        self._triplanar_tab = QWidget()
        self._viewer3d_tab = QWidget()
        self._histogram_tab = QWidget()
        self._porosity_tab = QWidget()
        self._binarize_tab = QWidget()
        self._beam_hardening_tab = QWidget()
        self._metadata_tab = QWidget()
        self._export_tab = QWidget()

        for w in (self._triplanar_tab, self._viewer3d_tab, self._histogram_tab,
                  self._porosity_tab, self._binarize_tab,
                  self._beam_hardening_tab, self._metadata_tab,
                  self._export_tab):
            w.setStyleSheet(f"background-color: {BG};")

        nb.addTab(self._triplanar_tab, "  Tri-Planar  ")
        nb.addTab(self._viewer3d_tab, "  3-D View  ")
        nb.addTab(self._histogram_tab, "  Histogram  ")
        nb.addTab(self._porosity_tab, "  Porosity  ")
        nb.addTab(self._binarize_tab, "  Binarise  ")
        nb.addTab(self._beam_hardening_tab, "  Beam Hardening  ")
        nb.addTab(self._metadata_tab, "  Metadata  ")
        nb.addTab(self._export_tab, "  Export  ")

        self._nb = nb
        # Named aliases so action handlers can switch tabs by key.
        self._tab_by_name = {
            'triplanar': self._triplanar_tab,
            'view3d':    self._viewer3d_tab,
            'histogram': self._histogram_tab,
            'porosity':  self._porosity_tab,
            'binarize':  self._binarize_tab,
            'beam_hardening': self._beam_hardening_tab,
            'metadata':  self._metadata_tab,
            'export':    self._export_tab,
        }

        self._build_triplanar(self._triplanar_tab)
        self._build_3d_view(self._viewer3d_tab)
        nb.currentChanged.connect(self._on_tab_changed)
        self._build_histogram_tab(self._histogram_tab)
        self._build_porosity_tab(self._porosity_tab)
        self._build_binarize_tab(self._binarize_tab)
        self._build_beam_hardening_tab(self._beam_hardening_tab)
        self._build_metadata_tab(self._metadata_tab)
        self._build_export_tab(self._export_tab)

        h_split.setSizes([380, 1200])
        v_split.addWidget(h_split)

        # Bottom: always-visible log pane
        log_pane = QWidget(v_split)
        log_pane.setStyleSheet(f"background-color: {BG};")
        log_pane.setMinimumHeight(120)
        self._build_log(log_pane)
        v_split.addWidget(log_pane)

        v_split.setStretchFactor(0, 4)
        v_split.setStretchFactor(1, 1)
        v_split.setSizes([720, 220])

        root.addWidget(v_split, 1)

        # Status bar
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

        # The Stop button used to live here. It moved to the application
        # bar, beside Open; the progress bar stays, since that is status
        # rather than an action.
        self._prog = StageProgressBar(status_bar)
        sb_lay.addWidget(self._prog)

        div = QFrame(status_bar)
        div.setFixedWidth(1)
        div.setStyleSheet(f"background-color: {BORDER};")
        sb_lay.addWidget(div)

        self._ram_label = QLabel("RAM  --", status_bar)
        self._ram_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self._ram_label.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {PANEL2};"
        )
        sb_lay.addWidget(self._ram_label)

        root.addWidget(status_bar)

    # Tk StringVar-style shims

    # ``.set(...)`` call sites in ``actions.py`` and the tabs expect
    # ``tk.StringVar`` objects; these properties wrap QLabels instead.

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


class _ElidingLabel(QLabel):
    """A label that shortens its text from the middle instead of clipping.

    The file summary is "name | shape | dtype | spacing". A plain QLabel
    either forces the row as wide as that string -- pushing the tool
    buttons off a narrow window -- or clips the end, losing the voxel size.
    Eliding from the middle keeps both ends, which are the parts worth
    reading at a glance.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self._color = TEXT_DIM

    def set_color(self, color: str):
        self._color = color
        self.update()

    def setText(self, text: str):
        self._full_text = text or ""
        super().setText(self._full_text)
        self.update()

    def full_text(self) -> str:
        return self._full_text

    def minimumSizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(80, super().minimumSizeHint().height())

    def paintEvent(self, event):
        from PyQt6.QtGui import QFontMetrics, QPainter
        painter = QPainter(self)
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, self.width())
        painter.setPen(QColor(self._color))
        painter.drawText(self.rect(), int(self.alignment()), elided)


class _LabelVar:
    """Tiny shim that presents a QLabel with a ``tk.StringVar``-like API."""

    def __init__(self, label: QLabel):
        self._lbl = label

    def get(self) -> str:
        return self._lbl.text()

    def set(self, value) -> None:
        self._lbl.setText(str(value))
