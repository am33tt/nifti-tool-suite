"""NIfTI tool suite.

Preprocessing and inspection toolkit for industrial and research CT NIfTI
volumes (concrete, ceramics, metals, additively manufactured parts).

Package layout:

    config, deps, utils   shared settings, optional-import flags, sysinfo
    core/                 scientific functions, no GUI
    ui_qt/                PyQt6 front-end, one file per tab

Entry point: :func:`niftitool.main`.
"""

from pathlib import Path

LOGO_PATH = Path(__file__).with_name("assets") / "logo.svg"


def __getattr__(name):
    """Import the Qt window only when it is actually asked for.

    ``niftitool.core`` is pure NumPy and SciPy and is usable on its own in
    batch scripts, cluster jobs and notebooks. Importing the GUI at package
    level would make those fail on a missing PyQt6 or OpenGL library, so the
    window is resolved lazily (PEP 562).
    """
    if name == "NiftiApp":
        from .ui_qt.app import NiftiApp as _NiftiApp
        return _NiftiApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_app_icon():
    """Return a QIcon built from the bundled logo SVG.

    The SVG is rasterised at each standard icon size, because Windows
    selects a taskbar pixmap by exact size match and otherwise falls back to
    the generic python.exe icon.
    """
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtGui import QIcon, QPainter, QPixmap
    from PyQt6.QtSvg import QSvgRenderer

    if not LOGO_PATH.exists():
        return QIcon()

    icon = QIcon()
    renderer = QSvgRenderer(str(LOGO_PATH))
    if renderer.isValid():
        for size in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256):
            pm = QPixmap(QSize(size, size))
            pm.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pm)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(pm)
    else:
        pm = QPixmap(str(LOGO_PATH))
        if not pm.isNull():
            icon.addPixmap(pm)
    return icon


def _set_windows_app_id():
    """Set an explicit AppUserModelID so Windows shows the application icon.

    No-op off Windows. Without it the taskbar groups the process under
    python.exe.
    """
    import sys
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "niftitool.simready.1"
        )
    except Exception:
        pass


def main() -> None:
    """Launch the GUI."""
    import sys
    from PyQt6.QtWidgets import QApplication
    from .ui_qt.app import NiftiApp
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("NIfTI Tool Suite")
    app.setWindowIcon(_load_app_icon())
    win = NiftiApp()
    win.setWindowIcon(_load_app_icon())
    # The tri-planar and controls layout assumes a wide workspace.
    win.showMaximized()
    sys.exit(app.exec())


__all__ = ["NiftiApp", "main"]
