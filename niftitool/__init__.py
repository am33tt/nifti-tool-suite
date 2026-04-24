"""NIfTI Tool Suite — SimReady.

A preprocessing and inspection toolkit for industrial/research CT NIfTI
volumes (concrete, ceramics, metals, additively-manufactured parts).

The package is organised so that each file covers one concept:

    config / deps / utils   — shared settings, optional-import flags, sysinfo
    core/                   — pure scientific functions (no GUI)
    ui/                     — tkinter front-end, one file per tab

Entry point: :func:`niftitool.main`.
"""

from pathlib import Path

from .ui_qt.app import NiftiApp

LOGO_PATH = Path(__file__).with_name("assets") / "logo.svg"


def _load_app_icon():
    """Return a QIcon built from the bundled cube-scan logo."""
    from PyQt6.QtGui import QIcon, QPixmap
    if not LOGO_PATH.exists():
        return QIcon()
    icon = QIcon(str(LOGO_PATH))
    if icon.isNull():
        # Fall back to a raster render if the SVG plugin isn't available.
        pm = QPixmap(str(LOGO_PATH))
        if not pm.isNull():
            icon = QIcon(pm)
    return icon


def _set_windows_app_id():
    """Ensure Windows uses our icon (not python.exe's) in the taskbar."""
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
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("NIfTI Tool Suite")
    app.setWindowIcon(_load_app_icon())
    win = NiftiApp()
    win.setWindowIcon(_load_app_icon())
    # Open maximised by default — the tri-planar + controls layout is
    # designed for a wide workspace and feels cramped below ~1400 px.
    win.showMaximized()
    sys.exit(app.exec())


__all__ = ["NiftiApp", "main"]
