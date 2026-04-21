"""NIfTI Tool Suite — SimReady.

A preprocessing and inspection toolkit for industrial/research CT NIfTI
volumes (concrete, ceramics, metals, additively-manufactured parts).

The package is organised so that each file covers one concept:

    config / deps / utils   — shared settings, optional-import flags, sysinfo
    core/                   — pure scientific functions (no GUI)
    ui/                     — tkinter front-end, one file per tab

Entry point: :func:`niftitool.main`.
"""

from .ui_qt.app import NiftiApp


def main() -> None:
    """Launch the GUI."""
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = NiftiApp()
    win.show()
    sys.exit(app.exec())


__all__ = ["NiftiApp", "main"]
