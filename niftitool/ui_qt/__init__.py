"""PyQt6 + VTK front-end for NIfTI Tool Suite.

Mirrors niftitool.ui (tkinter) using the same mixin layout and method
names, so niftitool.core logic can be reused without changes.
"""

from .app import NiftiApp

__all__ = ["NiftiApp"]
