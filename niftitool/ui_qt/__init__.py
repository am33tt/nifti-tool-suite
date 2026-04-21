"""PyQt6 + VTK front-end for NIfTI Tool Suite.

Mirror of :mod:`niftitool.ui` (tkinter) with the same mixin architecture
and the same attribute / method names on ``self``, so
:mod:`niftitool.core` business logic can be reused unchanged.
"""

from .app import NiftiApp

__all__ = ["NiftiApp"]
