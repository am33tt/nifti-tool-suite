"""Export & Report tab.

One clean place to save everything the app has computed:

* the multi-page PDF analysis report,
* per-void measurements as CSV,
* per-phase statistics as CSV,
* the segmentation label mask as NIfTI (openable in any viewer).
"""

from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..config import BG, BORDER, PANEL, TEXT, TEXT_DIM
from .widgets import hline, styled_btn


class ExportTabMixin:
    """Adds the Export & Report tab to :class:`NiftiApp`."""

    def _build_export_tab(self, parent):
        outer = QWidget(parent)
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(16, 12, 16, 12)
        outer_lay.setSpacing(6)

        root_lay = QVBoxLayout(parent)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.addWidget(outer)

        title = QLabel("Export & Report", outer)
        title.setFont(QFont("Segoe UI Semibold", 11))
        title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        outer_lay.addWidget(title)

        sub = QLabel(
            "Save what you've computed. The PDF report bundles everything; "
            "the CSVs hold the raw numbers.", outer,
        )
        sub.setFont(QFont("Segoe UI", 9))
        sub.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        sub.setWordWrap(True)
        outer_lay.addWidget(sub)

        outer_lay.addWidget(hline(outer))

        for label, tip, cmd, accent in [
            ("Generate PDF Report",
             "Multi-page report: slices, histogram, phases, void analytics",
             self._do_generate_report, True),
            ("Export Voids CSV...",
             "One row per void: volume, diameter, sphericity, position",
             self._export_voids_csv, False),
            ("Export Phase Statistics CSV",
             "Per-phase volume fraction and intensity statistics",
             self._export_stats_csv, False),
            ("Export Phase Labels as NIfTI",
             "uint8 mask (0 = void, 1 = matrix, 2 = aggregate) — opens in "
             "any NIfTI viewer",
             self._export_labels_nifti, False),
        ]:
            btn_f = QFrame(outer)
            btn_f.setStyleSheet(
                f"QFrame {{ background-color: {PANEL}; border: 1px solid {BORDER}; }}"
            )
            bf_lay = QHBoxLayout(btn_f)
            bf_lay.setContentsMargins(12, 8, 12, 8)
            bf_lay.setSpacing(12)
            bf_lay.addWidget(
                styled_btn(btn_f, label, cmd, accent=accent, teal=not accent)
            )
            tip_lbl = QLabel(tip, btn_f)
            tip_lbl.setFont(QFont("Segoe UI", 9))
            tip_lbl.setStyleSheet(
                f"color: {TEXT_DIM}; background-color: {PANEL};"
            )
            tip_lbl.setWordWrap(True)
            bf_lay.addWidget(tip_lbl, 1)
            outer_lay.addWidget(btn_f)

        outer_lay.addStretch(1)
