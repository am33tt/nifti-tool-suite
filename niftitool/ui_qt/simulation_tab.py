"""Simulation tab — run mlhp linear-elastic FCM directly from the GUI.

Picks up the loaded NIfTI volume and (optionally) the computed E-Map
from ``self``, gathers solver parameters from a small form, then runs
:func:`niftitool.core.simulation.run_linear_elastic_in_memory` on a
worker thread so the UI stays responsive. Logs are forwarded to the
existing log pane via ``self._append_log``.

A second mode shells out to a pre-built C++ executable (``Run external
binary…``) for solvers that have no Python binding (steel_specimen,
voxel_fracturetest, …).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMessageBox, QSpinBox, QVBoxLayout, QWidget,
)

from ..config import ACCENT, BG, BORDER, PANEL, PANEL2, TEAL, TEXT, TEXT_DIM
from ..core.simulation import (
    SimParams, load_saved_mlhp_path, mlhp_available, run_external_executable,
    run_linear_elastic_in_memory, save_mlhp_path, try_import_mlhp,
)
from .widgets import hline, styled_btn, styled_entry


_FACE_LABELS = [
    ("0  -X face (left)",   0),
    ("1  +X face (right)",  1),
    ("2  -Y face (front)",  2),
    ("3  +Y face (back)",   3),
    ("4  -Z face (bottom)", 4),
    ("5  +Z face (top)",    5),
]


class SimulationTabMixin:
    """Adds the Simulation tab to :class:`NiftiApp`."""

    def _build_simulation_tab(self, parent):
        outer = QWidget(parent)
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(16, 12, 16, 12)
        outer_lay.setSpacing(6)

        root_lay = QVBoxLayout(parent)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.addWidget(outer)

        title = QLabel("Linear-Elastic Simulation (mlhp / FCM)", outer)
        title.setFont(QFont("Segoe UI Semibold", 11))
        title.setStyleSheet(f"color: {TEXT}; background-color: {BG};")
        outer_lay.addWidget(title)

        sub_text = (
            "Runs an mlhp finite-cell linear elasticity solve on the "
            "currently loaded volume. Output: a .pvtu in the chosen folder "
            "(open it in ParaView, or load the .nii.gz E-Map back into the "
            "3-D View tab)."
        )
        sub = QLabel(sub_text, outer)
        sub.setFont(QFont("Segoe UI", 9))
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {TEXT_DIM}; background-color: {BG};")
        outer_lay.addWidget(sub)

        # mlhp status row — shows whether the bindings are importable and
        # offers a folder picker to point at a custom build location.
        mlhp_row = QFrame(outer)
        mlhp_row.setStyleSheet(
            f"QFrame {{ background-color: {PANEL2}; border: 1px solid {BORDER}; }}"
        )
        mr_lay = QHBoxLayout(mlhp_row)
        mr_lay.setContentsMargins(10, 6, 10, 6)
        mr_lay.setSpacing(8)

        self._mlhp_status_lbl = QLabel("", mlhp_row)
        self._mlhp_status_lbl.setFont(QFont("Segoe UI", 9))
        self._mlhp_status_lbl.setWordWrap(True)
        mr_lay.addWidget(self._mlhp_status_lbl, 1)

        mr_lay.addWidget(styled_btn(
            mlhp_row, "Locate mlhp folder…",
            self._sim_pick_mlhp_folder, small=True,
        ))
        outer_lay.addWidget(mlhp_row)

        # Show the right colour/text on first paint.
        self._sim_refresh_mlhp_status()

        outer_lay.addWidget(hline(outer))

        # Parameter form
        form = QFrame(outer)
        form.setStyleSheet(
            f"QFrame {{ background-color: {PANEL}; border: 1px solid {BORDER}; }}"
        )
        grid = QGridLayout(form)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        def _label(text):
            q = QLabel(text, form)
            q.setStyleSheet(f"color: {TEXT}; background-color: {PANEL};")
            q.setFont(QFont("Segoe UI", 9))
            return q

        row = 0

        # Mesh
        grid.addWidget(_label("Elements (Nx, Ny, Nz):"), row, 0)
        self._sim_nx = QSpinBox(form); self._sim_nx.setRange(1, 512); self._sim_nx.setValue(8)
        self._sim_ny = QSpinBox(form); self._sim_ny.setRange(1, 512); self._sim_ny.setValue(8)
        self._sim_nz = QSpinBox(form); self._sim_nz.setRange(1, 512); self._sim_nz.setValue(8)
        nrow = QHBoxLayout(); nrow.setSpacing(4)
        for w in (self._sim_nx, self._sim_ny, self._sim_nz):
            nrow.addWidget(w)
        nrow.addStretch(1)
        nbox = QWidget(form); nbox.setLayout(nrow)
        nbox.setStyleSheet(f"background-color: {PANEL};")
        grid.addWidget(nbox, row, 1)
        row += 1

        grid.addWidget(_label("Polynomial degree:"), row, 0)
        self._sim_p = QSpinBox(form); self._sim_p.setRange(1, 6); self._sim_p.setValue(2)
        grid.addWidget(self._sim_p, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        grid.addWidget(_label("Refinement depth:"), row, 0)
        self._sim_refdepth = QSpinBox(form); self._sim_refdepth.setRange(0, 6); self._sim_refdepth.setValue(1)
        grid.addWidget(self._sim_refdepth, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        # Material
        grid.addWidget(_label("Young's modulus E [Pa]:"), row, 0)
        self._sim_E = styled_entry(form, width=18)
        self._sim_E.setText("2.0e11")
        grid.addWidget(self._sim_E, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        grid.addWidget(_label("Poisson's ratio ν:"), row, 0)
        self._sim_nu = QDoubleSpinBox(form)
        self._sim_nu.setDecimals(3); self._sim_nu.setRange(0.0, 0.499)
        self._sim_nu.setSingleStep(0.05); self._sim_nu.setValue(0.3)
        grid.addWidget(self._sim_nu, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        self._sim_use_emap = QCheckBox("Use computed E-Map as modulus field", form)
        self._sim_use_emap.setStyleSheet(f"color: {TEXT}; background-color: {PANEL};")
        grid.addWidget(self._sim_use_emap, row, 0, 1, 2)
        row += 1

        # Body force
        grid.addWidget(_label("Body force [N/m³] (fx, fy, fz):"), row, 0)
        self._sim_fx = styled_entry(form, width=8); self._sim_fx.setText("0.0")
        self._sim_fy = styled_entry(form, width=8); self._sim_fy.setText("0.0")
        self._sim_fz = styled_entry(form, width=8); self._sim_fz.setText("-20.0")
        frow = QHBoxLayout(); frow.setSpacing(4)
        for w in (self._sim_fx, self._sim_fy, self._sim_fz):
            frow.addWidget(w)
        frow.addStretch(1)
        fbox = QWidget(form); fbox.setLayout(frow)
        fbox.setStyleSheet(f"background-color: {PANEL};")
        grid.addWidget(fbox, row, 1)
        row += 1

        # BCs
        grid.addWidget(_label("Zero-displacement face:"), row, 0)
        self._sim_face = QComboBox(form)
        for label, _idx in _FACE_LABELS:
            self._sim_face.addItem(label)
        self._sim_face.setCurrentIndex(0)
        grid.addWidget(self._sim_face, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        # Solver
        grid.addWidget(_label("CG max iterations:"), row, 0)
        self._sim_cg_iter = QSpinBox(form); self._sim_cg_iter.setRange(10, 100_000); self._sim_cg_iter.setValue(1000)
        grid.addWidget(self._sim_cg_iter, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        grid.addWidget(_label("FCM α (outside penalty):"), row, 0)
        self._sim_alpha = styled_entry(form, width=12); self._sim_alpha.setText("1e-8")
        grid.addWidget(self._sim_alpha, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        # Output dir
        grid.addWidget(_label("Output directory:"), row, 0)
        self._sim_outdir = styled_entry(form, width=40)
        self._sim_outdir.setText(str(Path.cwd() / "outputs"))
        out_row = QHBoxLayout(); out_row.setSpacing(4)
        out_row.addWidget(self._sim_outdir, 1)
        out_row.addWidget(styled_btn(form, "Browse…", self._sim_pick_outdir, small=True))
        out_box = QWidget(form); out_box.setLayout(out_row)
        out_box.setStyleSheet(f"background-color: {PANEL};")
        grid.addWidget(out_box, row, 1)
        row += 1

        outer_lay.addWidget(form)

        # Run buttons
        btn_row = QFrame(outer)
        btn_row.setStyleSheet(f"background-color: {BG};")
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 6, 0, 6)
        bl.setSpacing(8)
        self._sim_run_btn = styled_btn(
            btn_row, "▶ Run Linear-Elastic Simulation",
            self._sim_run_in_memory, accent=True,
        )
        bl.addWidget(self._sim_run_btn)
        bl.addWidget(styled_btn(
            btn_row, "Run external binary…",
            self._sim_run_external, teal=True,
        ))
        bl.addStretch(1)
        outer_lay.addWidget(btn_row)

        # Result line
        self._sim_result_lbl = QLabel("No run yet.", outer)
        self._sim_result_lbl.setFont(QFont("Consolas", 9))
        self._sim_result_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; background-color: {BG};"
        )
        outer_lay.addWidget(self._sim_result_lbl)

        outer_lay.addStretch(1)

    # actions

    def _sim_refresh_mlhp_status(self):
        """Recolour the status row based on whether mlhp can be imported."""
        if mlhp_available():
            saved = load_saved_mlhp_path()
            tail = f"  [from: {saved}]" if saved else ""
            self._mlhp_status_lbl.setText(f"✓ mlhp is importable.{tail}")
            self._mlhp_status_lbl.setStyleSheet(
                f"color: #FFFFFF; background-color: {TEAL}; padding: 4px 6px;"
            )
        else:
            self._mlhp_status_lbl.setText(
                "mlhp is not importable. Click 'Locate mlhp folder…' and "
                "pick the folder containing mlhp.py / pymlhpcore (e.g. "
                "scanbasedanalysis/external/mlhp/src/python or your build "
                "output directory)."
            )
            self._mlhp_status_lbl.setStyleSheet(
                f"color: #FFFFFF; background-color: #C08020; padding: 4px 6px;"
            )

    def _sim_pick_mlhp_folder(self):
        """Folder picker → sys.path-prepend → re-attempt ``import mlhp``."""
        start = load_saved_mlhp_path() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(
            self, "Pick the folder containing mlhp.py / pymlhpcore", start,
        )
        if not folder:
            return

        ok, msg = try_import_mlhp(folder)
        if ok:
            save_mlhp_path(folder)
            self._append_log(f"  mlhp located: {folder}", 'ok')
            self._append_log(f"  {msg}", 'dim')
        else:
            self._append_log(f"  {msg}", 'err')
            QMessageBox.critical(
                self, "Could not import mlhp",
                f"{msg}\n\nMake sure the folder contains mlhp.py and a "
                f"compiled pymlhpcore extension (.pyd / .so) — usually the "
                f"CMake build output for scanbasedanalysis/external/mlhp.",
            )
        self._sim_refresh_mlhp_status()

    def _sim_pick_outdir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Pick output directory", self._sim_outdir.text() or os.getcwd(),
        )
        if d:
            self._sim_outdir.setText(d)

    def _sim_collect_params(self) -> SimParams | None:
        """Read the form into a SimParams. Returns None on validation error."""
        try:
            E_pa = float(self._sim_E.text())
            fx = float(self._sim_fx.text())
            fy = float(self._sim_fy.text())
            fz = float(self._sim_fz.text())
            alpha = float(self._sim_alpha.text())
        except ValueError as ex:
            QMessageBox.critical(self, "Invalid input",
                                 f"Could not parse a numeric field: {ex}")
            return None

        out_dir = (self._sim_outdir.text() or "outputs").strip()
        # mlhp's PVtuOutput appends ".pvtu", so we pass a stem without extension.
        out_stem = str(Path(out_dir) / "linear_elasticity")

        face_idx = self._sim_face.currentIndex()
        _label, fixed_face = _FACE_LABELS[face_idx]

        return SimParams(
            nelements=(self._sim_nx.value(), self._sim_ny.value(), self._sim_nz.value()),
            polynomial_degree=self._sim_p.value(),
            refinement_depth=self._sim_refdepth.value(),
            youngs_modulus_pa=E_pa,
            poisson_ratio=float(self._sim_nu.value()),
            body_force=(fx, fy, fz),
            fixed_face=fixed_face,
            alpha_fcm=alpha,
            cg_max_iter=self._sim_cg_iter.value(),
            use_emap_as_E=bool(self._sim_use_emap.isChecked()),
            output_stem=out_stem,
        )

    def _sim_run_in_memory(self):
        if self._img is None or self._gray is None:
            QMessageBox.warning(self, "No volume",
                                "Open a NIfTI file first.")
            return
        if not mlhp_available():
            reply = QMessageBox.question(
                self, "Locate mlhp",
                "The 'mlhp' Python module isn't importable yet.\n\n"
                "Pick the folder containing mlhp.py / pymlhpcore now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._sim_pick_mlhp_folder()
            if not mlhp_available():
                # User cancelled or pointed at the wrong folder.
                return

        params = self._sim_collect_params()
        if params is None:
            return

        if params.use_emap_as_E and self._E_map is None:
            QMessageBox.warning(
                self, "No E-Map",
                "Compute the E-Map first, or uncheck 'Use computed E-Map "
                "as modulus field'.",
            )
            return

        # Snapshot the inputs while we're still on the GUI thread.
        img = self._img
        gray = self._get_gray()
        emap = self._E_map if params.use_emap_as_E else None

        self._log_sep("Linear-elastic simulation (mlhp)")
        self._set_status("Running mlhp simulation…", busy=True)
        self._sim_run_btn.setEnabled(False)
        self._sim_result_lbl.setText("Running…")

        def _log(msg: str, tag: str = ''):
            # Thread-safe via _post_log_signal inside _append_log.
            self._append_log(msg, tag)

        def _worker():
            try:
                result = run_linear_elastic_in_memory(
                    img, gray, params, emap=emap, log=_log,
                )
                self.after(0, lambda: self._sim_on_done(result))
            except Exception as ex:
                self._append_log(f"  Simulation failed: {ex}", 'err')
                self.after(0, lambda: self._sim_on_error(ex))

        threading.Thread(target=_worker, daemon=True).start()

    def _sim_on_done(self, result: dict):
        self._sim_run_btn.setEnabled(True)
        self._set_status("Simulation complete.", busy=False)
        out = result.get("output_pvtu", "?")
        iters = result.get("cg_iterations", "?")
        resid = result.get("cg_final_residual", float('nan'))
        self._sim_result_lbl.setText(
            f"OK  →  {out}   (CG iters: {iters}, residual: {resid:.2e})"
        )
        self._append_log(f"  → {out}", 'ok')

    def _sim_on_error(self, ex: Exception):
        self._sim_run_btn.setEnabled(True)
        self._set_status("Simulation failed.", busy=False)
        self._sim_result_lbl.setText(f"FAILED  ({type(ex).__name__}: {ex})")

    def _sim_run_external(self):
        """Spawn a pre-compiled mlhp solver (e.g. steel_specimen.exe).

        The selected executable is launched with the chosen output
        directory as its working directory, so any files it writes
        (.pvtu, .vtu, logs) land where the user expects.
        """
        exe, _ = QFileDialog.getOpenFileName(
            self, "Pick a compiled mlhp/fracture executable",
            "",
            "Executables (*.exe);;All files (*.*)",
        )
        if not exe:
            return

        out_dir = (self._sim_outdir.text() or "outputs").strip()
        os.makedirs(out_dir, exist_ok=True)

        self._log_sep(f"External solver: {Path(exe).name}")
        self._set_status("Running external solver…", busy=True)

        def _log(msg: str, tag: str = ''):
            self._append_log(msg, tag)

        def _worker():
            try:
                rc = run_external_executable(exe, cwd=out_dir, log=_log)
                ok = (rc == 0)
                self.after(0, lambda: self._set_status(
                    "External solver done." if ok else "External solver failed.",
                    busy=False,
                ))
            except Exception as ex:
                self._append_log(f"  Spawn failed: {ex}", 'err')
                self.after(0, lambda: self._set_status(
                    "External solver failed.", busy=False,
                ))

        threading.Thread(target=_worker, daemon=True).start()
