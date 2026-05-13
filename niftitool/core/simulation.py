"""mlhp-backed linear-elastic FCM simulation driver.

Mirrors ``examples/ctScanLinearElastic.py`` from the scanbasedanalysis
repo, but takes its inputs (voxel volume, spacing, origin, material
fields) from the running GUI's in-memory state rather than from disk.

The pipeline:
  1. Build an FCM scaling field from the loaded voxel volume.
  2. Build a refined hp grid + tensor-product basis.
  3. Apply zero-Dirichlet BCs on the user-selected face.
  4. Integrate K and f over the FCM domain (constant E and nu, or a
     voxel-data-backed E field if the GUI has an E-Map ready).
  5. Solve with CG + additive-Schwarz preconditioner.
  6. Postprocess to a .pvtu (PVtuOutput) for ParaView / the 3-D tab.

Two entry points:
  ``run_linear_elastic_in_memory`` — calls mlhp directly in this process.
  ``run_external_executable``       — spawns a compiled fracture/steel
                                      driver via subprocess; the GUI
                                      passes a ct.hpp through stdin /
                                      file path so the C++ driver can
                                      pick up the calibrated material.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


LogFn = Callable[[str, str], None]  # (message, tag) — tag in {'', 'ok', 'err', 'warn', 'dim', 'teal'}


# Path to the JSON file remembering the last-used mlhp folder so the user
# only has to pick it once.
_CONFIG_DIR = Path.home() / ".niftitool"
_MLHP_CONFIG = _CONFIG_DIR / "mlhp_path.json"


def load_saved_mlhp_path() -> Optional[str]:
    """Return the path the user previously picked, or None."""
    try:
        if _MLHP_CONFIG.exists():
            data = json.loads(_MLHP_CONFIG.read_text(encoding="utf-8"))
            p = data.get("folder")
            if p and Path(p).exists():
                return p
    except Exception:
        pass
    return None


def save_mlhp_path(folder: str) -> None:
    """Persist ``folder`` so the next launch can re-import mlhp silently."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _MLHP_CONFIG.write_text(
            json.dumps({"folder": str(folder)}, indent=2), encoding="utf-8",
        )
    except Exception:
        # Persistence is best-effort; never fail the run because of it.
        pass


def _discover_mlhp_paths(folder: str) -> list[str]:
    """Find folders inside ``folder`` that hold ``mlhp.py`` or the
    compiled ``pymlhpcore`` extension, so we can ``sys.path``-prepend
    each one. The user might point at the repo root, the build folder,
    or the python-bindings source dir — we accept any of those.
    """
    root = Path(folder)
    if not root.exists():
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            found.append(s)

    if (root / "mlhp.py").exists() or any(root.glob("pymlhpcore*")):
        _add(root)

    # Limit recursion — bindings live near the top of the tree in practice.
    for pattern in ("mlhp.py", "pymlhpcore*.pyd", "pymlhpcore*.so",
                    "pymlhpcore*.dll"):
        for hit in root.rglob(pattern):
            try:
                depth = len(hit.relative_to(root).parts)
            except ValueError:
                continue
            if depth > 6:
                continue
            _add(hit.parent)

    return found


def try_import_mlhp(folder: Optional[str] = None) -> tuple[bool, str]:
    """Attempt to import the ``mlhp`` module.

    If ``folder`` is given, scan it for ``mlhp.py`` / ``pymlhpcore*``
    and prepend each match to ``sys.path`` before trying. Returns
    ``(ok, message)`` — ``message`` is human-readable for the GUI log.
    """
    # Already importable? Nothing to do.
    if "mlhp" in sys.modules:
        return True, "mlhp already imported"

    if folder:
        paths = _discover_mlhp_paths(folder)
        if not paths:
            return (
                False,
                f"No mlhp.py or pymlhpcore extension found under {folder}",
            )
        for p in paths:
            if p not in sys.path:
                sys.path.insert(0, p)

    try:
        import mlhp  # noqa: F401
        return True, "mlhp imported successfully"
    except ImportError as ex:
        return False, f"mlhp import failed: {ex}"
    except Exception as ex:  # pragma: no cover — surface unexpected errors
        return False, f"mlhp import raised {type(ex).__name__}: {ex}"


def _noop_log(_msg: str, _tag: str = '') -> None:
    pass


@dataclass
class SimParams:
    """User-configurable knobs for the linear-elastic FCM run."""

    nelements: tuple[int, int, int] = (8, 8, 8)
    polynomial_degree: int = 2
    refinement_depth: int = 1

    # Material — used when ``use_emap_as_E`` is False, or as units when True.
    youngs_modulus_pa: float = 200e9
    poisson_ratio: float = 0.3

    # Body force [N/m^3] (gravity-style by default).
    body_force: tuple[float, float, float] = (0.0, 0.0, -20.0)

    # Zero-Dirichlet face index in mlhp's convention (0 = -x face).
    fixed_face: int = 0

    # FCM penalty for the "outside" region — keeps K nonsingular without
    # actually loading void elements.
    alpha_fcm: float = 1e-8

    # Quadrature subdivision per element (one int per axis).
    grid_quadrature: tuple[int, int, int] = (10, 10, 10)

    # CG tolerance / iteration cap.
    cg_max_iter: int = 1000

    # If True and an E-Map is supplied, the per-voxel modulus replaces the
    # constant ``youngs_modulus_pa`` (still in Pa — caller must scale MPa→Pa).
    use_emap_as_E: bool = False

    # Output path stem (no extension); mlhp adds ``.pvtu``.
    output_stem: str = "outputs/linear_elasticity"

    # Extra metadata recorded into the log only.
    notes: str = field(default="")


def mlhp_available() -> bool:
    """Return True if the ``mlhp`` Python module is importable.

    Honours a previously-saved folder (see :func:`load_saved_mlhp_path`)
    so a user who picked the path in an earlier run doesn't have to
    pick it again.
    """
    if "mlhp" in sys.modules:
        return True
    saved = load_saved_mlhp_path()
    ok, _msg = try_import_mlhp(saved) if saved else try_import_mlhp(None)
    return ok


def _voxel_origin_from_affine(affine) -> list[float]:
    """Pull the (x,y,z) world-space origin out of a 4×4 NIfTI affine."""
    return [float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3])]


def run_linear_elastic_in_memory(
    img,                              # nibabel.Nifti1Image
    voxel_volume,                     # numpy.ndarray (3-D, float-castable)
    params: SimParams,
    *,
    emap=None,                        # Optional[np.ndarray] — modulus per voxel
    log: LogFn = _noop_log,
) -> dict:
    """Run a linear-elastic FCM analysis on ``voxel_volume``.

    Returns a dict with summary stats and the output file path. Raises
    on any solver / setup failure — callers should wrap in try/except
    and route exceptions to the GUI log.
    """
    import mlhp
    import numpy as np

    D = 3

    spacing = [float(s) for s in img.header.get_zooms()[:D]]
    nvoxels = [int(n) for n in voxel_volume.shape[:D]]
    lengths = [nvoxels[i] * spacing[i] for i in range(D)]

    # Align the voxel data to the mesh's own [0, lengths] frame.
    # NIfTI's affine origin is for anatomical / world-space registration —
    # using it here produces a mesh / FCM mismatch when the affine offset
    # is non-zero (you see the volume occupy only the overlap region with
    # the rest of the mesh sitting in the FCM "outside" α ~ 1e-8 phase).
    origin = [0.0, 0.0, 0.0]
    affine_origin = _voxel_origin_from_affine(img.affine)

    log("0. Reading volume into mlhp scaling field", 'dim')
    log(f"   shape          : {nvoxels}", 'dim')
    log(f"   spacing        : {spacing} mm", 'dim')
    log(f"   lengths        : {lengths} mm", 'dim')
    log(f"   FCM origin     : {origin} (mesh-aligned)", 'dim')
    if any(abs(x) > 1e-9 for x in affine_origin):
        log(
            f"   note: NIfTI affine origin {affine_origin} ignored — "
            f"using (0,0,0) so the voxel data fills the whole mesh.",
            'dim',
        )

    flat = np.ascontiguousarray(voxel_volume, dtype=np.float64).ravel()
    voxel_data = mlhp.DoubleVector(flat)
    scaling = mlhp.scalarFieldFromVoxelData(
        voxel_data, nvoxels, lengths,
        origin=origin, outside=params.alpha_fcm,
    )

    log("1. Building mesh and basis", 'dim')
    grid = mlhp.makeRefinedGrid(list(params.nelements), lengths)
    basis = mlhp.makeHpTensorSpace(grid, params.polynomial_degree, nfields=D)

    log("2. Dirichlet BCs", 'dim')
    dirichlet = mlhp.integrateDirichletDofs(
        mlhp.vectorField(D, [0.0] * D), basis, [int(params.fixed_face)],
    )

    log("3. Material + integrand", 'dim')
    if params.use_emap_as_E and emap is not None:
        # emap is in MPa from the GUI; mlhp expects consistent units —
        # if the user picked Pa for body force, scale here.
        # Same origin convention as the scaling field above — the E-Map
        # voxel grid must share the mesh's (0,0,0)-anchored frame, not
        # NIfTI world-space, otherwise modulus values land in the wrong
        # cells.
        emap_flat = np.ascontiguousarray(emap, dtype=np.float64).ravel()
        E_field = mlhp.scalarFieldFromVoxelData(
            mlhp.DoubleVector(emap_flat), nvoxels, lengths,
            origin=origin, outside=params.alpha_fcm,
        )
        E = E_field
        log(f"   E field : voxel-backed (E-Map, {emap.shape})", 'teal')
    else:
        E = mlhp.scalarField(D, float(params.youngs_modulus_pa))
        log(f"   E       : {params.youngs_modulus_pa:.3e} (constant)", 'dim')

    nu = mlhp.scalarField(D, float(params.poisson_ratio))
    rhs = mlhp.vectorField(D, list(params.body_force))

    kinematics = mlhp.smallStrainKinematics(D)
    constitutive = mlhp.isotropicElasticMaterial(E, nu)
    integrand = mlhp.staticDomainIntegrand(kinematics, constitutive, rhs)

    log("4. Allocating sparse system", 'dim')
    matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
    vector = mlhp.allocateRhsVector(matrix)

    log("5. Integrating system (this is the slow step)", 'dim')
    quadrature = mlhp.gridQuadrature(list(params.grid_quadrature), scaling)
    mlhp.integrateOnDomain(
        basis, integrand, [matrix, vector],
        dirichletDofs=dirichlet, quadrature=quadrature,
    )

    log("6. Solving with CG + additive Schwarz", 'dim')
    P = mlhp.additiveSchwarzPreconditioner(matrix, basis, dirichlet[0])
    interior, norms = mlhp.cg(
        matrix, vector, M=P, maxiter=int(params.cg_max_iter),
        residualNorms=True,
    )
    all_dofs = mlhp.inflateDofs(interior, dirichlet)

    iters = len(norms) if hasattr(norms, '__len__') else int(params.cg_max_iter)
    final_resid = float(norms[-1]) if iters else float('nan')
    log(f"   CG iterations: {iters}   final residual: {final_resid:.3e}", 'ok')

    log("7. Postprocessing → PVtuOutput", 'dim')
    out_stem = params.output_stem
    out_dir = os.path.dirname(out_stem)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processors = [
        mlhp.solutionProcessor(D, all_dofs, "Displacement"),
        mlhp.functionProcessor(scaling, "MaskField"),
        mlhp.vonMisesProcessor(all_dofs, kinematics, constitutive, "VonMises"),
    ]
    postmesh = mlhp.gridCellMesh([params.polynomial_degree + 3] * D)
    output = mlhp.PVtuOutput(filename=out_stem)
    mlhp.basisOutput(basis, postmesh, output, processors)

    pvtu_path = f"{out_stem}.pvtu"
    log(f"   Wrote: {pvtu_path}", 'ok')

    return {
        "output_pvtu": pvtu_path,
        "cg_iterations": iters,
        "cg_final_residual": final_resid,
        "nelements": list(params.nelements),
        "polynomial_degree": params.polynomial_degree,
        "spacing_mm": spacing,
        "lengths_mm": lengths,
        "origin_mm": origin,
    }


def run_external_executable(
    exe_path: str,
    *,
    cwd: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    env: Optional[dict] = None,
    log: LogFn = _noop_log,
) -> int:
    """Spawn a compiled mlhp/fracture driver and stream its stdout to ``log``.

    Use this for solvers that exist only as C++ binaries (e.g. the
    ``steel_specimen`` / ``voxel_fracturetest`` executables in the
    scanbasedanalysis repo). The GUI is expected to have already
    written any required ``ct.hpp`` / input files into ``cwd``.

    Returns the process exit code.
    """
    exe = Path(exe_path)
    if not exe.exists():
        raise FileNotFoundError(f"Executable not found: {exe}")

    cmd = [str(exe), *(extra_args or [])]
    log(f"   $ {' '.join(cmd)}", 'dim')

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"   | {line}", '')

    rc = proc.wait()
    if rc == 0:
        log(f"   exit code: {rc}", 'ok')
    else:
        log(f"   exit code: {rc}", 'err')
    return rc
