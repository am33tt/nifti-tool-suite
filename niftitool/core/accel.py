"""Optional GPU acceleration for the filter kernels.

Routes the neighbourhood-filter/morphology calls used elsewhere in this
package (Gaussian smoothing, the EDT, resampling, connected-component
labelling) to `cupyx.scipy.ndimage` when a CUDA device is usable and the
operands fit in free VRAM, and to SciPy otherwise. Selection is per call
(see :func:`would_use_gpu`); a CUDA OOM frees the pool, retries once, then
falls back to the CPU result. The GPU never holds a whole volume, only one
slab at a time.

Results agree with SciPy to rounding, not bit-for-bit; `label` ids are not
stable across backends, only sizes/masks are. For reproducible numbers, set
`NIFTITOOL_GPU=off` (or the GUI toggle).
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .. import config

# getattr, not from-import: stays loadable against an older config.py.
GPU_MODE_DEFAULT = getattr(config, "GPU_MODE_DEFAULT", "auto")
GPU_VRAM_FRACTION = getattr(config, "GPU_VRAM_FRACTION", 0.6)

# Imported directly (not via ..deps) so a missing nibabel can't disable this.
try:
    import scipy.ndimage as _sndi
except ImportError:                                  # pragma: no cover
    _sndi = None                                     # type: ignore

__all__ = [
    "set_mode", "get_mode", "gpu_available", "gpu_info", "gpu_status_line",
    "free_vram_bytes", "total_vram_bytes", "would_use_gpu",
    "video_memory_mb", "fallback_notes",
    "gaussian_filter", "gaussian_filter1d", "map_coordinates",
    "distance_transform_edt", "median_filter", "uniform_filter",
    "label", "binary_fill_holes", "binary_erosion", "binary_dilation",
    "binary_closing", "binary_opening", "rotate", "last_backend",
]

#: Modes for :func:`set_mode`: ``auto`` (use GPU if it fits), ``on`` (auto,
#: but warn loudly on fallback), ``off`` (CPU only, for reproducible runs).
MODES = ("auto", "on", "off")

#: Fraction of *free* VRAM one call may claim; the rest covers the display
#: driver, the VTK viewer and allocator fragmentation.
VRAM_FRACTION = GPU_VRAM_FRACTION

#: Assumed working-set multiple of input size (copy + output + scratch),
#: deliberately pessimistic so a wrong guess costs a fallback, not an OOM.
WORKING_SET_FACTOR = 4.0

#: Below this, PCIe transfer + launch overhead exceeds the SciPy CPU cost.
MIN_GPU_BYTES = 4 * 1024 ** 2

_lock = threading.Lock()
_mode = os.environ.get("NIFTITOOL_GPU", GPU_MODE_DEFAULT).strip().lower()
if _mode not in MODES:
    _mode = "auto"

_probe_done = False
_cp = None          # the cupy module, when usable
_cndi = None        # cupyx.scipy.ndimage, when usable
_probe_error: str | None = None
_last_backend = "cpu"
_vtk_vram_mb: float | None = None   # cached dedicated VRAM, MiB (0.0 = none)
_fallback_notes: dict[str, str] = {}


# ---------------------------------------------------------------------------
#  probing
# ---------------------------------------------------------------------------

def _probe() -> None:
    """Import CuPy and confirm a device actually runs a kernel (import alone
    can succeed with no usable device)."""
    global _probe_done, _cp, _cndi, _probe_error
    if _probe_done:
        return
    _probe_done = True
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi
    except Exception as ex:                      # not installed, or no CUDA
        _probe_error = f"CuPy not available ({type(ex).__name__}: {ex})"
        return
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            _probe_error = "CuPy is installed but no CUDA device was found"
            return
        # Force a real launch and a real synchronisation.
        probe = cp.arange(8, dtype=cp.float32)
        value = float((probe * 2).sum())
        cp.cuda.Stream.null.synchronize()
        if value != 56.0:
            _probe_error = "CUDA device returned a wrong result on the probe"
            return
    except Exception as ex:
        _probe_error = f"CUDA device unusable ({type(ex).__name__}: {ex})"
        return
    _cp, _cndi = cp, cndi


def set_mode(mode: str) -> None:
    """Set the acceleration mode to one of :data:`MODES`."""
    global _mode
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _mode = mode


def get_mode() -> str:
    """The current acceleration mode."""
    return _mode


def gpu_available() -> bool:
    """True when a CUDA device is present and executes kernels correctly.

    Independent of the mode: this asks what the hardware can do, not what
    the user asked for.
    """
    with _lock:
        _probe()
    return _cp is not None


def free_vram_bytes() -> int | None:
    """Free device memory in bytes, or ``None`` without a usable GPU."""
    if not gpu_available():
        return None
    try:
        free, _total = _cp.cuda.Device().mem_info
        return int(free)
    except Exception:
        return None


def total_vram_bytes() -> int | None:
    """Total device memory in bytes, or ``None`` without a usable GPU."""
    if not gpu_available():
        return None
    try:
        _free, total = _cp.cuda.Device().mem_info
        return int(total)
    except Exception:
        return None


def video_memory_mb() -> tuple[float | None, float | None, str]:
    """Best available ``(free_mb, total_mb, source)`` for the display GPU.

    ``source`` is ``"cuda"`` (CuPy, free+total), ``"vtk"`` (dedicated total
    only, ``free_mb`` is ``None``), or ``"none"`` (fall back to a
    system-RAM budget). A dedicated pool under 256 MiB counts as ``"none"``.
    """
    free = free_vram_bytes()
    total = total_vram_bytes()
    if total:
        return (free / 1024 ** 2 if free else None, total / 1024 ** 2, "cuda")

    # The VTK probe walks the display adapters and is far too slow to run
    # from a 2-second GUI timer, and the answer is a hardware constant, so
    # it is done once and cached.
    global _vtk_vram_mb
    if _vtk_vram_mb is None:
        _vtk_vram_mb = 0.0
        try:
            from ..deps import HAS_VTK, vtk
            if HAS_VTK and hasattr(vtk, "vtkGPUInfoList"):
                gpus = vtk.vtkGPUInfoList()
                gpus.Probe()
                best = 0
                for i in range(gpus.GetNumberOfGPUs()):
                    best = max(
                        best,
                        int(gpus.GetGPUInfo(i).GetDedicatedVideoMemory()),
                    )
                if best >= 256 * 1024 ** 2:
                    _vtk_vram_mb = best / 1024 ** 2
        except Exception:
            pass
    if _vtk_vram_mb:
        return (None, _vtk_vram_mb, "vtk")

    return (None, None, "none")


def gpu_info() -> dict:
    """Describe the acceleration backend.

    Always returns a dictionary; ``available`` says whether the rest of the
    fields are meaningful, and ``reason`` explains a negative.
    """
    with _lock:
        _probe()
    info = {
        "available": _cp is not None,
        "mode": _mode,
        "reason": _probe_error,
        "name": None,
        "total_vram_mb": None,
        "free_vram_mb": None,
        "cupy_version": None,
        "cuda_version": None,
        "compute_capability": None,
    }
    if _cp is None:
        return info
    try:
        device = _cp.cuda.Device()
        props = _cp.cuda.runtime.getDeviceProperties(device.id)
        free, total = device.mem_info
        name = props["name"]
        info.update(
            name=name.decode() if isinstance(name, bytes) else str(name),
            total_vram_mb=total / 1024 ** 2,
            free_vram_mb=free / 1024 ** 2,
            cupy_version=_cp.__version__,
            cuda_version=_cp.cuda.runtime.runtimeGetVersion(),
            compute_capability="%d.%d" % device.compute_capability_major_minor
            if hasattr(device, "compute_capability_major_minor")
            else device.compute_capability,
        )
    except Exception:
        pass
    return info


def gpu_status_line() -> str:
    """One line describing the backend, for the log pane and reports."""
    info = gpu_info()
    if not info["available"]:
        return f"GPU acceleration: off ({info['reason']}); using CPU (SciPy)."
    if _mode == "off":
        return (f"GPU acceleration: disabled by setting "
                f"(device present: {info['name']}).")
    total = info["total_vram_mb"]
    free = info["free_vram_mb"]
    where = f"{info['name']}"
    if total is not None:
        where += f", {total / 1024:.1f} GiB VRAM ({free / 1024:.1f} GiB free)"
    return f"GPU acceleration: {_mode} on {where}, CuPy {info['cupy_version']}."


def last_backend() -> str:
    """``'gpu'`` or ``'cpu'``: where the most recent call actually ran.

    Useful in tests and in the log, so a run can state what produced its
    numbers rather than what was configured.
    """
    return _last_backend


# ---------------------------------------------------------------------------
#  dispatch
# ---------------------------------------------------------------------------

def _nbytes(args, kwargs) -> int:
    """Total bytes of every array operand in *args* and *kwargs*."""
    total = 0
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return total


def would_use_gpu(nbytes: int) -> bool:
    """Whether a kernel over *nbytes* of operands would run on the GPU.

    Exposed so callers can log the decision, or size a slab against it,
    without having to reproduce the rules.
    """
    if _mode == "off" or not gpu_available():
        return False
    if nbytes < MIN_GPU_BYTES:
        return False
    free = free_vram_bytes()
    if free is None:
        return False
    return nbytes * WORKING_SET_FACTOR <= free * VRAM_FRACTION


def _to_device(value):
    """A device copy of *value* when it is an array, else *value* itself."""
    if isinstance(value, np.ndarray):
        # Bit-packed masks and other exotic dtypes are not worth special
        # casing; CuPy handles the numeric dtypes this toolkit produces.
        return _cp.asarray(value)
    return value


def _to_host(value):
    """A host copy of *value* when it is a device array, else *value*."""
    if _cp is not None and isinstance(value, _cp.ndarray):
        return _cp.asnumpy(value)
    if isinstance(value, tuple):
        return tuple(_to_host(item) for item in value)
    return value


def _free_pool() -> None:
    """Release CuPy's cached blocks. Called before retrying after an OOM."""
    try:
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


#: Morphology kernels whose multi-iteration form CuPy implements only in
#: its brute-force variant.
_BRUTE_FORCE_KERNELS = frozenset({
    "binary_erosion", "binary_dilation", "binary_closing", "binary_opening",
})


def _gpu_kwargs(name: str, kwargs: dict) -> dict:
    """*kwargs* adjusted for the CuPy backend: sets ``brute_force=True`` for
    multi-iteration binary morphology, which CuPy implements only in that
    form (SciPy defaults to False; both give identical results, just at
    different speed). An explicit caller ``brute_force`` is respected."""
    if (name in _BRUTE_FORCE_KERNELS
            and int(kwargs.get("iterations", 1) or 1) != 1
            and "brute_force" not in kwargs):
        return {**kwargs, "brute_force": True}
    return kwargs


def _dispatch(name: str, args: tuple, kwargs: dict):
    """Run ``ndimage.<name>`` on the GPU when that is possible, else SciPy.

    Falls back to SciPy on any device-side failure, including running out
    of VRAM, so the caller sees one behaviour and one result type
    (:class:`numpy.ndarray`) regardless of the hardware.
    """
    global _last_backend
    if _sndi is None:
        raise ImportError(
            f"ndimage.{name} needs SciPy, which is not installed. "
            "Install it with: pip install scipy"
        )
    cpu_fn = getattr(_sndi, name)

    if not would_use_gpu(_nbytes(args, kwargs)):
        _last_backend = "cpu"
        return cpu_fn(*args, **kwargs)

    for attempt in (1, 2):
        try:
            gpu_args = tuple(_to_device(a) for a in args)
            gpu_kwargs = {k: _to_device(v)
                          for k, v in _gpu_kwargs(name, kwargs).items()}
            result = getattr(_cndi, name)(*gpu_args, **gpu_kwargs)
            host = _to_host(result)
            del gpu_args, gpu_kwargs, result
            _last_backend = "gpu"
            return host
        except Exception as ex:                       # noqa: BLE001
            is_oom = _cp is not None and isinstance(
                ex, (_cp.cuda.memory.OutOfMemoryError,))
            _free_pool()
            if is_oom and attempt == 1:
                continue                              # retry with a clean pool
            _note_fallback(name, ex)
            break

    _last_backend = "cpu"
    return cpu_fn(*args, **kwargs)


def _note_fallback(name: str, ex: Exception) -> None:
    """Record, once per kernel, why the GPU path was abandoned (so a
    silent slowdown shows up in the log instead)."""
    if name not in _fallback_notes:
        _fallback_notes[name] = f"{type(ex).__name__}: {ex}"


def fallback_notes() -> dict[str, str]:
    """Kernels that were asked to run on the GPU and could not, and why.
    Empty when everything attempted succeeded; a recovered OOM is not
    recorded, one that persisted is."""
    return dict(_fallback_notes)


# ---------------------------------------------------------------------------
#  the routed kernels
# ---------------------------------------------------------------------------
#  Each is a drop-in replacement for the identically named scipy.ndimage
#  function: same signature, same return type, NumPy arrays in and out.

def gaussian_filter(input, sigma, **kwargs):
    """:func:`scipy.ndimage.gaussian_filter`, GPU-accelerated when possible."""
    return _dispatch("gaussian_filter", (input, sigma), kwargs)


def gaussian_filter1d(input, sigma, **kwargs):
    """:func:`scipy.ndimage.gaussian_filter1d`, GPU when possible."""
    return _dispatch("gaussian_filter1d", (input, sigma), kwargs)


def map_coordinates(input, coordinates, **kwargs):
    """:func:`scipy.ndimage.map_coordinates`, GPU when possible."""
    return _dispatch("map_coordinates", (input, coordinates), kwargs)


def distance_transform_edt(input, **kwargs):
    """:func:`scipy.ndimage.distance_transform_edt`, GPU when possible.

    Both implementations are exact Euclidean transforms, so the results
    differ only by the rounding of the anisotropic ``sampling`` multiply.
    """
    return _dispatch("distance_transform_edt", (input,), kwargs)


def median_filter(input, **kwargs):
    """:func:`scipy.ndimage.median_filter`, GPU when possible."""
    return _dispatch("median_filter", (input,), kwargs)


def uniform_filter(input, **kwargs):
    """:func:`scipy.ndimage.uniform_filter`, GPU when possible."""
    return _dispatch("uniform_filter", (input,), kwargs)


def label(input, **kwargs):
    """:func:`scipy.ndimage.label`, GPU when possible.

    Returns ``(labels, n)`` as NumPy. The partition into components is the
    same on both backends, but the integer *ids* assigned to them are not
    guaranteed to match, so downstream code must use sizes and masks, never
    a label id compared across runs.
    """
    return _dispatch("label", (input,), kwargs)


def binary_fill_holes(input, **kwargs):
    """:func:`scipy.ndimage.binary_fill_holes`, GPU when possible."""
    return _dispatch("binary_fill_holes", (input,), kwargs)


def binary_erosion(input, **kwargs):
    """:func:`scipy.ndimage.binary_erosion`, GPU when possible."""
    return _dispatch("binary_erosion", (input,), kwargs)


def binary_dilation(input, **kwargs):
    """:func:`scipy.ndimage.binary_dilation`, GPU when possible."""
    return _dispatch("binary_dilation", (input,), kwargs)


def binary_closing(input, **kwargs):
    """:func:`scipy.ndimage.binary_closing`, GPU when possible."""
    return _dispatch("binary_closing", (input,), kwargs)


def binary_opening(input, **kwargs):
    """:func:`scipy.ndimage.binary_opening`, GPU when possible."""
    return _dispatch("binary_opening", (input,), kwargs)


def rotate(input, angle, **kwargs):
    """:func:`scipy.ndimage.rotate`, GPU when possible."""
    return _dispatch("rotate", (input, angle), kwargs)
