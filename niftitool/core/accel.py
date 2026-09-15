"""Optional GPU acceleration for the filter kernels.

The heavy steps of this toolkit are neighbourhood filters and morphology on
slabs of a few tens to a few hundreds of megabytes: Gaussian smoothing
before thresholding, the exact Euclidean distance transform behind the
beam-hardening depth field, bilinear resampling of the coarse correction
field back onto the full grid, and connected-component labelling. Those are
exactly the kernels `cupyx.scipy.ndimage` implements, so this module routes
them to the GPU when one is usable and to SciPy when it is not.

What this module does *not* change
----------------------------------
Nothing about how the volume is read, written or held. The slab-wise
streaming in :mod:`~niftitool.core.volume_stream` stays the memory model of
the whole toolkit, and the 16 GB target is unaffected: a slab is copied to
the device, filtered, and copied back. The GPU is a coprocessor for one
slab at a time, never a place a volume lives. Steps whose cost is disk I/O
-- background removal, the masked write-out, the streaming binarisation --
are bounded by the file, not by arithmetic, and will not get much faster.

Backend selection
-----------------
CuPy is the only backend. It requires an NVIDIA GPU with a CUDA driver (or
an AMD card under ROCm on Linux); on Intel or AMD integrated graphics no
compute backend is available, and every call here silently runs the SciPy
path. That is the intended behaviour, not a degraded mode: on integrated
graphics "VRAM" is system RAM behind a slower path, so offloading would
cost time rather than save it.

The choice is made per call, from three inputs:

* the user's mode (:func:`set_mode`, or the ``NIFTITOOL_GPU`` environment
  variable, one of ``auto`` / ``on`` / ``off``);
* whether a working device was found at first use;
* whether the operands plus a working margin fit in currently free VRAM.

A CUDA out-of-memory error is caught, the memory pool is released, the call
is retried once, and if it fails again the CPU result is returned. A caller
therefore never has to handle a GPU-specific failure.

Numerical agreement
-------------------
The GPU kernels implement the same algorithms, but floating-point
reductions run in a different order, so results agree to rounding rather
than bit for bit. Measured against SciPy on float32 input:

===========================  ======================================
kernel                       agreement
===========================  ======================================
``gaussian_filter``          ~1e-6 relative; a voxel whose smoothed
                             value sits exactly on a threshold can
                             fall to the other side of it
``map_coordinates`` order=1  ~1e-6 relative
``distance_transform_edt``   exact algorithm both sides; differences
                             only from the ``sampling`` multiply
``median_filter``            exact (selection, not arithmetic)
``binary_*`` morphology      exact
``label``                    identical components; the *numbering*
                             of the components is not guaranteed to
                             match, so never compare label ids
                             across backends, only sizes and masks
===========================  ======================================

For a published run, set ``NIFTITOOL_GPU=off`` (or the GUI toggle) if you
want the numbers to be reproducible on any machine regardless of what
hardware it has. ``tests/test_accel.py`` checks every routed kernel against
SciPy and will report the observed deviation on the machine it runs on.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .. import config

# Read through getattr rather than a from-import. A from-import makes this
# module -- and therefore every core function that filters anything --
# refuse to load against a config.py that predates these settings, which is
# what a half-applied update or a stale editor buffer looks like. Falling
# back to the defaults keeps the science working and costs only the
# configurability.
GPU_MODE_DEFAULT = getattr(config, "GPU_MODE_DEFAULT", "auto")
GPU_VRAM_FRACTION = getattr(config, "GPU_VRAM_FRACTION", 0.6)

# Imported directly rather than through ..deps, whose probe bundles NumPy,
# nibabel and SciPy into one try block: a machine with SciPy but without
# nibabel would otherwise see every kernel here refuse to run.
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

#: Modes accepted by :func:`set_mode`.
#:
#: ``auto``
#:     Use the GPU when one is available and the operands fit. The default.
#: ``on``
#:     Same as ``auto``, but log loudly when the GPU is unusable, so a run
#:     that was meant to be accelerated does not quietly fall back.
#: ``off``
#:     Never touch the GPU. Use this for runs whose numbers go into a
#:     paper, so the result does not depend on the host's hardware.
MODES = ("auto", "on", "off")

#: Fraction of *free* VRAM a single call may claim. The rest is left for
#: the display driver, the VTK 3-D viewer (which shares the same device)
#: and CuPy's own allocator fragmentation.
VRAM_FRACTION = GPU_VRAM_FRACTION

#: Multiple of the input size assumed for a kernel's working set: the input
#: copy, the output, and scratch. Separable filters and the EDT need the
#: most; 4x is a deliberately pessimistic single number, because a wrong
#: guess that is too small only costs a fallback, while one that is too
#: large costs an out-of-memory error mid-run.
WORKING_SET_FACTOR = 4.0

#: Arrays smaller than this are not worth a round trip over PCIe: the
#: transfer and kernel-launch overhead exceeds what SciPy takes on the
#: host. 4 MiB is roughly the crossover measured for a 3-D Gaussian.
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
    """Import CuPy and confirm a device actually executes a kernel.

    Importing CuPy succeeds on machines whose driver is missing, too old or
    busy, and the failure then surfaces at the first kernel launch deep
    inside a worker thread. So the probe runs a trivial kernel and only
    reports success if it returns.
    """
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

    Three sources, in decreasing order of accuracy:

    ``"cuda"``
        CuPy's ``mem_info``. Reports both free and total, and is what the
        compute kernels are budgeted against.
    ``"vtk"``
        :class:`vtkGPUInfoList`, which reports *dedicated* video memory for
        any vendor. There is no free-memory figure, so ``free_mb`` is
        ``None`` and callers must budget against the total.
    ``"none"``
        Nothing could be queried -- no GPU, integrated graphics with no
        dedicated memory, or a VTK build without the probe. Callers should
        fall back to a system-RAM budget, which for integrated graphics is
        the correct model anyway: its "VRAM" *is* system RAM.

    A dedicated pool under 256 MiB is reported as ``"none"``: that is what
    an integrated adapter reserves for the framebuffer, not a budget a
    volume can be sized against.
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
    """*kwargs* adjusted for the CuPy backend.

    SciPy's binary morphology defaults to ``brute_force=False``, an
    optimisation that tracks only the voxels changed by the previous
    iteration instead of re-scanning the whole array. CuPy implements only
    the brute-force form and raises ``NotImplementedError`` for anything
    else once ``iterations`` exceeds one, which sent every multi-iteration
    erosion, dilation and closing back to the CPU.

    The flag selects an algorithm, not a result: SciPy's own documentation
    describes it as a performance option, and both settings are verified
    identical for 1 to 5 iterations in ``tests/test_accel.py``. Setting it
    for the GPU call is therefore a backend detail, not a change to what is
    computed. It is *not* set for the SciPy path, where the default is the
    faster of the two.

    An explicit ``brute_force`` from the caller is always respected.
    """
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
    """Record, once per kernel, why the GPU path was abandoned.

    A silent fallback is the worst outcome here: the run is correct but
    slow, and nothing says why. Keeping one note per kernel means an
    unsupported keyword or a driver problem surfaces in the log instead of
    being inferred from a stopwatch.
    """
    if name not in _fallback_notes:
        _fallback_notes[name] = f"{type(ex).__name__}: {ex}"


def fallback_notes() -> dict[str, str]:
    """Kernels that were asked to run on the GPU and could not, and why.

    Empty when everything that was attempted succeeded. Out-of-memory
    fallbacks that a retry recovered from are not recorded; one that
    persisted is.
    """
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
