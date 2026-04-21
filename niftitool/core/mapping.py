"""HU → Young's-modulus mappings.

The ultimate goal of the suite is to turn a CT intensity volume into a
spatially varying elasticity field :math:`E(\\mathbf{x})` in MPa that a
finite-cell-method or fracture-mechanics solver can consume.

Four analytical models are supported (more can be added here without
touching the GUI):

============  ================================================================
Model         Expression
============  ================================================================
linear        :math:`E = a\\,HU + b`
power         :math:`E = a\\,HU^{b}` (common for trabecular bone / porous media)
bilinear      piecewise: ``E_void`` below a threshold, ``E_solid`` above
table         piecewise-linear interpolation through user breakpoints
============  ================================================================

All functions return ``float32`` for memory efficiency and clamp to at
least ``E_void`` so downstream solvers never see zero stiffness where
they expect finite stiffness.
"""

from __future__ import annotations

from ..deps import np


def hu_to_E_linear(hu_vol, a, b, void_thresh=None, E_void=0.001):
    """:math:`E = a \\cdot HU + b` [MPa], clamped below *void_thresh*."""
    E = a * hu_vol + b
    if void_thresh is not None:
        E = np.where(hu_vol < void_thresh, E_void, E)
    return np.clip(E, E_void, None).astype(np.float32)


def hu_to_E_power(hu_vol, a, b, hu_min_clamp=1.0, void_thresh=None, E_void=0.001):
    """:math:`E = a \\cdot \\max(HU, HU_{min})^{b}` [MPa]."""
    hu_safe = np.clip(hu_vol, hu_min_clamp, None)
    E = a * np.power(hu_safe, b)
    if void_thresh is not None:
        E = np.where(hu_vol < void_thresh, E_void, E)
    return np.clip(E, E_void, None).astype(np.float32)


def hu_to_E_bilinear(hu_vol, hu_thresh, E_void, E_solid):
    """Piecewise: ``E_void`` below ``hu_thresh``, ``E_solid`` above."""
    return np.where(
        hu_vol < hu_thresh,
        np.float32(E_void),
        np.float32(E_solid),
    ).astype(np.float32)


def hu_to_E_table(hu_vol, hu_breakpoints, E_breakpoints):
    """Piecewise-linear interpolation through a user table."""
    hu_arr = np.array(hu_breakpoints, dtype=np.float64)
    E_arr  = np.array(E_breakpoints,  dtype=np.float64)
    return (
        np.interp(hu_vol.ravel(), hu_arr, E_arr)
        .reshape(hu_vol.shape)
        .astype(np.float32)
    )


def compute_E_map(hu_vol, model: str, params: dict, void_thresh, E_void_default=0.001):
    """Dispatch helper — calls the appropriate ``hu_to_E_*`` above.

    Raises :class:`ValueError` if *model* is unknown.
    """
    if model == "linear":
        return hu_to_E_linear(
            hu_vol, params['a'], params['b'],
            void_thresh, params.get('E_void', E_void_default),
        )
    if model == "power":
        return hu_to_E_power(
            hu_vol, params['a'], params['b'],
            params.get('hu_min_clamp', 1.0),
            void_thresh, params.get('E_void', E_void_default),
        )
    if model == "bilinear":
        return hu_to_E_bilinear(
            hu_vol, params['hu_thresh'],
            params['E_void'], params['E_solid'],
        )
    if model == "table":
        return hu_to_E_table(hu_vol, params['hu'], params['E'])
    raise ValueError(f"Unknown model: {model}")
