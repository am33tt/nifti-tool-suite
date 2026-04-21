"""Hounsfield-Unit calibration.

Industrial CT scanners report arbitrary attenuation intensities.  To
compare scans, drive material models, or export to simulation, we need to
map those intensities onto the Hounsfield scale

.. math::

    HU = m \\cdot I + c

where :math:`I` is the raw voxel intensity.  A two-point linear fit is
sufficient as long as one point is *air* (HU = −1000 by definition) and
the second is a known-density reference (water = 0, a calibration puck,
etc.).

Given two known pairs :math:`(I_{air}, HU_{air})` and
:math:`(I_{ref}, HU_{ref})`:

.. math::

    m = \\frac{HU_{ref} - HU_{air}}{I_{ref} - I_{air}}
    \\qquad
    c = HU_{air} - m \\cdot I_{air}
"""

from __future__ import annotations

from ..deps import np


def calibrate_to_hu(
    gray,
    hu_air_intensity: float,
    hu_ref_intensity: float,
    hu_air_value: float = -1000.0,
    hu_ref_value: float = 0.0,
):
    """Two-point linear HU calibration.

    Parameters
    ----------
    gray : ndarray
        Raw voxel intensity volume (float32).
    hu_air_intensity, hu_ref_intensity : float
        Raw intensities corresponding to the air and reference samples.
    hu_air_value, hu_ref_value : float, optional
        HU values of the two calibration points. Defaults are the IEC
        convention (air = −1000, water = 0).

    Returns
    -------
    hu_vol : ndarray (float32)
        Calibrated volume, same shape as *gray*.
    m, c : float
        Linear coefficients — useful for generating the matching ``ct.hpp``
        snippet and for displaying the dual-axis histogram.

    Raises
    ------
    ValueError
        If the two calibration intensities are equal (degenerate fit).
    """
    if hu_ref_intensity == hu_air_intensity:
        raise ValueError("Air and reference intensities must differ.")
    m = (hu_ref_value - hu_air_value) / (hu_ref_intensity - hu_air_intensity)
    c = hu_air_value - m * hu_air_intensity
    return gray.astype(np.float32) * m + c, m, c
