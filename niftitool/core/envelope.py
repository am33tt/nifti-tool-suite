"""Outer boundary of a cylindrical specimen, as a radial envelope.

A specimen scanned inside a sleeve, a mould or a wrapping shares its
attenuation with that shell, and the two touch somewhere over the height of
almost any scan. One contact is enough: the largest connected component
then holds specimen and shell together, and filling the specimen's cavities
fills the air gap between them as well, so the shell survives the
background removal untouched.

Nor does a single cut radius separate them. On an additively manufactured
cylinder the printed layers leave ribs, so the surface radius varies by
several percent around the circumference, and those ribs reach past the
inner face of the shell. Any radius that clears the shell everywhere cuts
material somewhere.

What does separate them is the air gap itself, found direction by
direction. For every slice and every angle around the axis, the specimen's
surface is the first sustained run of void encountered on the way out from
the axis; the shell lies beyond it. The result is a boundary R(theta, z)
that follows the real surface, ribs included, and owes nothing to how badly
the shell is attached.

The method assumes the specimen is star-shaped about its axis -- every ray
from the axis leaves it exactly once -- which holds for a cylinder, a prism
or a core, and fails for a specimen with an undercut or a surface-breaking
cavity wider than the search band.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..deps import np
from . import accel

#: Rays cast per slice. The surface varies smoothly with angle, so this sets
#: the angular resolution of the boundary rather than its accuracy.
DEFAULT_ANGLES = 720

#: Search band for the surface, as a fraction of the slice's nominal radius.
DEFAULT_BAND = (0.70, 1.35)

#: A void run must be at least this many voxels deep to count as the outside
#: rather than a pore near the surface.
DEFAULT_MIN_GAP = 3

#: Half-width, in slices, of the median filter along the axis. The surface of
#: a specimen printed along its axis varies with angle and barely with z, so
#: a generous window rejects outliers without touching the ribs.
DEFAULT_Z_WINDOW = 25

#: Half-width, in rays, of the median filter around the circumference.
DEFAULT_ANGLE_WINDOW = 1

#: Polynomial degree of the specimen axis along z. A rigid specimen scanned
#: once has a straight axis, so a line is the honest model; raise it only for
#: a specimen that is visibly bent.
DEFAULT_AXIS_DEGREE = 1

#: Voxels added to the fitted radius, so the surface itself is never clipped.
DEFAULT_MARGIN = 2.0

#: A ray is rejected when it departs from the fitted surface by more than
#: this many voxels, or by more than DEFAULT_OUTLIER_SIGMA robust standard
#: deviations of the fit, whichever is larger. It has to be well above the
#: roughness of a real surface and well below the width of the gap.
DEFAULT_OUTLIER_PX = 6.0
DEFAULT_OUTLIER_SIGMA = 5.0

#: A slice with fewer solid voxels than this is treated as empty.
MIN_SLICE_VOXELS = 1000


@dataclass
class RadialEnvelope:
    """The fitted boundary of a specimen, one radius per angle and slice.

    Attributes
    ----------
    centres
        ``(nz, 2)`` axis position in voxel coordinates, per slice.
    radii
        ``(nz, n_angles)`` surface radius in voxels. Angles run from
        ``-pi`` to ``pi``, matching :func:`numpy.arctan2`.
    valid
        ``(nz,)`` flags the slices a boundary could be measured on; the
        others were interpolated from them.
    shape
        Shape of the volume the envelope belongs to.
    """

    centres: np.ndarray
    radii: np.ndarray
    valid: np.ndarray
    shape: tuple

    def __post_init__(self) -> None:
        self._grid_key = None
        self._grid = None
        #: Rays replaced by the fitted model because they ran past the
        #: specimen. Set by :func:`fit_envelope`.
        self.replaced_rays = 0

    @property
    def n_angles(self) -> int:
        return int(self.radii.shape[1])

    def diameters(self):
        """Mean diameter of each slice, in voxels."""
        return 2.0 * np.nanmean(self.radii, axis=1)

    def summary(self) -> dict:
        """Compact description of the fit, for the log and the JSON record."""
        radii = self.radii[self.valid] if self.valid.any() else self.radii
        centres = self.centres[self.valid] if self.valid.any() else self.centres
        drift = (centres.max(axis=0) - centres.min(axis=0)) if centres.size else [0, 0]
        return {
            "n_angles": self.n_angles,
            "slices_measured": int(self.valid.sum()),
            "slices_interpolated": int((~self.valid).sum()),
            "rays_replaced": int(self.replaced_rays),
            "rays_total": int(self.radii.size),
            "radius_px": {
                "p5": float(np.nanpercentile(radii, 5)),
                "median": float(np.nanmedian(radii)),
                "p95": float(np.nanpercentile(radii, 95)),
            },
            "slice_median_radius_px": {
                "min": float(np.nanmin(np.nanmedian(radii, axis=1))),
                "max": float(np.nanmax(np.nanmedian(radii, axis=1))),
            },
            "axis_drift_px": [float(drift[0]), float(drift[1])],
        }

    def mask_slice(self, z: int):
        """Boolean slice: True inside the envelope.

        The polar grid is rebuilt only when the axis moves to a different
        whole voxel, which happens a handful of times over a scan; the
        rounding is far below the fitted margin.
        """
        nx, ny = self.shape[0], self.shape[1]
        cx, cy = self.centres[z]
        key = (int(round(float(cx))), int(round(float(cy))))
        if key != self._grid_key:
            xs = np.arange(nx, dtype=np.float32)[:, None] - key[0]
            ys = np.arange(ny, dtype=np.float32)[None, :] - key[1]
            radius = np.hypot(xs, ys)
            angle = np.arctan2(ys, xs)
            bins = ((angle + np.pi) * (self.n_angles / (2.0 * np.pi)))
            bins = bins.astype(np.int32) % self.n_angles
            self._grid = (radius, bins)
            self._grid_key = key
        radius, bins = self._grid
        return radius <= self.radii[z][bins]


def _slice_centre_and_scale(mask):
    """Axis position and nominal radius of one slice.

    The centroid is taken over shrinking central regions rather than over the
    whole slice. A shell, or a contact bridging the gap to one, sits at large
    radius and would otherwise pull the axis towards it -- by only a voxel or
    two, but consistently over every slice where it appears, which is enough
    to bias every radius measured from that axis. Restricting the centroid to
    the inner part leaves it where it belongs; for a disc the inner centroid
    and the full one coincide.

    The nominal radius comes from the median distance of the solid voxels,
    which for a disc of radius R is R/sqrt(2).
    """
    xs, ys = np.nonzero(mask)
    if xs.size < MIN_SLICE_VOXELS:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    distance = np.hypot(xs - cx, ys - cy)
    nominal = float(np.median(distance)) * np.sqrt(2.0)
    for fraction in (0.8, 0.6, 0.6):
        inner = distance < fraction * nominal
        if inner.sum() < MIN_SLICE_VOXELS:
            break
        cx, cy = float(xs[inner].mean()), float(ys[inner].mean())
        distance = np.hypot(xs - cx, ys - cy)
        nominal = float(np.median(distance)) * np.sqrt(2.0)
    return cx, cy, nominal


def _surface_radii(mask, centre, nominal, *, cos_a, sin_a, band, min_gap,
                   step=0.5):
    """Surface radius along each ray, or NaN where no void was found.

    Samples the mask outwards over ``band * nominal`` and returns the start
    of the first run of at least *min_gap* void samples.
    """
    nx, ny = mask.shape
    cx, cy = centre
    radii = np.arange(band[0] * nominal, band[1] * nominal, step,
                      dtype=np.float32)
    if radii.size <= min_gap:
        return np.full(cos_a.size, np.nan, dtype=np.float32)

    xs = np.rint(cx + cos_a[:, None] * radii[None, :]).astype(np.int32)
    ys = np.rint(cy + sin_a[:, None] * radii[None, :]).astype(np.int32)
    np.clip(xs, 0, nx - 1, out=xs)
    np.clip(ys, 0, ny - 1, out=ys)
    void = ~mask[xs, ys]

    # Void samples in every window of min_gap consecutive positions.
    counts = np.cumsum(void, axis=1, dtype=np.int32)
    window = counts[:, min_gap - 1:].copy()
    window[:, 1:] -= counts[:, :-min_gap]
    full = window == min_gap

    found = full.any(axis=1)
    out = np.full(cos_a.size, np.nan, dtype=np.float32)
    out[found] = radii[np.argmax(full, axis=1)[found]]
    return out


def _regularise_track(values, *, window, degree, tol_px, fallback):
    """Clean a per-slice quantity that should vary smoothly along the axis.

    Used for the axis position and the slice size. Both are measured from a
    mask that may still contain a shell, and a stretch of slices where the
    shell touches biases them all in the same direction -- a median filter
    would follow that bias, since inside its window the contaminated slices
    are in the majority. So the trend is fitted first, as a low-order
    polynomial with the outliers iteratively excluded; readings that
    disagree with it are replaced by it, and only then is the result
    filtered. Genuine slow wobble of the axis survives; a biased stretch
    does not.
    """
    values = np.asarray(values, dtype=np.float64).copy()
    known = np.isfinite(values)
    if not known.any():
        return np.full(values.shape, float(fallback))

    positions = np.arange(values.size, dtype=np.float64)
    good = known.copy()
    trend = np.full(values.shape, float(np.nanmedian(values)))
    for _ in range(3):
        if good.sum() < degree + 2:
            break
        trend = np.polyval(np.polyfit(positions[good], values[good], degree),
                           positions)
        residual = np.abs(values - trend)
        spread = np.nanmedian(residual[known])
        spread = 0.0 if not np.isfinite(spread) else 1.4826 * float(spread)
        good = known & (residual <= max(float(tol_px), 4.0 * spread))

    values = np.where(good, values, trend)
    if window > 0 and values.size > 1:
        width = min(2 * int(window) + 1, values.size)
        values = accel.median_filter(values, size=width, mode='nearest')
    return values


def _separable_model(radii):
    """Robust model of the surface: a size per slice times a shape per ray.

    ``R(theta, z) = scale(z) * shape(theta)``. The scale is the median over
    the rays of one slice, so it survives a minority of rays being wrong;
    the shape is the median over the slices of the normalised radius, so it
    survives a minority of slices being wrong. Their product is a surface
    that keeps the ribs -- they are what ``shape`` is -- while following a
    specimen that tapers or changes size along its axis.
    """
    with np.errstate(invalid='ignore'):
        scale = np.nanmedian(radii, axis=1)
    known = np.isfinite(scale)
    if not known.any():
        return None, None
    if not known.all():
        scale[~known] = np.interp(np.flatnonzero(~known),
                                  np.flatnonzero(known), scale[known])
    with np.errstate(invalid='ignore'):
        shape = np.nanmedian(radii / scale[:, None], axis=0)
    shape[~np.isfinite(shape)] = 1.0
    return scale, shape


def _inliers(radii, model, *, outlier_px, outlier_sigma):
    """Rays that agree with the fitted surface.

    A ray crossing a contact between the specimen and whatever surrounds it
    finds no void until beyond the shell, and reports a radius that is too
    large. Those readings are not noise and a median filter will not remove
    them: where the contact persists over neighbouring slices -- the usual
    case, since a sleeve rests against a specimen over a stretch of its
    length -- they outvote the correct ones inside the filter window. They
    have to be identified against the model and excluded.
    """
    residual = np.abs(radii - model)
    spread = np.nanmedian(residual)
    spread = 0.0 if not np.isfinite(spread) else 1.4826 * float(spread)
    tolerance = max(float(outlier_px), float(outlier_sigma) * spread)
    return np.isfinite(radii) & (residual <= tolerance)


def _fit_surface(radii, *, outlier_px, outlier_sigma, rounds=2):
    """``(model, inliers, scale, shape)`` for the measured radii."""
    working = radii.astype(np.float64, copy=True)
    scale = shape = model = inliers = None
    for _ in range(rounds):
        scale, shape = _separable_model(working)
        if scale is None:
            return None, None, None, None
        model = scale[:, None] * shape[None, :]
        inliers = _inliers(radii, model, outlier_px=outlier_px,
                           outlier_sigma=outlier_sigma)
        working = np.where(inliers, radii, np.nan)
    return model, inliers, scale, shape


def _centre_offsets(radii, inliers, shape, cos_a, sin_a):
    """How far the assumed axis sits from the centre of the measured surface.

    Measured from a point offset by ``d`` from the true axis, the radius of
    a round specimen reads ``R(theta) + dx cos(theta) + dy sin(theta)``: the
    offset is the first angular harmonic, and nothing else in the profile
    looks like it. Fitting it alongside the specimen's own shape -- so ribs
    cannot leak into the estimate -- and adding it back is the standard way
    to centre a roundness measurement, and it anchors the axis to the
    surface rather than to the mask's centre of area, which a shell pulls
    sideways.
    """
    nz = radii.shape[0]
    offsets = np.zeros((nz, 2), dtype=np.float64)
    basis = np.stack([shape, cos_a, sin_a], axis=1)
    for z in range(nz):
        use = inliers[z]
        if use.sum() < 3 * basis.shape[1] or use.mean() < 0.25:
            continue
        solution, *_ = np.linalg.lstsq(basis[use], radii[z][use], rcond=None)
        offsets[z] = solution[1:]
    return offsets


def _fill_and_smooth(radii, model, inliers, *, z_window, angle_window):
    """Replace the rejected rays by the model, then filter what is left."""
    radii = np.where(inliers, radii, model).astype(np.float64)
    if angle_window > 0:
        # The circumference wraps, so pad with the opposite end.
        width = 2 * angle_window + 1
        padded = np.concatenate(
            [radii[:, -angle_window:], radii, radii[:, :angle_window]], axis=1)
        radii = accel.median_filter(padded, size=(1, width),
                                      mode='nearest')[:, angle_window:-angle_window]
    if z_window > 0 and radii.shape[0] > 1:
        width = min(2 * z_window + 1, radii.shape[0])
        radii = accel.median_filter(radii, size=(width, 1), mode='nearest')
    return radii.astype(np.float32)


def fit_envelope(
    solid,
    *,
    n_angles: int = DEFAULT_ANGLES,
    band=DEFAULT_BAND,
    min_gap: int = DEFAULT_MIN_GAP,
    z_window: int = DEFAULT_Z_WINDOW,
    angle_window: int = DEFAULT_ANGLE_WINDOW,
    axis_degree: int = DEFAULT_AXIS_DEGREE,
    margin: float = DEFAULT_MARGIN,
    outlier_px: float = DEFAULT_OUTLIER_PX,
    outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
    progress=None,
    cancel=None,
) -> RadialEnvelope:
    """Fit the outer boundary of the specimen in *solid*.

    *solid* is the thresholded volume as a
    :class:`~niftitool.core.label_stream.PackedMask`; the fit reads it slice
    by slice and allocates only the rays, so it costs one pass and a few
    megabytes whatever the volume size.

    Raises :class:`ValueError` when no slice yields a boundary.
    """
    from ..utils import OperationCancelled

    nx, ny, nz = solid.shape
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False,
                         dtype=np.float32)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    # First pass: the axis and the size of every slice. Both are measured
    # from a mask that still holds the shell, so both are then regularised
    # along z before they are used -- a specimen axis is a smooth curve, and
    # a slice where the shell touches would otherwise pull it sideways and
    # bias every radius measured from it.
    raw_centres = np.full((nz, 2), np.nan, dtype=np.float64)
    nominals = np.full(nz, np.nan, dtype=np.float64)
    for z in range(nz):
        if cancel is not None and cancel():
            raise OperationCancelled("cancelled")
        found = _slice_centre_and_scale(solid.get(z))
        if found is not None:
            raw_centres[z] = found[:2]
            nominals[z] = found[2]
        if progress is not None and (z % 64 == 0 or z == nz - 1):
            progress("locating the axis", (z + 1) / nz)

    if not np.isfinite(nominals).any():
        raise ValueError("No material found: every slice is empty.")
    centres = np.stack(
        [_regularise_track(raw_centres[:, 0], window=z_window,
                           degree=axis_degree, tol_px=2.0, fallback=nx / 2.0),
         _regularise_track(raw_centres[:, 1], window=z_window,
                           degree=axis_degree, tol_px=2.0, fallback=ny / 2.0)],
        axis=1,
    )
    nominals = _regularise_track(
        nominals, window=z_window, degree=axis_degree, tol_px=3.0,
        fallback=float(np.nanmedian(nominals)),
    )

    # Then the surface itself, measured along every ray. It is measured
    # twice: the first pass tells us where the axis really is, and the
    # second is taken about that axis.
    def measure(stage):
        radii = np.full((nz, n_angles), np.nan, dtype=np.float32)
        for z in range(nz):
            if cancel is not None and cancel():
                raise OperationCancelled("cancelled")
            plane = solid.get(z)
            if plane.sum() < MIN_SLICE_VOXELS:
                continue
            radii[z] = _surface_radii(
                plane, tuple(centres[z]), nominals[z],
                cos_a=cos_a, sin_a=sin_a, band=band, min_gap=min_gap,
            )
            if progress is not None and (z % 32 == 0 or z == nz - 1):
                progress(stage, (z + 1) / nz)
        return radii

    radii = measure("locating the surface")
    model, inliers, _, shape = _fit_surface(
        radii, outlier_px=outlier_px, outlier_sigma=outlier_sigma)
    if model is None:
        raise ValueError(
            "No specimen surface found: every ray left the volume without "
            "crossing a void. The threshold may be below the background, or "
            "the specimen may fill the field of view."
        )

    offsets = _centre_offsets(radii, inliers, shape, cos_a, sin_a)
    centres = np.stack(
        [_regularise_track(centres[:, 0] + offsets[:, 0], window=z_window,
                           degree=axis_degree, tol_px=1.0, fallback=nx / 2.0),
         _regularise_track(centres[:, 1] + offsets[:, 1], window=z_window,
                           degree=axis_degree, tol_px=1.0, fallback=ny / 2.0)],
        axis=1,
    )

    radii = measure("fitting envelope")
    model, inliers, _, _ = _fit_surface(
        radii, outlier_px=outlier_px, outlier_sigma=outlier_sigma)
    if model is None:
        raise ValueError("The specimen surface could not be fitted.")

    valid = inliers.mean(axis=1) > 0.5
    smoothed = _fill_and_smooth(radii, model, inliers, z_window=z_window,
                                angle_window=angle_window) + float(margin)
    envelope = RadialEnvelope(centres, smoothed, valid, solid.shape)
    envelope.replaced_rays = int((~inliers).sum())
    return envelope


def apply_envelope(solid, envelope, *, progress=None, cancel=None) -> int:
    """Drop everything outside *envelope* from *solid*, in place.

    Returns the number of solid voxels removed -- the material the boundary
    rejected, which is the number to check before trusting a run: it should
    account for the shell and nothing else.
    """
    from ..utils import OperationCancelled

    nz = solid.shape[2]
    removed = 0
    for z in range(nz):
        if cancel is not None and cancel():
            raise OperationCancelled("cancelled")
        plane = solid.get(z)
        if not plane.any():
            continue
        inside = envelope.mask_slice(z)
        cut = int(plane.sum()) - int((plane & inside).sum())
        if cut:
            removed += cut
            solid.set(z, plane & inside)
        if progress is not None and (z % 32 == 0 or z == nz - 1):
            progress("applying envelope", (z + 1) / nz)
    return removed
