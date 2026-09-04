"""Human-readable metadata report + axis-label helpers."""

from __future__ import annotations

from ..deps import np, nio


def _flat_sample(gray, max_voxels: int = 2_000_000):
    """Flat intensity sample that never materialises a lazy volume."""
    if hasattr(gray, "subsample_flat"):
        return gray.subsample_flat(max_voxels)
    flat = gray.ravel()
    if flat.size > max_voxels:
        flat = flat[:: flat.size // max_voxels]
    return flat


def get_axis_labels(affine) -> dict:
    """Return compass labels for every slice-plane axis.

    Each entry is a ``(positive_code, negative_code)`` tuple such as
    ``('R', 'L')``, used by the triplanar viewer to draw orientation
    compasses.
    """
    try:
        codes = nio.aff2axcodes(affine)
        flips = {'R': 'L', 'L': 'R', 'A': 'P', 'P': 'A', 'S': 'I', 'I': 'S'}
        return {
            'x': (codes[0], flips.get(codes[0], '?')),
            'y': (codes[1], flips.get(codes[1], '?')),
            'z': (codes[2], flips.get(codes[2], '?')),
        }
    except Exception:
        return {'x': ('X', '-'), 'y': ('Y', '-'), 'z': ('Z', '-')}


def collect_metadata(img, gray=None) -> list:
    """Structured metadata bundle used by the Metadata viewer tab.

    Returns a list of sections, each ``{'title': str, 'rows': [(k, v), ...]}``
    or ``{'title': str, 'matrix': 2-D list}`` for the affine.
    :func:`read_metadata` renders the same data as plain text.
    """
    from nibabel.orientations import aff2axcodes
    affine  = img.affine
    header  = img.header
    orient  = "".join(aff2axcodes(affine))
    spacing = header.get_zooms()
    m       = affine[:3, :3]
    scale   = np.sqrt(np.sum(m ** 2, axis=0))

    vox_mm3 = float(np.prod(scale[:3]))
    total_vox = int(np.prod(img.shape[:3]))
    phys_vol_mm3 = vox_mm3 * total_vox
    phys_vol_cm3 = phys_vol_mm3 / 1000.0
    phys_size_mm = tuple(
        round(float(img.shape[i]) * float(scale[i]), 2)
        for i in range(min(3, len(img.shape)))
    )
    rot   = m / scale
    pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    roll  = np.degrees(np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2)))
    yaw   = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))

    sections = []

    sections.append({
        'title': 'Volume Geometry',
        'rows': [
            ("Shape",          str(img.shape)),
            ("Dtype",          str(img.get_data_dtype())),
            ("Orientation",    orient),
            ("Voxel spacing",  f"{tuple(round(float(s), 4) for s in spacing)} mm"),
            ("Scale factors",  str(tuple(round(float(s), 4) for s in scale))),
            ("Physical size",  f"{phys_size_mm} mm"),
            ("Voxel volume",   f"{vox_mm3:.4f} mm³"),
            ("Total volume",   f"{phys_vol_mm3:.2f} mm³   ({phys_vol_cm3:.4f} cm³)"),
            ("Origin (world)", str(tuple(round(float(v), 3) for v in affine[:3, 3]))),
            ("Euler angles",   f"Pitch={pitch:.2f}°   Roll={roll:.2f}°   Yaw={yaw:.2f}°"),
        ],
    })

    if gray is not None:
        flat = _flat_sample(gray)
        rows = [
            ("Min",      f"{float(flat.min()):.2f}"),
            ("Max",      f"{float(flat.max()):.2f}"),
            ("Mean",     f"{float(flat.mean()):.4f}"),
            ("Std",      f"{float(flat.std()):.4f}"),
            ("p1 / p99", f"{float(np.percentile(flat, 1)):.2f}  /  "
                         f"{float(np.percentile(flat, 99)):.2f}"),
        ]
        try:
            cx, cy, cz = [s // 2 for s in gray.shape[:3]]
            r = max(3, min(20, min(gray.shape[:3]) // 8))
            roi = gray[cx - r:cx + r, cy - r:cy + r, cz - r:cz + r]
            snr = float(roi.mean() / (roi.std() + 1e-9))
            rows.append(("SNR (central ROI)",
                         f"{snr:.1f}   (higher = less noise)"))
        except Exception:
            pass
        sections.append({'title': 'Intensity Statistics', 'rows': rows})

    return sections


def read_metadata(img, gray=None) -> str:
    """Multi-section metadata report suitable for the log pane."""
    from nibabel.orientations import aff2axcodes
    affine  = img.affine
    header  = img.header
    orient  = "".join(aff2axcodes(affine))
    spacing = header.get_zooms()
    m       = affine[:3, :3]
    scale   = np.sqrt(np.sum(m ** 2, axis=0))

    vox_mm3 = float(np.prod(scale[:3]))
    total_vox = int(np.prod(img.shape[:3]))
    phys_vol_mm3 = vox_mm3 * total_vox
    phys_vol_cm3 = phys_vol_mm3 / 1000.0

    phys_size_mm = tuple(
        round(float(img.shape[i]) * float(scale[i]), 2)
        for i in range(min(3, len(img.shape)))
    )

    rot   = m / scale
    pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    roll  = np.degrees(np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2)))
    yaw   = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))

    lines = [
        "══ VOLUME GEOMETRY ══════════════════════════════════",
        f"  Shape            : {img.shape}",
        f"  Dtype            : {img.get_data_dtype()}",
        f"  Orientation      : {orient}",
        f"  Voxel spacing    : {tuple(round(float(s), 4) for s in spacing)} mm",
        f"  Scale factors    : {tuple(round(float(s), 4) for s in scale)}",
        f"  Physical size    : {phys_size_mm} mm",
        f"  Voxel volume     : {vox_mm3:.4f} mm³",
        f"  Total volume     : {phys_vol_mm3:.2f} mm³  ({phys_vol_cm3:.4f} cm³)",
        f"  Origin (world)   : {tuple(round(float(v), 3) for v in affine[:3, 3])}",
        f"  Euler angles     : Pitch={pitch:.2f}°  Roll={roll:.2f}°  Yaw={yaw:.2f}°",
        "",
        "══ INTENSITY STATISTICS ════════════════════════════",
    ]

    if gray is not None:
        flat = _flat_sample(gray)
        lines += [
            f"  Min              : {float(flat.min()):.2f}",
            f"  Max              : {float(flat.max()):.2f}",
            f"  Mean             : {float(flat.mean()):.4f}",
            f"  Std              : {float(flat.std()):.4f}",
            f"  p1 / p99         : {float(np.percentile(flat, 1)):.2f} / {float(np.percentile(flat, 99)):.2f}",
        ]
        # Central-ROI SNR estimate, indicative only.
        try:
            cx, cy, cz = [s // 2 for s in gray.shape[:3]]
            r = max(3, min(20, min(gray.shape[:3]) // 8))
            roi = gray[cx - r:cx + r, cy - r:cy + r, cz - r:cz + r]
            snr = float(roi.mean() / (roi.std() + 1e-9))
            lines.append(f"  SNR (central ROI): {snr:.1f}  (higher = less noise)")
        except Exception:
            pass

    return "\n".join(lines)
