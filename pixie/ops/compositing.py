"""Mask feathering, background removal, and masked compositing.

Pure NumPy + OpenCV. Masks are accepted as any array where ``> 0`` means
"inside the selection"; they are normalised internally.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is a hard runtime dep, soft here
    cv2 = None


def _as_binary(mask: np.ndarray) -> np.ndarray:
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.float32)


def feather_alpha(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Return a float alpha map in ``[0, 1]`` with softened edges."""
    m = _as_binary(mask)
    if radius and radius > 0 and cv2 is not None:
        k = int(radius) * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    elif radius and radius > 0:  # pragma: no cover - cv2 normally present
        # crude separable box blur fallback if cv2 is missing
        from scipy.ndimage import gaussian_filter  # type: ignore

        m = gaussian_filter(m, sigma=radius)
    return np.clip(m, 0.0, 1.0)


def remove_background(image: np.ndarray, mask: np.ndarray, feather: int = 3) -> np.ndarray:
    """Keep the masked subject, make everything else transparent.

    Returns an RGBA uint8 array.
    """
    rgb = image[..., :3]
    alpha = (feather_alpha(mask, feather) * 255.0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def composite_into(
    base: np.ndarray, edited: np.ndarray, mask: np.ndarray, feather: int = 3
) -> np.ndarray:
    """Blend ``edited`` into ``base`` only within ``mask`` (feathered)."""
    b = base[..., :3].astype(np.float32)
    e = edited[..., :3].astype(np.float32)
    a = feather_alpha(mask, feather)[..., None]
    out = b * (1.0 - a) + e * a
    return np.clip(out, 0, 255).astype(np.uint8)
