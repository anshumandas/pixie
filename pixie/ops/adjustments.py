"""Brightness / contrast adjustment, optionally limited to a masked region.

Pure NumPy + OpenCV — no models, instant, always available.
"""
from __future__ import annotations

import numpy as np

from .compositing import feather_alpha


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    mask: np.ndarray | None = None,
    feather: int = 3,
) -> np.ndarray:
    """Return a brightness/contrast-adjusted copy of ``image`` (RGB uint8).

    ``brightness`` and ``contrast`` are in ``[-1, 1]``. If ``mask`` is given,
    only the masked region is changed and blended back with a feathered edge.
    """
    rgb = image[..., :3].astype(np.float32)
    alpha = 1.0 + _clamp(contrast)          # contrast gain in [0, 2]
    beta = _clamp(brightness) * 127.0        # brightness shift in [-127, 127]

    adjusted = (rgb - 127.0) * alpha + 127.0 + beta
    adjusted = np.clip(adjusted, 0, 255)

    if mask is None:
        return adjusted.astype(np.uint8)

    a = feather_alpha(mask, feather)[..., None]
    out = rgb * (1.0 - a) + adjusted * a
    return np.clip(out, 0, 255).astype(np.uint8)
