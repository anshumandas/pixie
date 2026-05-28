"""Stylisation filters.

The ink/shadow/pencil/oil styles are pure OpenCV — instant, no weights, always
available. Neural / Stable-Diffusion stylisation is routed through a compute
backend (local GPU or cloud) and is not implemented here.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for stylisation.")


def stylize_ink(image: np.ndarray) -> np.ndarray:
    """Crisp black-line ink drawing via an XDoG (extended difference of Gaussians)."""
    _require_cv2()
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sigma, k, tau, phi, eps = 0.6, 1.6, 0.985, 10.0, -0.02
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - tau * g2
    ink = np.where(dog >= eps, 1.0, 1.0 + np.tanh(phi * (dog - eps)))
    ink = np.clip(ink, 0.0, 1.0)
    ink = (ink * 255.0).astype(np.uint8)
    return cv2.cvtColor(ink, cv2.COLOR_GRAY2RGB)


def stylize_shadow(image: np.ndarray) -> np.ndarray:
    """High-contrast two-tone silhouette / shadow-art look."""
    _require_cv2()
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)


def stylize_pencil(image: np.ndarray) -> np.ndarray:
    """Soft grey pencil sketch (OpenCV's non-photorealistic rendering)."""
    _require_cv2()
    bgr = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)
    gray, _ = cv2.pencilSketch(bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def stylize_oil(image: np.ndarray) -> np.ndarray:
    """Painterly / stylised look via edge-aware smoothing."""
    _require_cv2()
    bgr = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)
    out = cv2.stylization(bgr, sigma_s=60, sigma_r=0.45)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# name -> function, for the UI to enumerate.
STYLES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Ink": stylize_ink,
    "Shadow": stylize_shadow,
    "Pencil": stylize_pencil,
    "Oil / paint": stylize_oil,
}


def apply_style(image: np.ndarray, style: str) -> np.ndarray:
    fn = STYLES.get(style)
    if fn is None:
        raise KeyError(f"Unknown style '{style}'. Options: {list(STYLES)}")
    return fn(image)
