"""Object removal (inpainting).

* :class:`LamaInpainter` — LaMa, high quality, runs on CPU in seconds (needs
  the ``simple-lama-inpainting`` package + weights, auto-downloaded by it).
* :class:`OpenCVInpainter` — built-in ``cv2.inpaint`` fallback; lower quality
  but always available with zero downloads.

``build_inpainter(config)`` returns the best one that loads.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .segmenter import ModelNotAvailable

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

log = logging.getLogger(__name__)


def _prep_mask(mask: np.ndarray, dilate: int = 3) -> np.ndarray:
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8) * 255
    if dilate and cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        m = cv2.dilate(m, k, iterations=1)
    return m


class BaseInpainter:
    name = "base"
    capability = "fallback"

    def erase(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LamaInpainter(BaseInpainter):
    name = "LaMa"
    capability = "local"

    def __init__(self):
        try:
            from simple_lama_inpainting import SimpleLama  # type: ignore
        except Exception as exc:
            raise ModelNotAvailable(f"simple-lama-inpainting not installed: {exc}") from exc
        try:
            self._lama = SimpleLama()
        except Exception as exc:
            raise ModelNotAvailable(f"Could not initialise LaMa: {exc}") from exc
        log.info("Loaded LaMa inpainter.")

    def erase(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image  # local import; pillow is a UI dep

        img = Image.fromarray(image_rgb[..., :3].astype(np.uint8))
        m = Image.fromarray(_prep_mask(mask))
        result = self._lama(img, m)
        return np.array(result.convert("RGB"))


class OpenCVInpainter(BaseInpainter):
    name = "OpenCV inpaint (built-in)"
    capability = "fallback"

    def __init__(self):
        if cv2 is None:
            raise ModelNotAvailable("OpenCV (cv2) is required.")

    def erase(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(image_rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        m = _prep_mask(mask)
        out = cv2.inpaint(bgr, m, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def build_inpainter(config) -> Tuple[BaseInpainter, str]:
    """Return (inpainter, capability). Tries LaMa, falls back to OpenCV."""
    try:
        inp = LamaInpainter()
        return inp, "local"
    except ModelNotAvailable as exc:
        log.info("LaMa unavailable (%s) — using OpenCV inpaint fallback.", exc)
    except Exception as exc:  # pragma: no cover
        log.warning("Unexpected error loading LaMa (%s) — using OpenCV.", exc)
    return OpenCVInpainter(), "fallback"
