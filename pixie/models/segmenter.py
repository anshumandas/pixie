"""Click-to-select segmentation.

Two implementations behind one interface:

* :class:`SamSegmenter`   — Segment Anything / MobileSAM (best quality, needs
  weights + torch). Computes the heavy image embedding once per image, then
  each click is near-instant.
* :class:`GrabCutSegmenter` — a built-in OpenCV GrabCut fallback that needs no
  weights, so click-to-select works on a fresh install / CPU-only machine.

``build_segmenter(config)`` returns the best one that actually loads, plus a
capability string ('local' for SAM, 'fallback' for GrabCut).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

log = logging.getLogger(__name__)


class ModelNotAvailable(RuntimeError):
    """Raised when a neural model can't be loaded (missing lib or weights)."""


class BaseSegmenter:
    name = "base"
    capability = "fallback"

    def set_image(self, image_rgb: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, points, labels) -> np.ndarray:
        """Return a uint8 mask (1 = selected) from point prompts.

        ``points`` is a list of (x, y); ``labels`` is a parallel list where
        ``1`` means "include this point" (foreground / add) and ``0`` means
        "exclude this point" (background / remove).
        """
        raise NotImplementedError


_SAM_CHECKPOINTS = {
    "mobile_sam": ("mobile_sam.pt", "vit_t"),
    "sam2_small": ("sam2_hiera_small.pt", "vit_b"),
    "sam2_large": ("sam2_hiera_large.pt", "vit_h"),
    "sam_vit_b": ("sam_vit_b_01ec64.pth", "vit_b"),
    "sam_vit_h": ("sam_vit_h_4b8939.pth", "vit_h"),
}


class SamSegmenter(BaseSegmenter):
    name = "SAM"
    capability = "local"

    def __init__(self, variant: str, weights_dir: Path, device: str = "cpu"):
        self.variant = variant
        self.device = device
        ckpt_name, model_type = _SAM_CHECKPOINTS.get(variant, _SAM_CHECKPOINTS["mobile_sam"])
        ckpt = Path(weights_dir) / ckpt_name
        if not ckpt.exists():
            raise ModelNotAvailable(
                f"SAM checkpoint not found at {ckpt}. Download it once and the app "
                f"will use it automatically (see README -> Models)."
            )
        try:
            try:
                from mobile_sam import SamPredictor, sam_model_registry  # type: ignore
            except Exception:
                from segment_anything import SamPredictor, sam_model_registry  # type: ignore
            sam = sam_model_registry[model_type](checkpoint=str(ckpt))
            sam.to(self.device)
            self._predictor = SamPredictor(sam)
        except ModelNotAvailable:
            raise
        except Exception as exc:
            raise ModelNotAvailable(f"Could not initialise SAM: {exc}") from exc
        log.info("Loaded SAM (%s) on %s", variant, device)

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._predictor.set_image(image_rgb)

    def predict(self, points, labels) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        lbls = np.array(labels, dtype=np.int32)
        multimask = len(points) == 1
        masks, scores, _ = self._predictor.predict(
            point_coords=pts, point_labels=lbls, multimask_output=multimask
        )
        best = masks[int(np.argmax(scores))]
        return best.astype(np.uint8)


class GrabCutSegmenter(BaseSegmenter):
    """No-weights fallback: seed GrabCut from the click location(s)."""

    name = "GrabCut (built-in)"
    capability = "fallback"

    def __init__(self):
        if cv2 is None:
            raise ModelNotAvailable("OpenCV (cv2) is required.")
        self._img = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._img = np.ascontiguousarray(image_rgb[..., :3])

    def predict(self, points, labels) -> np.ndarray:
        if self._img is None:
            raise RuntimeError("Call set_image() before predicting.")
        h, w = self._img.shape[:2]
        r = max(8, int(0.06 * min(h, w)))
        b = max(2, int(0.02 * min(h, w)))

        def _clip(px, py):
            return int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))

        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        for (px, py), lab in zip(points, labels):
            if lab == 1:
                cx, cy = _clip(px, py)
                cv2.circle(gc, (cx, cy), 2 * r, cv2.GC_PR_FGD, -1)
        for (px, py), lab in zip(points, labels):
            cx, cy = _clip(px, py)
            cv2.circle(gc, (cx, cy), r, cv2.GC_FGD if lab == 1 else cv2.GC_BGD, -1)
        gc[:b, :] = cv2.GC_BGD
        gc[-b:, :] = cv2.GC_BGD
        gc[:, :b] = cv2.GC_BGD
        gc[:, -b:] = cv2.GC_BGD

        first_fg = next(((px, py) for (px, py), lab in zip(points, labels) if lab == 1), None)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(self._img, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        except Exception as exc:  # pragma: no cover
            log.warning("GrabCut failed (%s); falling back to a disc selection.", exc)
            disc = np.zeros((h, w), np.uint8)
            if first_fg is not None:
                cv2.circle(disc, _clip(*first_fg), r, 1, -1)
            return disc
        mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        if mask.sum() == 0 and first_fg is not None:
            cv2.circle(mask, _clip(*first_fg), r, 1, -1)
        return mask


def build_segmenter(config) -> Tuple[BaseSegmenter, str]:
    """Return (segmenter, capability). Tries SAM, falls back to GrabCut."""
    try:
        seg = SamSegmenter(config.sam_variant, config.weights_dir, config.device)
        return seg, "local"
    except ModelNotAvailable as exc:
        log.info("SAM unavailable (%s) -- using GrabCut fallback.", exc)
    except Exception as exc:  # pragma: no cover
        log.warning("Unexpected error loading SAM (%s) -- using GrabCut.", exc)
    return GrabCutSegmenter(), "fallback"
