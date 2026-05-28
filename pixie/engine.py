"""The editor engine — orchestrates models, ops, history, and backend routing.

The UI talks only to this class. It is deliberately free of any Qt imports so
the whole pipeline can be unit-tested headlessly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .backends import Operation, build_registry
from .config import AppConfig, auto_configure
from .models import apply_style, build_inpainter, build_segmenter
from .ops import History, adjust_brightness_contrast, composite_into, remove_background

log = logging.getLogger(__name__)


class EditorError(RuntimeError):
    pass


class Editor:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or auto_configure()
        self.registry = build_registry(self.config)

        self._segmenter = None
        self._inpainter = None

        self.full_res: Optional[np.ndarray] = None  # original RGB
        self.image: Optional[np.ndarray] = None      # working RGB (downscaled)
        self.scale: float = 1.0                       # working = full * scale
        self.mask: Optional[np.ndarray] = None        # current selection (working coords)
        self.history: Optional[History] = None

        # accumulated selection prompts: parallel lists of (x, y) and label
        # (1 = include / add, 0 = exclude / remove)
        self._points: list[tuple[int, int]] = []
        self._labels: list[int] = []

        # which backend handles each routable op (default from config)
        self.backend_choice = {
            Operation.GENERATIVE_FILL: self.config.default_backend_id,
            Operation.STYLIZE_NEURAL: self.config.default_backend_id,
        }

    # ------------------------------------------------------------------ models
    @property
    def segmenter(self):
        if self._segmenter is None:
            self._segmenter, cap = build_segmenter(self.config)
            self.config.capabilities.selection = cap
            self.config.capabilities.background_removal = cap
            if self.image is not None:
                self._segmenter.set_image(self.image)
        return self._segmenter

    @property
    def inpainter(self):
        if self._inpainter is None:
            self._inpainter, cap = build_inpainter(self.config)
            self.config.capabilities.erase_object = cap
        return self._inpainter

    # -------------------------------------------------------------------- I/O
    def _downscale(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = rgb.shape[:2]
        longest = max(h, w)
        budget = self.config.max_process_dim
        if longest <= budget:
            return rgb, 1.0
        scale = budget / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        try:
            import cv2  # type: ignore

            work = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image

            work = np.array(Image.fromarray(rgb).resize(new_size, Image.LANCZOS))
        log.info("Downscaled %sx%s -> %sx%s for processing", w, h, new_size[0], new_size[1])
        return work, scale

    def load_image_array(self, rgb: np.ndarray) -> None:
        if rgb.ndim != 3:
            raise EditorError("Expected an RGB image array.")
        self.full_res = rgb[..., :3].astype(np.uint8).copy()
        self.image, self.scale = self._downscale(self.full_res)
        self.mask = None
        self._points = []
        self._labels = []
        self.history = History(self.image)
        if self._segmenter is not None:
            self._segmenter.set_image(self.image)

    def load_image(self, path: str | Path) -> None:
        from PIL import Image

        rgb = np.array(Image.open(path).convert("RGB"))
        self.load_image_array(rgb)

    def export(self, path: str | Path) -> None:
        if self.history is None:
            raise EditorError("Nothing to export — load an image first.")
        from PIL import Image

        arr = self.current
        if arr.shape[-1] == 4:
            Image.fromarray(arr, "RGBA").save(path)
        else:
            Image.fromarray(arr[..., :3], "RGB").save(path)

    # ----------------------------------------------------------------- state
    @property
    def current(self) -> np.ndarray:
        if self.history is None:
            raise EditorError("No image loaded.")
        return self.history.current

    @property
    def has_image(self) -> bool:
        return self.history is not None

    @property
    def has_selection(self) -> bool:
        return self.mask is not None and bool(self.mask.any())

    def _require_selection(self) -> np.ndarray:
        if not self.has_selection:
            raise EditorError("Select an object first (click on it).")
        return self.mask  # type: ignore[return-value]

    # ------------------------------------------------------------- selection
    def select_at(self, x: int, y: int, mode: str = "new") -> np.ndarray:
        """Update the selection from a click.

        ``mode`` is one of:
          * 'new'    — start a fresh selection at (x, y)        [plain click]
          * 'add'    — add this point to the selection          [Shift+click]
          * 'remove' — subtract around this point               [Ctrl+click]
        """
        if not self.has_image:
            raise EditorError("Load an image first.")
        seg = self.segmenter  # builds + sets the image embedding on first use

        x, y = int(x), int(y)
        if mode == "add" and self._points:
            self._points.append((x, y))
            self._labels.append(1)
        elif mode == "remove" and self._points:
            self._points.append((x, y))
            self._labels.append(0)
        else:  # 'new' (or add/remove with no existing points)
            self._points = [(x, y)]
            self._labels = [1]

        self.mask = seg.predict(self._points, self._labels)
        return self.mask

    def selection_points(self) -> tuple[list[tuple[int, int]], list[int]]:
        """Current prompt points and their labels (1 = add, 0 = remove)."""
        return list(self._points), list(self._labels)

    def clear_selection(self) -> None:
        self.mask = None
        self._points = []
        self._labels = []

    # ---------------------------------------------------------------- edits
    def adjust(self, brightness: float, contrast: float, commit: bool = True) -> np.ndarray:
        """Brightness/contrast on the selection (or whole image if none)."""
        base = self.current
        out = adjust_brightness_contrast(base, brightness, contrast, self.mask)
        if commit:
            self.history.push(out)  # type: ignore[union-attr]
        return out

    def remove_background(self) -> np.ndarray:
        mask = self._require_selection()
        out = remove_background(self.current, mask)
        self.history.push(out)  # type: ignore[union-attr]
        return out

    def erase_selection(self) -> np.ndarray:
        mask = self._require_selection()
        out = self.inpainter.erase(self.current, mask)
        self.history.push(out)  # type: ignore[union-attr]
        self.clear_selection()
        return out

    def stylize(self, style: str, whole_image: bool = False) -> np.ndarray:
        styled = apply_style(self.current, style)
        if whole_image or not self.has_selection:
            out = styled
        else:
            out = composite_into(self.current, styled, self.mask)  # type: ignore[arg-type]
        self.history.push(out)  # type: ignore[union-attr]
        return out

    # --------------------------------------------------- routed (heavy) edits
    def _resolve_backend(self, op: Operation):
        chosen = self.backend_choice.get(op, self.config.default_backend_id)
        backend = self.registry.get(chosen) or self.registry.best_default_for(op)
        return backend

    def generative_fill(self, prompt: str) -> np.ndarray:
        mask = self._require_selection()
        backend = self._resolve_backend(Operation.GENERATIVE_FILL)
        if backend.kind == "local":
            raise EditorError(
                "Generative fill needs a GPU or a cloud backend. Your machine was "
                "auto-detected as CPU/integrated. Pick a cloud backend in Settings, "
                "or use 'Erase object' which works locally."
            )
        avail = backend.availability()
        if not avail.ready:
            raise EditorError(f"{backend.display_name} not configured: {avail.reason}")
        out = backend.run(Operation.GENERATIVE_FILL, image=self.current, mask=mask, prompt=prompt)
        self.history.push(out)  # type: ignore[union-attr]
        self.clear_selection()
        return out

    def neural_stylize(self, prompt: str) -> np.ndarray:
        backend = self._resolve_backend(Operation.STYLIZE_NEURAL)
        if backend.kind == "local":
            raise EditorError(
                "Neural/SD stylise needs a GPU or cloud backend. Try the built-in "
                "Ink / Shadow / Pencil / Oil styles, which run locally."
            )
        avail = backend.availability()
        if not avail.ready:
            raise EditorError(f"{backend.display_name} not configured: {avail.reason}")
        out = backend.run(Operation.STYLIZE_NEURAL, image=self.current, prompt=prompt)
        self.history.push(out)  # type: ignore[union-attr]
        return out

    # ----------------------------------------------------------- history etc.
    def undo(self) -> Optional[np.ndarray]:
        return self.history.undo() if self.history else None

    def redo(self) -> Optional[np.ndarray]:
        return self.history.redo() if self.history else None

    def set_backend(self, op: Operation, backend_id: str) -> None:
        self.backend_choice[op] = backend_id

    def backend_options(self, op: Operation):
        return self.registry.cost_table(op)
