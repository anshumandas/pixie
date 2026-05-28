"""Image canvas: shows the picture, forwards clicks (with Shift/Ctrl intent) as
image-pixel coords, and draws a translucent overlay for the current selection
mask plus markers for each prompt point."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert an HxWx3 (RGB) or HxWx4 (RGBA) uint8 array to a QPixmap.

    Builds the QImage from an immutable ``bytes`` buffer (via ``tobytes()``)
    rather than a numpy memoryview — the latter is rejected by some PySide6
    builds and is fragile w.r.t. buffer lifetime. ``copy()`` then gives the
    QImage its own storage so the temporary bytes can be freed safely.
    """
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 3 and arr.shape[2] == 4:
        buf = arr.tobytes()
        img = QImage(buf, w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        rgb = np.ascontiguousarray(arr[..., :3])
        buf = rgb.tobytes()
        img = QImage(buf, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class ImageCanvas(QGraphicsView):
    # (x, y, mode) where mode is 'new' | 'add' | 'remove'
    clicked = Signal(int, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)
        self._scene.addItem(self._overlay_item)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundBrush(Qt.darkGray)
        self.setMinimumSize(420, 360)
        self._has_image = False

    def set_image(self, arr: np.ndarray) -> None:
        self._image_item.setPixmap(ndarray_to_qpixmap(arr))
        self._overlay_item.setPixmap(QPixmap())
        self._scene.setSceneRect(self._image_item.boundingRect())
        self._has_image = True
        self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def set_overlay(self, mask, points=None, labels=None) -> None:
        if mask is None:
            self._overlay_item.setPixmap(QPixmap())
            return
        h, w = mask.shape[:2]
        rgba = np.zeros((h, w, 4), np.uint8)
        inside = mask > 0
        rgba[..., 0] = 60
        rgba[..., 1] = 150
        rgba[..., 2] = 255
        rgba[..., 3] = np.where(inside, 120, 0).astype(np.uint8)

        if points:
            labels = labels or [1] * len(points)
            half = max(3, int(0.006 * max(h, w)))
            for (px, py), lab in zip(points, labels):
                color = (80, 220, 120) if lab == 1 else (235, 80, 80)
                self._draw_marker(rgba, int(px), int(py), half, color)

        self._overlay_item.setPixmap(ndarray_to_qpixmap(rgba))

    @staticmethod
    def _draw_marker(rgba: np.ndarray, px: int, py: int, half: int, color) -> None:
        h, w = rgba.shape[:2]
        # white outline first
        bx0, bx1 = max(0, px - half - 1), min(w, px + half + 2)
        by0, by1 = max(0, py - half - 1), min(h, py + half + 2)
        rgba[by0:by1, bx0:bx1, :3] = 255
        rgba[by0:by1, bx0:bx1, 3] = 255
        # coloured centre
        x0, x1 = max(0, px - half), min(w, px + half + 1)
        y0, y1 = max(0, py - half), min(h, py + half + 1)
        rgba[y0:y1, x0:x1, 0] = color[0]
        rgba[y0:y1, x0:x1, 1] = color[1]
        rgba[y0:y1, x0:x1, 2] = color[2]
        rgba[y0:y1, x0:x1, 3] = 255

    def mousePressEvent(self, event) -> None:
        if self._has_image and event.button() == Qt.LeftButton:
            scene_pt = self.mapToScene(event.position().toPoint())
            item_pt = self._image_item.mapFromScene(scene_pt)
            x, y = int(item_pt.x()), int(item_pt.y())
            rect = self._image_item.boundingRect()
            if 0 <= x < rect.width() and 0 <= y < rect.height():
                mods = event.modifiers()
                if mods & Qt.ShiftModifier:
                    mode = "add"
                elif mods & Qt.ControlModifier:
                    mode = "remove"
                else:
                    mode = "new"
                self.clicked.emit(x, y, mode)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        if self._has_image:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)
        super().resizeEvent(event)
