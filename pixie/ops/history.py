"""A simple non-destructive undo/redo stack of image states."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class History:
    def __init__(self, initial: np.ndarray, max_states: int = 30):
        self._states: List[np.ndarray] = [initial.copy()]
        self._index = 0
        self._max = max_states

    @property
    def current(self) -> np.ndarray:
        return self._states[self._index]

    def push(self, state: np.ndarray) -> None:
        # drop any redo branch, then append
        del self._states[self._index + 1 :]
        self._states.append(state.copy())
        if len(self._states) > self._max:
            self._states.pop(0)
        self._index = len(self._states) - 1

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1

    def undo(self) -> Optional[np.ndarray]:
        if self.can_undo():
            self._index -= 1
            return self.current
        return None

    def redo(self) -> Optional[np.ndarray]:
        if self.can_redo():
            self._index += 1
            return self.current
        return None
