"""Pure image operations: adjustments, compositing, and edit history."""
from .adjustments import adjust_brightness_contrast
from .compositing import composite_into, feather_alpha, remove_background
from .history import History

__all__ = [
    "adjust_brightness_contrast",
    "composite_into",
    "feather_alpha",
    "remove_background",
    "History",
]
