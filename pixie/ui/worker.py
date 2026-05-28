"""Run a blocking callable on a background thread so the UI stays responsive."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    finished = Signal(object)
    # failed carries (short_message, full_traceback) so the UI can show a concise
    # popup while logging the complete traceback to the in-app log panel.
    failed = Signal(str, str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:  # noqa: D401
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # surface to the UI thread
            self.signals.failed.emit(str(exc), traceback.format_exc())
        else:
            self.signals.finished.emit(result)
