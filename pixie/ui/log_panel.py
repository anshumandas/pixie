"""An in-app log panel.

Routes Python ``logging`` output (from any thread) into a read-only text view
so errors and tracebacks are visible inside the app, not just in a terminal.
A ``logging.Handler`` forwards records through a Qt signal; because the panel
lives on the GUI thread, Qt delivers those signals via a queued (thread-safe)
connection even when the log call came from a worker thread.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _LogEmitter(QObject):
    message = Signal(str, int)  # formatted text, levelno


class QtLogHandler(logging.Handler):
    """A logging handler that emits records through a Qt signal."""

    def __init__(self, level: int = logging.INFO):
        super().__init__(level)
        self.emitter = _LogEmitter()
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        # Queued delivery to the GUI thread (panel lives there).
        self.emitter.message.emit(msg, record.levelno)


class LogPanel(QWidget):
    """Read-only log view with Clear / Copy buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Activity / errors"))
        bar.addStretch(1)
        self.btn_copy = QPushButton("Copy")
        self.btn_clear = QPushButton("Clear")
        bar.addWidget(self.btn_copy)
        bar.addWidget(self.btn_clear)
        layout.addLayout(bar)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.view.setMaximumBlockCount(5000)
        self.view.setFont(QFont("Consolas", 9))
        layout.addWidget(self.view)

        self.btn_clear.clicked.connect(self.view.clear)
        self.btn_copy.clicked.connect(self._copy_all)

    def append_message(self, text: str, levelno: int = logging.INFO) -> None:
        # Colour-code warnings/errors so they stand out.
        if levelno >= logging.ERROR:
            html = f'<span style="color:#e06c6c;">{_escape(text)}</span>'
        elif levelno >= logging.WARNING:
            html = f'<span style="color:#d6a13a;">{_escape(text)}</span>'
        else:
            html = _escape(text)
        self.view.appendHtml(html)
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _copy_all(self) -> None:
        QApplication.clipboard().setText(self.view.toPlainText())


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
