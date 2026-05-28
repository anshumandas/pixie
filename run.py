#!/usr/bin/env python3
"""Launch the Pixie desktop app.

    python run.py

On first launch it auto-detects your hardware and configures itself. It works
immediately with built-in OpenCV methods (no downloads); it upgrades to SAM /
LaMa / Stable Diffusion automatically if those weights or a GPU/cloud backend
are available. See README.md.
"""
from __future__ import annotations

import logging
import sys


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Print the detected profile to the console before the GUI starts.
    from pixie.config import auto_configure

    cfg = auto_configure()
    print("Detected:", cfg.profile.summary())
    print(cfg.describe())

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        print(
            "\nPySide6 is not installed. Install the app dependencies first:\n"
            "    pip install -r requirements.txt\n"
            f"(import error: {exc})",
            file=sys.stderr,
        )
        return 1

    from pixie.ui import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
