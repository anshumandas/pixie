"""Pixie main window.

Layout:
  * a **top toolbar** that invokes each action (open/save, remove background,
    erase, and the parametric tools: stylise, brightness/contrast, generative);
  * the **canvas** in the centre with a persistent selection hint above it;
  * a **contextual property panel** on the right (a QStackedWidget) that shows
    the controls for whichever tool is active;
  * a dockable **log panel** at the bottom.

Heavy operations run on a background thread pool so the UI stays responsive.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..backends import Operation
from ..engine import Editor, EditorError
from ..models import STYLES
from ..nlp import default_parser
from .backend_dialog import BackendPicker
from .canvas import ImageCanvas
from .log_panel import LogPanel, QtLogHandler
from .worker import Worker

log = logging.getLogger(__name__)

_PAGE_TITLES = {
    "home": "Info & commands",
    "stylize": "Stylise",
    "adjust": "Brightness / contrast",
    "generative": "Generative fill",
}

SELECTION_HINT = (
    "Select:  click an object   ·   Shift + Click to add to selection   ·   "
    "Ctrl + Click to remove from selection"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixie")
        self.resize(1240, 800)
        self.setMinimumSize(960, 640)

        self.editor = Editor()
        self.pool = QThreadPool.globalInstance()
        self.parser = default_parser()
        self._busy = False
        self._workers = set()
        self._pages: dict[str, int] = {}

        self.canvas = ImageCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.clicked.connect(self._on_canvas_click)

        self._build_toolbar()

        # central area: [hint + canvas]  |  [property panel]
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        left = QVBoxLayout()
        self.hint_label = QLabel(SELECTION_HINT)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "QLabel { padding: 6px 8px; border: 1px solid palette(mid);"
            " border-radius: 4px; }"
        )
        left.addWidget(self.hint_label)
        left.addWidget(self.canvas, stretch=1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        root.addWidget(left_widget, stretch=1)
        root.addWidget(self._build_property_panel())
        self.setCentralWidget(central)

        self._build_log_dock()

        self._refresh_capability_label()
        self._refresh_backend_label()
        self._show_page("home")
        self._set_busy(False)
        self.statusBar().showMessage(self.editor.config.profile.summary())

        self.log = logging.getLogger("pixie.ui")
        self.log.info("Pixie started.")
        self.log.info("Detected: %s", self.editor.config.profile.summary())
        for note in self.editor.config.profile.notes:
            self.log.info("note: %s", note)

    # --------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        self._toolbar = tb

        def action(text, slot, shortcut=None, tip=None):
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(shortcut)
            if tip:
                act.setToolTip(tip)
            act.triggered.connect(slot)
            tb.addAction(act)
            return act

        action("Open Image", self._open, "Ctrl+O", "Open an image to edit")
        action("Save", self._save, "Ctrl+S", "Save the current image")
        tb.addSeparator()
        action("Remove Background", self._remove_bg, tip="Keep the selected object, drop the rest")
        action("Erase Object", self._erase, tip="Delete the selection and fill behind it")
        tb.addSeparator()
        action("Stylise", lambda: self._show_page("stylize"), tip="Ink / shadow / pencil / oil")
        action("Brightness/Contrast", lambda: self._show_page("adjust"))
        action("Generative Fill", lambda: self._show_page("generative"), tip="GPU / cloud")
        tb.addSeparator()
        action("Undo", self._undo, "Ctrl+Z")
        action("Redo", self._redo, "Ctrl+Y")

    # -------------------------------------------------------- property panel
    def _build_property_panel(self) -> QWidget:
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        self.ctx_title = QLabel("Properties")
        f = self.ctx_title.font()
        f.setBold(True)
        self.ctx_title.setFont(f)
        v.addWidget(self.ctx_title)

        self.stack = QStackedWidget()
        self._add_page("home", self._page_home())
        self._add_page("stylize", self._page_stylize())
        self._add_page("adjust", self._page_adjust())
        self._add_page("generative", self._page_generative())
        v.addWidget(self.stack)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(340)
        scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._props_container = scroll
        return scroll

    def _add_page(self, name: str, widget: QWidget) -> None:
        self._pages[name] = self.stack.addWidget(widget)

    def _show_page(self, name: str) -> None:
        self.stack.setCurrentIndex(self._pages[name])
        self.ctx_title.setText(f"Properties — {_PAGE_TITLES.get(name, name)}")

    def _page_home(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        sysbox = QGroupBox("Detected setup (auto)")
        sb = QVBoxLayout(sysbox)
        self.caps_label = QLabel()
        self.caps_label.setWordWrap(True)
        sb.addWidget(self.caps_label)
        lay.addWidget(sysbox)

        cmdbox = QGroupBox("Command bar")
        cb = QVBoxLayout(cmdbox)
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("e.g. remove the background and darken it")
        self.cmd_input.returnPressed.connect(self._run_command)
        btn = QPushButton("Run command")
        btn.clicked.connect(self._run_command)
        cb.addWidget(self.cmd_input)
        cb.addWidget(btn)
        lay.addWidget(cmdbox)

        tips = QLabel(
            "Tip: pick a tool from the toolbar above. Use Remove Background or "
            "Erase Object after selecting something on the canvas."
        )
        tips.setWordWrap(True)
        lay.addWidget(tips)
        return w

    def _page_stylize(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.style_combo = QComboBox()
        self.style_combo.addItems(list(STYLES.keys()))
        self.style_whole = QCheckBox("Apply to whole image")
        btn = QPushButton("Apply style")
        btn.clicked.connect(self._apply_style)
        form.addRow("Style:", self.style_combo)
        form.addRow(self.style_whole)
        form.addRow(btn)
        return w

    def _page_adjust(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.s_bright = QSlider(Qt.Horizontal)
        self.s_bright.setRange(-100, 100)
        self.s_bright.sliderReleased.connect(self._preview_adjust)
        self.s_contrast = QSlider(Qt.Horizontal)
        self.s_contrast.setRange(-100, 100)
        self.s_contrast.sliderReleased.connect(self._preview_adjust)
        btn = QPushButton("Apply adjustment")
        btn.clicked.connect(self._apply_adjust)
        form.addRow("Brightness:", self.s_bright)
        form.addRow("Contrast:", self.s_contrast)
        form.addRow(btn)
        note = QLabel("Applies to the current selection (or the whole image if nothing is selected).")
        note.setWordWrap(True)
        form.addRow(note)
        return w

    def _page_generative(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.gen_prompt = QLineEdit()
        self.gen_prompt.setPlaceholderText("Replace selection with… (e.g. a potted plant)")
        lay.addWidget(self.gen_prompt)
        btn_gen = QPushButton("Generative fill")
        btn_gen.clicked.connect(self._generative_fill)
        lay.addWidget(btn_gen)
        self.backend_label = QLabel()
        self.backend_label.setWordWrap(True)
        lay.addWidget(self.backend_label)
        btn_backend = QPushButton("Choose backend (shows cost)…")
        btn_backend.clicked.connect(self._choose_backend)
        lay.addWidget(btn_backend)
        return w

    # ------------------------------------------------------------- log dock
    def _build_log_dock(self) -> None:
        self.log_panel = LogPanel()
        dock = QDockWidget("Log", self)
        dock.setObjectName("log_dock")
        dock.setWidget(self.log_panel)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self._log_dock = dock

        handler = QtLogHandler(level=logging.INFO)
        handler.emitter.message.connect(self.log_panel.append_message)
        root_logger = logging.getLogger()
        if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        self._log_handler = handler

    # --------------------------------------------------------------- labels
    def _refresh_capability_label(self) -> None:
        caps = self.editor.config.capabilities.as_dict()
        lines = [self.editor.config.profile.summary(), ""]
        for k, val in caps.items():
            badge = {"local": "local ✓", "fallback": "built-in ✓", "cloud": "needs GPU/cloud"}.get(
                val, val
            )
            lines.append(f"• {k}: {badge}")
        self.caps_label.setText("\n".join(lines))

    def _refresh_backend_label(self) -> None:
        bid = self.editor.backend_choice.get(
            Operation.GENERATIVE_FILL, self.editor.config.default_backend_id
        )
        backend = self.editor.registry.get(bid)
        if backend is None:
            self.backend_label.setText("Backend: (none)")
            return
        avail = backend.availability()
        status = "ready" if avail.ready else f"needs setup: {avail.reason}"
        cost = backend.estimate_cost(Operation.GENERATIVE_FILL).display()
        self.backend_label.setText(f"Backend: {backend.display_name}\n{cost}\n({status})")

    # --------------------------------------------------------------- helpers
    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self._busy = busy
        self._toolbar.setEnabled(not busy)
        self._props_container.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
        if msg:
            self.statusBar().showMessage(msg)

    def _run_async(self, fn, on_done, busy_msg: str, requires_image: bool = True) -> None:
        if self._busy:
            return
        if requires_image and not self.editor.has_image:
            self._warn("Open an image first.")
            return
        self._set_busy(True, busy_msg)
        worker = Worker(fn)
        worker.setAutoDelete(False)  # we own its lifetime via self._workers
        self._workers.add(worker)

        def _cleanup(*_args, _w=worker):
            self._workers.discard(_w)

        worker.signals.finished.connect(lambda r: self._after(on_done, r))
        worker.signals.failed.connect(self._on_failed)
        worker.signals.finished.connect(_cleanup)
        worker.signals.failed.connect(_cleanup)
        self.pool.start(worker)

    def _after(self, on_done, result) -> None:
        self._set_busy(False)
        try:
            on_done(result)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Post-processing failed")
            self._warn(str(exc))
        self.statusBar().showMessage(self.editor.config.profile.summary())

    def _on_failed(self, message: str, tb: str = "") -> None:
        self._set_busy(False)
        if tb:
            log.error("Operation failed:\n%s", tb)
        else:
            log.error("Operation failed: %s", message)
        self._warn(message or "Operation failed (see the Log panel for details).")

    def _warn(self, message: str) -> None:
        QMessageBox.information(self, "Pixie", message)

    def _show_image(self, arr: np.ndarray) -> None:
        if arr is not None:
            self.canvas.set_image(arr)

    # ----------------------------------------------------------------- slots
    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
        )
        if not path:
            return

        def do_load():
            self.editor.load_image(path)
            return self.editor.current

        def done(img):
            self._show_image(img)
            self.canvas.set_overlay(None)
            self._refresh_capability_label()
            h, w = img.shape[:2]
            log.info("Loaded %s  (working size %dx%d)", path, w, h)

        log.info("Opening %s", path)
        self._run_async(do_load, done, f"Loading {path}…", requires_image=False)

    def _save(self) -> None:
        if not self.editor.has_image:
            self._warn("Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "edited.png", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        try:
            self.editor.export(path)
            self.statusBar().showMessage(f"Saved {path}")
            log.info("Saved %s", path)
        except Exception as exc:
            log.exception("Save failed")
            self._warn(str(exc))

    def _on_canvas_click(self, x: int, y: int, mode: str) -> None:
        def done(mask):
            pts, labs = self.editor.selection_points()
            self.canvas.set_overlay(mask, pts, labs)
            adds = sum(1 for v in labs if v == 1)
            subs = sum(1 for v in labs if v == 0)
            log.info("Selection updated: %d add, %d remove point(s)", adds, subs)

        busy = {"new": "Selecting…", "add": "Adding to selection…",
                "remove": "Removing from selection…"}.get(mode, "Selecting…")
        self._run_async(lambda: self.editor.select_at(x, y, mode), done, busy)

    def _remove_bg(self) -> None:
        self._run_async(self.editor.remove_background, self._show_image, "Removing background…")

    def _erase(self) -> None:
        def done(img):
            self._show_image(img)
            self.canvas.set_overlay(None)

        self._run_async(self.editor.erase_selection, done, "Erasing object…")

    def _apply_style(self) -> None:
        style = self.style_combo.currentText()
        whole = self.style_whole.isChecked()
        self._run_async(
            lambda: self.editor.stylize(style, whole_image=whole),
            self._show_image,
            f"Applying {style}…",
        )

    def _preview_adjust(self) -> None:
        if self._busy or not self.editor.has_image:
            return
        b = self.s_bright.value() / 100.0
        c = self.s_contrast.value() / 100.0
        try:
            self._show_image(self.editor.adjust(b, c, commit=False))
        except EditorError:
            pass

    def _apply_adjust(self) -> None:
        b = self.s_bright.value() / 100.0
        c = self.s_contrast.value() / 100.0

        def done(img):
            self._show_image(img)
            self.s_bright.setValue(0)
            self.s_contrast.setValue(0)

        self._run_async(lambda: self.editor.adjust(b, c, commit=True), done, "Applying adjustment…")

    def _generative_fill(self) -> None:
        prompt = self.gen_prompt.text().strip()
        self._run_async(
            lambda: self.editor.generative_fill(prompt),
            self._show_image,
            "Generative fill…",
        )

    def _choose_backend(self) -> None:
        op = Operation.GENERATIVE_FILL
        rows = self.editor.backend_options(op)
        current = self.editor.backend_choice.get(op, self.editor.config.default_backend_id)
        dlg = BackendPicker(op.label, rows, current, self)
        if dlg.exec():
            chosen = dlg.selected_id()
            if chosen:
                self.editor.set_backend(Operation.GENERATIVE_FILL, chosen)
                self.editor.set_backend(Operation.STYLIZE_NEURAL, chosen)
                self._refresh_backend_label()
                self.statusBar().showMessage(f"Backend for generative ops: {chosen}")
                log.info("Generative backend set to %s", chosen)

    def _run_command(self) -> None:
        text = self.cmd_input.text().strip()
        if not text:
            return
        cmds = self.parser.parse(text)
        if not cmds:
            self._warn("Didn't understand that. Try: 'remove background', 'erase', 'ink', 'darker'.")
            return

        def do_all():
            last = self.editor.current
            for cmd in cmds:
                if cmd.action == "remove_background" and self.editor.has_selection:
                    last = self.editor.remove_background()
                elif cmd.action == "erase" and self.editor.has_selection:
                    last = self.editor.erase_selection()
                elif cmd.action == "stylize":
                    last = self.editor.stylize(
                        cmd.args.get("style", "Ink"),
                        whole_image=not self.editor.has_selection,
                    )
                elif cmd.action == "adjust":
                    last = self.editor.adjust(
                        float(cmd.args.get("brightness", 0.0)),
                        float(cmd.args.get("contrast", 0.0)),
                        commit=True,
                    )
            return last

        log.info("Command: %s", "; ".join(str(c) for c in cmds))
        self._run_async(do_all, self._show_image, "Running command…")

    def _undo(self) -> None:
        img = self.editor.undo()
        if img is not None:
            self.editor.clear_selection()
            self._show_image(img)
            self.canvas.set_overlay(None)

    def _redo(self) -> None:
        img = self.editor.redo()
        if img is not None:
            self._show_image(img)
