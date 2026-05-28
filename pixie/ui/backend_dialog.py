"""A dialog that lets the user pick which backend runs a heavy operation,
showing the estimated cost and privacy note for each option at runtime."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)


class BackendPicker(QDialog):
    def __init__(self, op_label: str, rows: List[dict], current_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Choose backend for: {op_label}")
        self.setMinimumWidth(460)
        self._chosen: Optional[str] = current_id

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{op_label}</b> — pick where it runs:"))

        self._group = QButtonGroup(self)
        self._buttons = {}
        for row in rows:
            ready = row["ready"] == "yes"
            text = f'{row["name"]}\n    {row["cost"]}  ·  {row["privacy"]}'
            if not ready:
                text += f'  ·  needs setup: {row["ready"][5:]}'  # strip "no — "
            btn = QRadioButton(text)
            btn.setEnabled(ready or row["id"] == current_id)
            if row["id"] == current_id:
                btn.setChecked(True)
            self._group.addButton(btn)
            self._buttons[btn] = row["id"]
            layout.addWidget(btn)

        note = QLabel(
            "<i>Cloud options upload your image off-device. Costs are approximate "
            "and billed by the provider.</i>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_id(self) -> Optional[str]:
        for btn, bid in self._buttons.items():
            if btn.isChecked():
                return bid
        return self._chosen
