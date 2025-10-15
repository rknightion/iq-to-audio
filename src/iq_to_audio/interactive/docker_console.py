from __future__ import annotations

import html
from pathlib import Path

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal


class DockerConsoleDialog(QtWidgets.QDialog):
    """Simple dialog that tails container logs and reserves space for interactive input."""

    cancel_requested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Digital decoder console")
        self.setModal(False)
        self.resize(720, 480)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.header_label = QtWidgets.QLabel("")
        self.header_label.setTextFormat(Qt.TextFormat.RichText)
        self.header_label.setWordWrap(True)
        layout.addWidget(self.header_label)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setPlaceholderText("Container logs will appear here.")
        layout.addWidget(self.log_view, stretch=1)

        self.input_entry = QtWidgets.QLineEdit()
        self.input_entry.setPlaceholderText("Interactive shell input (future release)")
        self.input_entry.setEnabled(False)
        layout.addWidget(self.input_entry)

        self.button_box = QtWidgets.QDialogButtonBox()
        self.stop_button = self.button_box.addButton(
            "Stop Container", QtWidgets.QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.close_button = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._running = False

    def _on_stop_clicked(self) -> None:
        """Handle stop button click by emitting cancellation signal."""
        if self._running:
            self.cancel_requested.emit()
            self.stop_button.setEnabled(False)
            self.append_log("\n[user] Cancellation requested...\n")

    def prepare(self, *, decoder_label: str, command: tuple[str, ...], audio_dir: Path) -> None:
        """Reset the dialog for a new container session."""
        decoder_html = html.escape(decoder_label)
        command_html = html.escape(" ".join(command))
        audio_html = html.escape(audio_dir.as_posix())
        self.header_label.setText(
            f"<b>{decoder_html}</b><br/>"
            f"Command: <code>{command_html}</code><br/>"
            f"Audio mount: <code>{audio_html}</code>"
        )
        self.log_view.clear()
        self.input_entry.clear()

    def append_log(self, text: str) -> None:
        cursor = self.log_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def set_running(self, running: bool) -> None:
        self._running = running
        self.stop_button.setEnabled(running)
        self.close_button.setEnabled(not running)
        title = "Digital decoder — running" if running else "Digital decoder — finished"
        self.setWindowTitle(title)

    def closeEvent(  # noqa: N802 - Qt override uses camelCase
        self, event: QtGui.QCloseEvent
    ) -> None:  # pragma: no cover - Qt handles UI cleanup
        self._running = False
        super().closeEvent(event)
