from __future__ import annotations

from collections.abc import Sequence

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from .state import COLOR_THEMES, InteractiveState
from .widgets import PanelGroup


class RecordingPanel(PanelGroup):
    def __init__(self, state: InteractiveState) -> None:
        super().__init__("Recording")
        layout = QtWidgets.QVBoxLayout()

        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(QtWidgets.QLabel("Input WAV:"))
        self.file_entry = QtWidgets.QLineEdit(state.selected_path.as_posix() if state.selected_path else "")
        self.file_entry.setPlaceholderText("Select a baseband WAV recording…")
        file_row.addWidget(self.file_entry, stretch=1)
        self.browse_button = QtWidgets.QPushButton("Browse…")
        file_row.addWidget(self.browse_button)
        layout.addLayout(file_row)

        format_row = QtWidgets.QHBoxLayout()
        format_row.addWidget(QtWidgets.QLabel("Input format:"))
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.setMinimumWidth(220)
        format_row.addWidget(self.format_combo)
        self.format_status_label = QtWidgets.QLabel(state.input_format_message)
        self.format_status_label.setWordWrap(True)
        format_row.addWidget(self.format_status_label, stretch=1)
        layout.addLayout(format_row)

        center_row = QtWidgets.QHBoxLayout()
        center_row.addWidget(QtWidgets.QLabel("Center freq (Hz):"))
        self.center_entry = QtWidgets.QLineEdit(
            f"{state.center_freq:.0f}" if state.center_freq is not None else ""
        )
        self.center_entry.setMinimumWidth(160)
        self.center_entry.setMaximumWidth(200)
        center_row.addWidget(self.center_entry)
        self.detect_button = QtWidgets.QPushButton("Detect from file")
        center_row.addWidget(self.detect_button)
        self.center_source_label = QtWidgets.QLabel(
            f"Center source: {state.center_source or '—'}"
        )
        self.center_source_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.center_source_label.setWordWrap(True)
        center_row.addWidget(self.center_source_label, stretch=1)
        layout.addLayout(center_row)

        snapshot_row = QtWidgets.QHBoxLayout()
        snapshot_row.addWidget(QtWidgets.QLabel("Snapshot (seconds):"))
        self.snapshot_entry = QtWidgets.QLineEdit(f"{state.snapshot_seconds:.2f}")
        self.snapshot_entry.setMinimumWidth(80)
        self.snapshot_entry.setMaximumWidth(100)
        snapshot_row.addWidget(self.snapshot_entry)
        self.full_snapshot_check = QtWidgets.QCheckBox("Analyze entire recording")
        self.full_snapshot_check.setChecked(state.full_snapshot)
        snapshot_row.addWidget(self.full_snapshot_check)
        self.load_fft_button = QtWidgets.QPushButton("Load FFT")
        snapshot_row.addWidget(self.load_fft_button)
        snapshot_row.addStretch(1)
        layout.addLayout(snapshot_row)

        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(QtWidgets.QLabel("Output directory:"))
        self.output_entry = QtWidgets.QLineEdit(state.output_dir.as_posix() if state.output_dir else "")
        self.output_entry.setPlaceholderText("Optional override – defaults beside input WAV")
        output_row.addWidget(self.output_entry, stretch=1)
        self.output_browse_button = QtWidgets.QPushButton("Browse…")
        output_row.addWidget(self.output_browse_button)
        layout.addLayout(output_row)

        self.output_hint_label = QtWidgets.QLabel(state.output_hint)
        self.output_hint_label.setWordWrap(True)
        layout.addWidget(self.output_hint_label)

        agc_row = QtWidgets.QHBoxLayout()
        self.agc_check = QtWidgets.QCheckBox("Automatic gain control")
        self.agc_check.setChecked(state.agc_enabled)
        agc_row.addWidget(self.agc_check)
        agc_row.addStretch()
        layout.addLayout(agc_row)

        self.set_layout(layout)


class DemodPanel(PanelGroup):
    def __init__(self, state: InteractiveState, options: Sequence[tuple[str, str, str]]) -> None:
        super().__init__("Demodulation")
        layout = QtWidgets.QVBoxLayout()

        self.mode_combo = QtWidgets.QComboBox()
        for value, label, _description in options:
            self.mode_combo.addItem(label, value)
        current_index = self.mode_combo.findData(state.demod_mode)
        if current_index >= 0:
            self.mode_combo.setCurrentIndex(current_index)
        layout.addWidget(self.mode_combo)

        self.description_label = QtWidgets.QLabel()
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        self.set_layout(layout)


class ChannelPanel(PanelGroup):
    def __init__(self, state: InteractiveState) -> None:
        super().__init__("Channel selection")
        layout = QtWidgets.QVBoxLayout()

        self.sample_rate_label = QtWidgets.QLabel(
            f"Sample rate: {state.sample_rate:,.0f} Hz" if state.sample_rate else "Sample rate: —"
        )
        layout.addWidget(self.sample_rate_label)

        sr_row = QtWidgets.QHBoxLayout()
        sr_row.addWidget(QtWidgets.QLabel("Override (Hz):"))
        self.sample_rate_entry = QtWidgets.QLineEdit(
            f"{state.sample_rate_override:.0f}" if state.sample_rate_override else ""
        )
        self.sample_rate_entry.setPlaceholderText("Auto")
        self.sample_rate_entry.setMinimumWidth(140)
        self.sample_rate_entry.setMaximumWidth(200)
        sr_row.addWidget(self.sample_rate_entry)
        sr_row.addStretch(1)
        layout.addLayout(sr_row)

        self.sample_rate_hint = QtWidgets.QLabel("")
        self.sample_rate_hint.setWordWrap(True)
        layout.addWidget(self.sample_rate_hint)

        bandwidth_row = QtWidgets.QHBoxLayout()
        bandwidth_row.addWidget(QtWidgets.QLabel("Bandwidth (Hz):"))
        self.bandwidth_entry = QtWidgets.QLineEdit(
            f"{state.bandwidth_hz:.0f}" if state.bandwidth_hz else ""
        )
        self.bandwidth_entry.setMinimumWidth(140)
        self.bandwidth_entry.setMaximumWidth(200)
        bandwidth_row.addWidget(self.bandwidth_entry)
        bandwidth_row.addStretch()
        layout.addLayout(bandwidth_row)

        self.offset_label = QtWidgets.QLabel("Offset: —")
        layout.addWidget(self.offset_label)

        self.set_layout(layout)


class TargetsPanel(PanelGroup):
    def __init__(self, state: InteractiveState) -> None:
        super().__init__("Target frequencies (Hz)")
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(12, 6, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self.entries: list[QtWidgets.QLineEdit] = []
        self.helper_label = QtWidgets.QLabel(
            "Click the spectrum preview to add target frequencies. "
            "Selections will fill the next empty slot."
        )
        self.helper_label.setWordWrap(True)
        self.helper_label.setStyleSheet("color: palette(windowText);")
        self.clear_button = QtWidgets.QPushButton("Clear all targets")
        self.clear_button.setAutoDefault(False)
        self.clear_button.setMinimumWidth(160)

        per_row = 2
        labels = state.target_text or [""] * state.max_target_freqs
        for idx in range(state.max_target_freqs):
            row = idx // per_row
            col = (idx % per_row) * 2
            layout.addWidget(QtWidgets.QLabel(f"Target {idx + 1}:"), row, col)
            entry = QtWidgets.QLineEdit(labels[idx] if idx < len(labels) else "")
            entry.setMinimumWidth(160)
            layout.addWidget(entry, row, col + 1)
            self.entries.append(entry)

        rows = (state.max_target_freqs + per_row - 1) // per_row
        layout.addWidget(self.helper_label, rows, 0, 1, per_row * 2)
        layout.addWidget(self.clear_button, rows + 1, 0, 1, per_row * 2)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self.set_layout(layout)


class StatusPanel(PanelGroup):
    def __init__(self) -> None:
        super().__init__("Status & Progress")
        layout = QtWidgets.QVBoxLayout()

        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        button_row = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Preview DSP")
        button_row.addWidget(self.preview_button)
        self.confirm_button = QtWidgets.QPushButton("Confirm && Run")
        button_row.addWidget(self.confirm_button)
        self.stop_button = QtWidgets.QPushButton("Stop preview")
        button_row.addWidget(self.stop_button)
        self.cancel_button = QtWidgets.QPushButton("Close")
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.set_layout(layout)


class SpectrumOptionsPanel(PanelGroup):
    def __init__(self, state: InteractiveState) -> None:
        super().__init__("Spectrum options")
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)

        layout.addWidget(QtWidgets.QLabel("FFT size"), 0, 0)
        self.nfft_combo = QtWidgets.QComboBox()
        self.nfft_combo.addItems(["65536", "131072", "262144", "524288"])
        self.nfft_combo.setCurrentText(str(state.nfft))
        layout.addWidget(self.nfft_combo, 0, 1)

        layout.addWidget(QtWidgets.QLabel("Smoothing"), 0, 2)
        self.smoothing_spin = QtWidgets.QSpinBox()
        self.smoothing_spin.setRange(1, 20)
        self.smoothing_spin.setValue(state.smoothing)
        layout.addWidget(self.smoothing_spin, 0, 3)

        layout.addWidget(QtWidgets.QLabel("Dynamic range (dB)"), 1, 0)
        self.range_spin = QtWidgets.QSpinBox()
        self.range_spin.setRange(20, 140)
        self.range_spin.setValue(state.dynamic_range)
        layout.addWidget(self.range_spin, 1, 1)

        layout.addWidget(QtWidgets.QLabel("Theme"), 1, 2)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(list(COLOR_THEMES.keys()))
        self.theme_combo.setCurrentText(state.theme if state.theme in COLOR_THEMES else "contrast")
        layout.addWidget(self.theme_combo, 1, 3)

        self.refresh_button = QtWidgets.QPushButton("Refresh preview")
        layout.addWidget(self.refresh_button, 2, 0, 1, 2)

        self.reset_button = QtWidgets.QPushButton("Reset defaults")
        layout.addWidget(self.reset_button, 2, 2, 1, 2)

        self.set_layout(layout)


class WaterfallOptionsPanel(PanelGroup):
    def __init__(self, state: InteractiveState) -> None:
        super().__init__("Waterfall options")
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Max slices"), 0, 0)
        self.slice_spin = QtWidgets.QSpinBox()
        self.slice_spin.setRange(50, 1000)
        self.slice_spin.setValue(state.waterfall_slices)
        layout.addWidget(self.slice_spin, 0, 1)

        layout.addWidget(QtWidgets.QLabel("Range (dB)"), 0, 2)
        self.floor_spin = QtWidgets.QSpinBox()
        self.floor_spin.setRange(20, 140)
        self.floor_spin.setValue(state.waterfall_floor)
        layout.addWidget(self.floor_spin, 0, 3)

        layout.addWidget(QtWidgets.QLabel("Colormap"), 1, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis"])
        self.cmap_combo.setCurrentText(state.waterfall_cmap)
        layout.addWidget(self.cmap_combo, 1, 1)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self.set_layout(layout)
