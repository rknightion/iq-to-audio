from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from ..squelch import AudioPostOptions, SquelchConfig, SquelchSummary
from .state import InteractiveState
from .widgets import PanelGroup


class AudioPostPage(QtWidgets.QWidget):
    """Audio post-processing UI for squelch cleanup."""

    process_requested = Signal(Path, object)

    def __init__(self, *, state: InteractiveState) -> None:
        super().__init__()
        self.state = state
        self._recent_capture_path: Path | None = state.selected_path
        self._recent_output_dir: Path | None = state.resolved_output_dir()
        self._processing = False

        self._control_widgets: list[QtWidgets.QWidget] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.source_panel = self._build_source_panel()
        self.options_panel = self._build_options_panel()
        self.output_panel = self._build_output_panel()
        self.results_panel = self._build_results_panel()

        layout.addWidget(self.source_panel)
        layout.addWidget(self.options_panel)
        layout.addWidget(self.output_panel)
        layout.addWidget(self.results_panel, stretch=1)

        self._update_recent_labels()
        self._apply_state_defaults()
        self._refresh_enablement()

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def _build_source_panel(self) -> PanelGroup:
        panel = PanelGroup("Source recordings")
        panel_layout = QtWidgets.QVBoxLayout()
        panel_layout.setSpacing(12)

        self.recent_capture_label = QtWidgets.QLabel("")
        self.recent_capture_label.setWordWrap(True)
        panel_layout.addWidget(self.recent_capture_label)

        selector_group = QtWidgets.QButtonGroup(self)

        self.use_recent_radio = QtWidgets.QRadioButton("Use latest capture output")
        self.use_recent_radio.setChecked(True)
        selector_group.addButton(self.use_recent_radio)
        panel_layout.addWidget(self.use_recent_radio)
        self._control_widgets.append(self.use_recent_radio)

        self.use_specific_file_radio = QtWidgets.QRadioButton("Choose individual file")
        selector_group.addButton(self.use_specific_file_radio)
        panel_layout.addWidget(self.use_specific_file_radio)
        self._control_widgets.append(self.use_specific_file_radio)

        self.use_directory_radio = QtWidgets.QRadioButton("Choose directory of files")
        selector_group.addButton(self.use_directory_radio)
        panel_layout.addWidget(self.use_directory_radio)
        self._control_widgets.append(self.use_directory_radio)

        path_row = QtWidgets.QHBoxLayout()
        self.path_entry = QtWidgets.QLineEdit()
        self.path_entry.setPlaceholderText("Outputs directory will be used.")
        path_row.addWidget(self.path_entry, stretch=1)
        self._control_widgets.append(self.path_entry)

        self.choose_file_button = QtWidgets.QPushButton("Browse File…")
        path_row.addWidget(self.choose_file_button)
        self._control_widgets.append(self.choose_file_button)

        self.choose_directory_button = QtWidgets.QPushButton("Browse Folder…")
        path_row.addWidget(self.choose_directory_button)
        self._control_widgets.append(self.choose_directory_button)
        panel_layout.addLayout(path_row)

        selector_group.buttonToggled.connect(self._on_selection_mode_changed)
        self.choose_file_button.clicked.connect(self._on_choose_file)
        self.choose_directory_button.clicked.connect(self._on_choose_directory)

        self.path_entry.setEnabled(False)
        self.choose_file_button.setEnabled(False)
        self.choose_directory_button.setEnabled(False)

        panel.set_layout(panel_layout)
        return panel

    def _build_options_panel(self) -> PanelGroup:
        panel = PanelGroup("Auto squelch & cleanup")
        panel_layout = QtWidgets.QVBoxLayout()
        metrics = self.fontMetrics()
        base_spacing = max(12, int(metrics.height() * 1.15))
        row_spacing = max(8, int(metrics.height() * 0.8))
        panel_layout.setSpacing(base_spacing)

        method_row = QtWidgets.QHBoxLayout()
        method_row.setContentsMargins(0, 0, 0, 0)
        method_row.setSpacing(row_spacing)
        method_row.addWidget(QtWidgets.QLabel("Squelch method:"))
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem("Adaptive (voice)", "adaptive")
        self.method_combo.addItem("Static threshold", "static")
        self.method_combo.addItem("Transient bursts (digital)", "transient")
        self.method_combo.addItem("WebRTC VAD (speech-trained)", "webrtc")
        method_row.addWidget(self.method_combo, stretch=1)
        panel_layout.addLayout(method_row)
        self._control_widgets.append(self.method_combo)
        self._elevate_height(self.method_combo)

        noise_mode_row = QtWidgets.QHBoxLayout()
        noise_mode_row.setContentsMargins(0, 0, 0, 0)
        noise_mode_row.setSpacing(row_spacing)
        noise_mode_row.addWidget(QtWidgets.QLabel("Noise floor mode:"))
        self.noise_mode_combo = QtWidgets.QComboBox()
        self.noise_mode_combo.addItem("Auto detect (percentile)")
        self.noise_mode_combo.addItem("Manual (dBFS)")
        noise_mode_row.addWidget(self.noise_mode_combo, stretch=1)
        panel_layout.addLayout(noise_mode_row)
        self._control_widgets.append(self.noise_mode_combo)
        self._elevate_height(self.noise_mode_combo)

        manual_row = QtWidgets.QHBoxLayout()
        manual_row.setContentsMargins(0, 0, 0, 0)
        manual_row.setSpacing(row_spacing)
        manual_row.addWidget(QtWidgets.QLabel("Manual floor (dBFS):"))
        self.noise_floor_spin = QtWidgets.QDoubleSpinBox()
        self.noise_floor_spin.setRange(-140.0, 0.0)
        self.noise_floor_spin.setDecimals(1)
        self.noise_floor_spin.setSingleStep(1.0)
        self.noise_floor_spin.setValue(-55.0)
        self.noise_floor_spin.setSuffix(" dB")
        manual_row.addWidget(self.noise_floor_spin)
        manual_row.addStretch(1)
        panel_layout.addLayout(manual_row)
        self._control_widgets.append(self.noise_floor_spin)
        self._elevate_height(self.noise_floor_spin)

        auto_row = QtWidgets.QHBoxLayout()
        auto_row.setContentsMargins(0, 0, 0, 0)
        auto_row.setSpacing(row_spacing)
        auto_row.addWidget(QtWidgets.QLabel("Auto percentile:"))
        self.percentile_spin = QtWidgets.QDoubleSpinBox()
        self.percentile_spin.setRange(0.01, 1.0)
        self.percentile_spin.setSingleStep(0.05)
        self.percentile_spin.setDecimals(2)
        self.percentile_spin.setValue(0.20)
        auto_row.addWidget(self.percentile_spin)
        auto_row.addWidget(QtWidgets.QLabel("(0.01 → 1.0)"))
        auto_row.addStretch(1)
        panel_layout.addLayout(auto_row)
        self._control_widgets.append(self.percentile_spin)
        self._elevate_height(self.percentile_spin)

        margin_row = QtWidgets.QHBoxLayout()
        margin_row.setContentsMargins(0, 0, 0, 0)
        margin_row.setSpacing(row_spacing)
        margin_row.addWidget(QtWidgets.QLabel("Threshold margin (dB):"))
        self.margin_spin = QtWidgets.QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 30.0)
        self.margin_spin.setSingleStep(0.5)
        self.margin_spin.setValue(6.0)
        margin_row.addWidget(self.margin_spin)
        margin_row.addStretch(1)
        panel_layout.addLayout(margin_row)
        self._control_widgets.append(self.margin_spin)
        self._elevate_height(self.margin_spin)

        self.webrtc_row_widget = QtWidgets.QWidget()
        webrtc_row = QtWidgets.QHBoxLayout(self.webrtc_row_widget)
        webrtc_row.setContentsMargins(0, 0, 0, 0)
        webrtc_row.setSpacing(row_spacing)
        webrtc_row.addWidget(QtWidgets.QLabel("WebRTC aggressiveness:"))
        self.webrtc_mode_combo = QtWidgets.QComboBox()
        self.webrtc_mode_combo.addItem("0 — Lowest (sensitive)", 0)
        self.webrtc_mode_combo.addItem("1 — Low", 1)
        self.webrtc_mode_combo.addItem("2 — Balanced", 2)
        self.webrtc_mode_combo.addItem("3 — Highest (strict)", 3)
        webrtc_row.addWidget(self.webrtc_mode_combo)
        self._elevate_height(self.webrtc_mode_combo)
        webrtc_row.addWidget(QtWidgets.QLabel("Frame (ms):"))
        self.webrtc_frame_combo = QtWidgets.QComboBox()
        for frame_ms in (10, 20, 30):
            self.webrtc_frame_combo.addItem(f"{frame_ms}", frame_ms)
        webrtc_row.addWidget(self.webrtc_frame_combo)
        self._elevate_height(self.webrtc_frame_combo)
        webrtc_row.addStretch(1)
        panel_layout.addWidget(self.webrtc_row_widget)
        self._control_widgets.extend([self.webrtc_mode_combo, self.webrtc_frame_combo])

        trim_row = QtWidgets.QHBoxLayout()
        trim_row.setContentsMargins(0, 0, 0, 0)
        trim_row.setSpacing(row_spacing)
        self.trim_silence_check = QtWidgets.QCheckBox("Trim silence after squelch")
        self.trim_silence_check.setChecked(True)
        trim_row.addWidget(self.trim_silence_check)

        self.lead_in_spin = QtWidgets.QDoubleSpinBox()
        self.lead_in_spin.setPrefix("Lead-in ")
        self.lead_in_spin.setSuffix(" s")
        self.lead_in_spin.setRange(0.0, 5.0)
        self.lead_in_spin.setSingleStep(0.05)
        self.lead_in_spin.setValue(0.15)
        trim_row.addWidget(self.lead_in_spin)
        self._elevate_height(self.lead_in_spin)

        self.trailing_spin = QtWidgets.QDoubleSpinBox()
        self.trailing_spin.setPrefix("Trailing ")
        self.trailing_spin.setSuffix(" s")
        self.trailing_spin.setRange(0.0, 5.0)
        self.trailing_spin.setSingleStep(0.05)
        self.trailing_spin.setValue(0.35)
        trim_row.addWidget(self.trailing_spin)
        self._elevate_height(self.trailing_spin)
        trim_row.addStretch(1)
        panel_layout.addLayout(trim_row)
        self._control_widgets.extend([self.trim_silence_check, self.lead_in_spin, self.trailing_spin])

        self.progress_label = QtWidgets.QLabel("Ready.")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("color: palette(mid);")
        panel_layout.addWidget(self.progress_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.preview_button = QtWidgets.QPushButton("Preview (coming soon)")
        self.preview_button.setEnabled(False)
        self.apply_button = QtWidgets.QPushButton("Apply cleanup")
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.apply_button)
        panel_layout.addLayout(button_row)

        self.preview_button.clicked.connect(self._show_preview_placeholder)
        self.apply_button.clicked.connect(self._emit_process_request)

        self.method_combo.currentIndexChanged.connect(self._on_option_changed)
        self.noise_mode_combo.currentIndexChanged.connect(self._on_noise_mode_changed)
        self.noise_floor_spin.valueChanged.connect(lambda _: self._on_option_changed())
        self.percentile_spin.valueChanged.connect(lambda _: self._on_option_changed())
        self.margin_spin.valueChanged.connect(lambda _: self._on_option_changed())
        self.trim_silence_check.toggled.connect(lambda _: self._on_option_changed())
        self.lead_in_spin.valueChanged.connect(lambda _: self._on_option_changed())
        self.trailing_spin.valueChanged.connect(lambda _: self._on_option_changed())

        panel.set_layout(panel_layout)
        return panel

    def _build_output_panel(self) -> PanelGroup:
        panel = PanelGroup("Output handling")
        panel_layout = QtWidgets.QVBoxLayout()
        panel_layout.setSpacing(12)

        self.copy_radio = QtWidgets.QRadioButton("Write cleaned copy (append suffix)")
        self.overwrite_radio = QtWidgets.QRadioButton("Overwrite original files")
        panel_layout.addWidget(self.copy_radio)
        panel_layout.addWidget(self.overwrite_radio)
        self._control_widgets.extend([self.copy_radio, self.overwrite_radio])

        suffix_row = QtWidgets.QHBoxLayout()
        suffix_row.addWidget(QtWidgets.QLabel("Suffix for cleaned copies:"))
        self.suffix_entry = QtWidgets.QLineEdit("-cleaned")
        suffix_row.addWidget(self.suffix_entry)
        panel_layout.addLayout(suffix_row)
        self._control_widgets.append(self.suffix_entry)

        self.copy_radio.toggled.connect(self._on_output_mode_changed)
        self.overwrite_radio.toggled.connect(self._on_output_mode_changed)
        self.suffix_entry.textChanged.connect(lambda _: self._on_option_changed())

        panel.set_layout(panel_layout)
        return panel

    def _build_results_panel(self) -> PanelGroup:
        panel = PanelGroup("Processed files")
        panel_layout = QtWidgets.QVBoxLayout()
        panel_layout.setSpacing(12)

        self.results_table = QtWidgets.QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Input", "Output", "Duration (s)", "Retained %", "Size Δ (bytes)"]
        )
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.results_table.setMinimumHeight(200)
        row_height = int(self.fontMetrics().height() * 1.6)
        vheader = self.results_table.verticalHeader()
        vheader.setDefaultSectionSize(row_height)
        vheader.setMinimumSectionSize(row_height)
        self.results_table.setStyleSheet("QTableWidget::item { padding: 4px 6px; }")
        panel_layout.addWidget(self.results_table, stretch=1)

        self.summary_label = QtWidgets.QLabel("Run a cleanup to populate results.")
        self.summary_label.setWordWrap(True)
        panel_layout.addWidget(self.summary_label)

        panel.set_layout(panel_layout)
        return panel

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update_recent_capture(self, *, selected_path: Path | None, output_dir: Path | None) -> None:
        self._recent_capture_path = selected_path
        self._recent_output_dir = output_dir
        self._update_recent_labels()
        if self.use_recent_radio.isChecked():
            self.path_entry.setText(self._recent_output_text())
        self._refresh_enablement()

    def set_processing(self, running: bool) -> None:
        self._processing = running
        if running:
            self.results_table.setRowCount(0)
            self.summary_label.setText("Processing…")
            self.progress_label.setText("Processing…")
        else:
            if self.results_table.rowCount() == 0:
                self.progress_label.setText("Ready.")
        self._refresh_enablement()

    def update_progress(self, completed: float, total: float) -> None:
        if total <= 0:
            self.progress_label.setText("Processing…")
            return
        completed = max(0.0, min(completed, total))
        ratio = completed / total if total else 0.0
        self.progress_label.setText(
            f"Processing {int(round(completed))}/{int(total)} file(s) — {ratio * 100.0:4.1f}%"
        )

    def display_summary(self, summary: SquelchSummary) -> None:
        self._processing = False
        self._refresh_enablement()

        self.results_table.setRowCount(len(summary.results))
        for row, item in enumerate(summary.results):
            self._set_table_item(row, 0, item.input_path.name, tooltip=item.input_path.as_posix())
            self._set_table_item(row, 1, item.output_path.name, tooltip=item.output_path.as_posix())
            self._set_table_item(row, 2, f"{item.duration_in:.2f} → {item.duration_out:.2f}")
            self._set_table_item(row, 3, f"{item.retained_ratio * 100.0:4.1f}")
            size_delta = item.bytes_out - item.bytes_in
            self._set_table_item(row, 4, f"{size_delta:+d}")
        self.results_table.resizeColumnsToContents()

        if summary.processed:
            size_delta = summary.aggregate_size_delta()
            duration_delta = summary.aggregate_duration_delta()
            self.summary_label.setText(
                f"Processed {summary.processed} file(s); size Δ {size_delta:+d} bytes, "
                f"duration Δ {duration_delta:+0.2f} s."
            )
            self.progress_label.setText("Audio post-processing complete.")
        else:
            self.summary_label.setText("No files were processed.")
            self.progress_label.setText("No files processed.")

        if summary.errors:
            messages = "\n".join(f"{path.name}: {exc}" for path, exc in summary.errors)
            QtWidgets.QMessageBox.warning(
                self,
                "Audio post-processing errors",
                f"Failed to process {summary.failed} file(s):\n{messages}",
            )

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def _emit_process_request(self) -> None:
        collected = self._collect_options()
        if collected is None:
            return
        target_path, options = collected
        self.set_processing(True)
        self.process_requested.emit(target_path, options)

    def _collect_options(self) -> tuple[Path, AudioPostOptions] | None:
        target_path = self._selected_target_path()
        if target_path is None:
            QtWidgets.QMessageBox.warning(self, "Audio post-processing", "Select a file or directory to process.")
            return None
        if not target_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Audio post-processing",
                f"{target_path.as_posix()} does not exist.",
            )
            return None

        method = self.method_combo.currentData()
        manual_noise = self.noise_mode_combo.currentIndex() == 1 and method != "webrtc"
        manual_value = float(self.noise_floor_spin.value())
        config = SquelchConfig(
            method=method,
            auto_noise_floor=not manual_noise,
            manual_noise_floor_db=manual_value,
            noise_floor_percentile=float(self.percentile_spin.value()),
            threshold_margin_db=float(self.margin_spin.value()),
            trim_silence=self.trim_silence_check.isChecked(),
            trim_lead_seconds=float(self.lead_in_spin.value()),
            trim_trail_seconds=float(self.trailing_spin.value()),
            webrtc_mode=int(self.webrtc_mode_combo.currentData()),
            webrtc_frame_ms=int(self.webrtc_frame_combo.currentData()),
        )
        suffix_text = self.suffix_entry.text().strip() or "-cleaned"
        options = AudioPostOptions(
            config=config,
            overwrite=self.overwrite_radio.isChecked(),
            cleaned_suffix=suffix_text,
        )
        self._sync_state_from_controls(options)
        return target_path, options

    def _selected_target_path(self) -> Path | None:
        if self.use_recent_radio.isChecked():
            text = self._recent_output_text()
        else:
            text = self.path_entry.text().strip()
        if not text:
            return None
        return Path(text)

    def _on_selection_mode_changed(self, button: QtWidgets.QAbstractButton, checked: bool) -> None:
        if not checked:
            return
        if button is self.use_recent_radio:
            self.path_entry.setText(self._recent_output_text())
        else:
            self.path_entry.clear()
        self._refresh_enablement()

    def _on_choose_file(self) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(["Audio files (*.wav *.flac *.ogg *.mp3)", "All files (*)"])
        default_dir = self._recent_output_dir or (self._recent_capture_path.parent if self._recent_capture_path else None)
        if default_dir:
            dialog.setDirectory(default_dir.as_posix())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        self.use_specific_file_radio.setChecked(True)
        self.path_entry.setText(selected[0])
        self._refresh_enablement()

    def _on_choose_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select processed audio directory",
            self._recent_output_text(),
        )
        if not directory:
            return
        self.use_directory_radio.setChecked(True)
        self.path_entry.setText(directory)
        self._refresh_enablement()

    def _on_noise_mode_changed(self) -> None:
        self._on_option_changed()

    def _on_output_mode_changed(self) -> None:
        self._on_option_changed()

    def _on_option_changed(self) -> None:
        self._sync_state_from_controls()
        self._refresh_enablement()

    def _show_preview_placeholder(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Preview not yet available",
            "Audio previews will be added in a future update.",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_state_from_controls(self, options: AudioPostOptions | None = None) -> None:
        method = self.method_combo.currentData()
        if isinstance(method, str):
            self.state.audio_post_method = method
        auto_noise = self.noise_mode_combo.currentIndex() == 0 or method == "webrtc"
        self.state.audio_post_auto_noise = auto_noise
        self.state.audio_post_manual_noise_floor = float(self.noise_floor_spin.value())
        self.state.audio_post_percentile = float(self.percentile_spin.value())
        self.state.audio_post_threshold_db = float(self.margin_spin.value())
        self.state.audio_post_trim = self.trim_silence_check.isChecked()
        self.state.audio_post_lead = float(self.lead_in_spin.value())
        self.state.audio_post_trail = float(self.trailing_spin.value())
        self.state.audio_post_webrtc_mode = int(self.webrtc_mode_combo.currentData())
        self.state.audio_post_webrtc_frame_ms = int(self.webrtc_frame_combo.currentData())
        overwrite = self.overwrite_radio.isChecked()
        self.state.audio_post_overwrite = overwrite
        suffix_text = self.suffix_entry.text().strip() or "-cleaned"
        self.state.audio_post_suffix = suffix_text
        if options is not None:
            options.cleaned_suffix = suffix_text

    def _apply_state_defaults(self) -> None:
        method = self.state.audio_post_method
        index = self.method_combo.findData(method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        if self.state.audio_post_manual_noise_floor is not None:
            self.noise_floor_spin.setValue(self.state.audio_post_manual_noise_floor)

        if self.state.audio_post_auto_noise:
            self.noise_mode_combo.setCurrentIndex(0)
        else:
            self.noise_mode_combo.setCurrentIndex(1)
        self.percentile_spin.setValue(self.state.audio_post_percentile)
        self.margin_spin.setValue(self.state.audio_post_threshold_db)
        self.trim_silence_check.setChecked(self.state.audio_post_trim)
        self.lead_in_spin.setValue(self.state.audio_post_lead)
        self.trailing_spin.setValue(self.state.audio_post_trail)
        mode_index = self.webrtc_mode_combo.findData(self.state.audio_post_webrtc_mode)
        if mode_index >= 0:
            self.webrtc_mode_combo.setCurrentIndex(mode_index)
        frame_index = self.webrtc_frame_combo.findData(self.state.audio_post_webrtc_frame_ms)
        if frame_index >= 0:
            self.webrtc_frame_combo.setCurrentIndex(frame_index)
        if self.state.audio_post_overwrite:
            self.overwrite_radio.setChecked(True)
        else:
            self.copy_radio.setChecked(True)
        suffix = self.state.audio_post_suffix or "-cleaned"
        self.suffix_entry.setText(suffix)

    def _update_recent_labels(self) -> None:
        if self._recent_output_dir:
            directory_text = self._recent_output_dir.as_posix()
            message = f"Latest capture outputs will be read from: {directory_text}"
        elif self._recent_capture_path:
            directory_text = self._recent_capture_path.parent.as_posix()
            message = (
                "Latest capture has not produced outputs yet. "
                f"Defaulting to recording directory: {directory_text}"
            )
        else:
            message = "Run a capture or choose files manually to begin cleanup."
        self.recent_capture_label.setText(message)

    def _recent_output_text(self) -> str:
        if self._recent_output_dir:
            return self._recent_output_dir.as_posix()
        if self._recent_capture_path:
            return self._recent_capture_path.parent.as_posix()
        return ""

    def _refresh_enablement(self) -> None:
        allow = not self._processing
        manual_selection = self.use_specific_file_radio.isChecked() or self.use_directory_radio.isChecked()
        self.use_recent_radio.setEnabled(allow)
        self.use_specific_file_radio.setEnabled(allow)
        self.use_directory_radio.setEnabled(allow)
        self.path_entry.setEnabled(allow and manual_selection)
        self.choose_file_button.setEnabled(allow and self.use_specific_file_radio.isChecked())
        self.choose_directory_button.setEnabled(allow and self.use_directory_radio.isChecked())

        method = self.method_combo.currentData()
        webrtc_active = method == "webrtc"
        manual_noise = self.noise_mode_combo.currentIndex() == 1 and not webrtc_active
        if webrtc_active and self.noise_mode_combo.currentIndex() != 0:
            self.noise_mode_combo.blockSignals(True)
            self.noise_mode_combo.setCurrentIndex(0)
            self.noise_mode_combo.blockSignals(False)
        self.noise_mode_combo.setEnabled(allow and not webrtc_active)
        self.noise_floor_spin.setEnabled(allow and manual_noise)
        self.percentile_spin.setEnabled(allow and not manual_noise and not webrtc_active)
        for widget in (
            self.method_combo,
            self.margin_spin,
            self.trim_silence_check,
            self.lead_in_spin,
            self.trailing_spin,
            self.copy_radio,
            self.overwrite_radio,
            self.suffix_entry,
        ):
            widget.setEnabled(allow)
        self.margin_spin.setEnabled(allow and not webrtc_active)
        self.webrtc_row_widget.setVisible(webrtc_active)
        self.webrtc_mode_combo.setEnabled(allow and webrtc_active)
        self.webrtc_frame_combo.setEnabled(allow and webrtc_active)
        self.suffix_entry.setEnabled(allow and not self.overwrite_radio.isChecked())
        self.apply_button.setEnabled(allow)
        self.preview_button.setEnabled(False)

    def _set_table_item(self, row: int, column: int, text: str, *, tooltip: str | None = None) -> None:
        item = QtWidgets.QTableWidgetItem(text)
        if tooltip:
            item.setToolTip(tooltip)
        self.results_table.setItem(row, column, item)

    def _elevate_height(self, widget: QtWidgets.QWidget) -> None:
        metrics = self.fontMetrics()
        height = max(int(metrics.height() * 1.6), widget.sizeHint().height() + 4)
        widget.setMinimumHeight(height)
        if isinstance(widget, (QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)


class DigitalPostPage(QtWidgets.QWidget):
    """Placeholder UI for digital decoder hand-off."""

    prepare_requested = Signal(str, Path)

    DECODER_OPTIONS: list[tuple[str, str, str]] = [
        (
            "dsd-fme",
            "DSD-FME — Digital voice decoding",
            "Configure piping voice channels to the DSD-FME toolkit.",
        ),
        (
            "multimon-ng",
            "Multimon-NG — Packet data decoding",
            "Batch process paging/data bursts via multimon-ng.",
        ),
        (
            "ft8",
            "FT8/FT4 — Weak signal workflows",
            "Stage audio bursts for FT8 or FT4 decoders.",
        ),
    ]

    def __init__(self, *, state: InteractiveState) -> None:
        super().__init__()
        self.state = state
        self._recent_output_dir: Path | None = state.resolved_output_dir()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(self._build_source_panel())
        layout.addWidget(self._build_decoder_panel())
        layout.addWidget(self._build_options_panel())

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.prepare_button = QtWidgets.QPushButton("Prepare transfer")
        self.prepare_button.clicked.connect(self._show_placeholder_message)
        button_row.addWidget(self.prepare_button)
        layout.addLayout(button_row)

        layout.addStretch(1)
        self._update_source_hint()
        self._on_tool_changed(0)

    def _build_source_panel(self) -> PanelGroup:
        panel = PanelGroup("Source recordings")
        panel_layout = QtWidgets.QVBoxLayout()
        panel_layout.setSpacing(12)

        self.source_hint_label = QtWidgets.QLabel("")
        self.source_hint_label.setWordWrap(True)
        panel_layout.addWidget(self.source_hint_label)

        path_row = QtWidgets.QHBoxLayout()
        self.source_path_entry = QtWidgets.QLineEdit()
        self.source_path_entry.setPlaceholderText("Choose directory that contains demodulated exports.")
        path_row.addWidget(self.source_path_entry, stretch=1)
        self.source_browse_button = QtWidgets.QPushButton("Browse Folder…")
        path_row.addWidget(self.source_browse_button)
        panel_layout.addLayout(path_row)

        note_label = QtWidgets.QLabel(
            "Digital workflows operate on one destination at a time. "
            "Select a single decoder target to continue."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: palette(mid);")
        panel_layout.addWidget(note_label)

        self.source_browse_button.clicked.connect(self._on_browse_source)
        panel.set_layout(panel_layout)
        return panel

    def _build_decoder_panel(self) -> PanelGroup:
        panel = PanelGroup("Decoder routing")
        panel_layout = QtWidgets.QVBoxLayout()
        panel_layout.setSpacing(12)

        self.decoder_combo = QtWidgets.QComboBox()
        for key, label, _description in self.DECODER_OPTIONS:
            self.decoder_combo.addItem(label, userData=key)
        panel_layout.addWidget(self.decoder_combo)

        self.decoder_description_label = QtWidgets.QLabel("")
        self.decoder_description_label.setWordWrap(True)
        panel_layout.addWidget(self.decoder_description_label)

        self.decoder_combo.currentIndexChanged.connect(self._on_tool_changed)

        panel.set_layout(panel_layout)
        return panel

    def _build_options_panel(self) -> PanelGroup:
        panel = PanelGroup("Tool-specific options")
        panel_layout = QtWidgets.QVBoxLayout()
        panel_layout.setSpacing(12)

        self.tool_options_stack = QtWidgets.QStackedWidget()
        self.tool_options_stack.addWidget(self._build_dsd_fme_options())
        self.tool_options_stack.addWidget(self._build_multimon_options())
        self.tool_options_stack.addWidget(self._build_ft8_options())
        panel_layout.addWidget(self.tool_options_stack)

        panel.set_layout(panel_layout)
        return panel

    def _build_dsd_fme_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.dsd_input_combo = QtWidgets.QComboBox()
        self.dsd_input_combo.addItems(["Auto detect", "P25", "DMR", "NXDN", "YSF"])
        layout.addRow("Input protocol:", self.dsd_input_combo)

        self.dsd_channel_mode_combo = QtWidgets.QComboBox()
        self.dsd_channel_mode_combo.addItems(["Single talkgroup", "Follow trunking control", "Manual slot assignment"])
        layout.addRow("Channel mode:", self.dsd_channel_mode_combo)

        self.dsd_talkgroup_entry = QtWidgets.QLineEdit()
        self.dsd_talkgroup_entry.setPlaceholderText("Optional: limit to talkgroup ID")
        layout.addRow("Talkgroup filter:", self.dsd_talkgroup_entry)

        self.dsd_record_check = QtWidgets.QCheckBox("Capture decoded voice audio")
        layout.addRow("", self.dsd_record_check)

        self.dsd_metadata_check = QtWidgets.QCheckBox("Generate metadata JSON for each call")
        layout.addRow("", self.dsd_metadata_check)
        return widget

    def _build_multimon_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.multimon_mode_combo = QtWidgets.QComboBox()
        self.multimon_mode_combo.addItems(["POCSAG1200", "POCSAG2400", "FLEX", "ACARS", "APRS"])
        layout.addRow("Decoder mode:", self.multimon_mode_combo)

        self.multimon_threshold_spin = QtWidgets.QSpinBox()
        self.multimon_threshold_spin.setRange(1, 10)
        self.multimon_threshold_spin.setValue(5)
        layout.addRow("Confidence level:", self.multimon_threshold_spin)

        self.multimon_log_check = QtWidgets.QCheckBox("Write decoded packets to CSV log")
        layout.addRow("", self.multimon_log_check)

        self.multimon_audio_check = QtWidgets.QCheckBox("Store discriminator audio for review")
        layout.addRow("", self.multimon_audio_check)
        return widget

    def _build_ft8_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.ft_mode_combo = QtWidgets.QComboBox()
        self.ft_mode_combo.addItems(["FT8", "FT4"])
        layout.addRow("Mode:", self.ft_mode_combo)

        self.ft_submode_combo = QtWidgets.QComboBox()
        self.ft_submode_combo.addItems(["Default profile", "Contest profile", "Custom offsets"])
        layout.addRow("Profile:", self.ft_submode_combo)

        self.ft_iteration_spin = QtWidgets.QSpinBox()
        self.ft_iteration_spin.setRange(1, 5)
        self.ft_iteration_spin.setValue(2)
        layout.addRow("Decode iterations:", self.ft_iteration_spin)

        self.ft_auto_sync_check = QtWidgets.QCheckBox("Attempt automatic frequency/time sync")
        self.ft_auto_sync_check.setChecked(True)
        layout.addRow("", self.ft_auto_sync_check)

        self.ft_summary_check = QtWidgets.QCheckBox("Summarize decoded messages to JSON")
        layout.addRow("", self.ft_summary_check)
        return widget

    def update_recent_capture(self, *, output_dir: Path | None) -> None:
        self._recent_output_dir = output_dir
        self._update_source_hint()
        if output_dir:
            self.source_path_entry.setPlaceholderText(output_dir.as_posix())

    def _update_source_hint(self) -> None:
        if self._recent_output_dir:
            hint = f"Defaulting to latest capture outputs: {self._recent_output_dir.as_posix()}"
        else:
            hint = "Select the directory that contains channelized audio to send to external decoders."
        self.source_hint_label.setText(hint)

    def _on_browse_source(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select directory for digital post-processing",
            str(self._recent_output_dir) if self._recent_output_dir else "",
        )
        if not directory:
            return
        path = Path(directory)
        self.source_path_entry.setText(path.as_posix())

    def _on_tool_changed(self, index: int) -> None:
        self.tool_options_stack.setCurrentIndex(index)
        if 0 <= index < len(self.DECODER_OPTIONS):
            _key, _label, description = self.DECODER_OPTIONS[index]
            self.decoder_description_label.setText(description)

    def _show_placeholder_message(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Coming soon",
            "Decoder integrations will be wired up in a future release.",
        )
