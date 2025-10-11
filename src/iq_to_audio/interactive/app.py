from __future__ import annotations

import contextlib
import logging
import math
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QWidget,
)

from ..probe import probe_sample_rate
from ..processing import ProcessingCancelled, ProcessingConfig
from ..progress import ProgressSink
from ..utils import detect_center_frequency
from ..visualize import SelectionResult, ensure_matplotlib
from .models import (
    InteractiveOutcome,
    InteractiveSessionResult,
    SnapshotData,
    StatusProgressSink,
)
from .panels import (
    ChannelPanel,
    RecordingPanel,
    SpectrumOptionsPanel,
    StatusPanel,
    TargetsPanel,
    WaterfallOptionsPanel,
)
from .state import COLOR_THEMES, InteractiveState
from .widgets import PanelGroup, SpanController, WaterfallWindow
from .workers import PreviewWorker, SnapshotJob, SnapshotWorker

LOG = logging.getLogger(__name__)

QT_DEPENDENCY_HINT = (
    "PySide6 is required for --interactive. Install it via `uv pip install PySide6 PySide6-Addons`."
)


class _SigintRelay:
    """Bridge SIGINT (Ctrl+C) into a graceful Qt application quit."""

    def __init__(
        self,
        app: QApplication,
        previous_handler,
        *,
        schedule_quit: Callable[[], None] | None = None,
        escalate: Callable[[int, object | None], None] | None = None,
    ) -> None:
        self._app = app
        self._previous_handler = previous_handler if previous_handler is not None else signal.SIG_DFL
        self._schedule_quit = schedule_quit or (lambda: QtCore.QTimer.singleShot(0, app.quit))
        self._escalate = escalate or self._default_escalate
        self._triggered = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self)

    def restore(self) -> None:
        signal.signal(signal.SIGINT, self._previous_handler)

    def __call__(self, signum: int, frame) -> None:
        if self._triggered:
            self._escalate(signum, frame)
            return
        self._triggered = True
        LOG.info("SIGINT received; requesting interactive session shutdown.")
        self._schedule_quit()

    def _default_escalate(self, signum: int, frame) -> None:
        prev = self._previous_handler
        if callable(prev):
            prev(signum, frame)
            return
        with contextlib.suppress(Exception):
            signal.signal(signum, signal.SIG_DFL)
        try:
            os.kill(os.getpid(), signum)
        except Exception as exc:
            raise KeyboardInterrupt() from exc


class InteractiveWindow(QMainWindow):
    """PySide6-based interactive spectrum viewer for IQ to Audio."""

    status_update_signal = Signal(str, bool)
    status_progress_signal = Signal(float)
    _PROCESSING_FIELDS = {
        "target_freq",
        "bandwidth",
        "center_freq",
        "center_freq_source",
        "demod_mode",
        "fs_ch_target",
        "deemph_us",
        "agc_enabled",
        "output_path",
        "dump_iq_path",
        "chunk_size",
        "filter_block",
        "iq_order",
        "probe_only",
        "mix_sign_override",
        "plot_stages_path",
        "fft_workers",
        "max_input_seconds",
    }

    def __init__(
        self,
        *,
        base_kwargs: dict[str, Any],
        initial_path: Path | None,
        snapshot_seconds: float,
    ):
        super().__init__()
        self.setObjectName("InteractiveWindow")

        self.state = InteractiveState(dict(base_kwargs), snapshot_seconds)
        if initial_path is not None:
            self.state.selected_path = initial_path
            self.state.output_hint = self._default_output_hint(initial_path)

        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._active_snapshot_worker: SnapshotWorker | None = None
        self._active_preview_worker: PreviewWorker | None = None
        self._active_pipeline = None
        self._status_sink: StatusProgressSink | None = None
        self.progress_sink: ProgressSink | None = None
        self.result_configs: list[ProcessingConfig] | None = None
        self.waterfall_window: WaterfallWindow | None = None

        self.figure: Figure | None = None
        self.canvas: FigureCanvasQTAgg | None = None
        self.toolbar: NavigationToolbar2QT | None = None
        self.ax_main = None
        self.span_controller: SpanController | None = None

        self.recording_panel: RecordingPanel | None = None
        self.channel_panel: ChannelPanel | None = None
        self.targets_panel: TargetsPanel | None = None
        self.status_panel: StatusPanel | None = None
        self.spectrum_panel: PanelGroup | None = None
        self.spectrum_options_panel: SpectrumOptionsPanel | None = None
        self.waterfall_options_panel: WaterfallOptionsPanel | None = None

        self.toolbar_widget: QtWidgets.QToolBar | None = None
        self.main_splitter: QSplitter | None = None
        self.preview_splitter: QSplitter | None = None

        self.status_update_signal.connect(self._apply_status_message)
        self.status_progress_signal.connect(self._apply_progress_ratio)

        self._configure_window()
        self._build_ui()
        self._build_actions()
        self._update_status_controls()
        self._apply_initial_splitter_sizes()
        self._update_output_hint()

        if self.state.selected_path is not None:
            self._schedule_snapshot(auto=True)

    # ------------------------------------------------------------------
    # Window + UI construction
    # ------------------------------------------------------------------

    def _configure_window(self) -> None:
        self.setWindowTitle("IQ to Audio — Interactive Mode")
        self.resize(1480, 900)
        self.setMinimumSize(1280, 760)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("interactiveMainSplitter")
        splitter.setChildrenCollapsible(False)

        options_scroll = QtWidgets.QScrollArea()
        options_scroll.setWidgetResizable(True)
        options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        options_scroll.setMinimumWidth(520)
        options_scroll.setMaximumWidth(560)

        options_container = QWidget()
        options_layout = QtWidgets.QVBoxLayout(options_container)
        options_layout.setContentsMargins(16, 16, 16, 16)
        options_layout.setSpacing(14)

        self.recording_panel = RecordingPanel(self.state)
        self.channel_panel = ChannelPanel(self.state)
        self.targets_panel = TargetsPanel(self.state)
        self.status_panel = StatusPanel()

        options_layout.addWidget(self.recording_panel)
        options_layout.addWidget(self.channel_panel)
        options_layout.addWidget(self.targets_panel)
        options_layout.addStretch(1)
        options_layout.addWidget(self.status_panel)

        options_scroll.setWidget(options_container)
        splitter.addWidget(options_scroll)

        preview_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_splitter.setObjectName("interactivePreviewSplitter")
        preview_splitter.setChildrenCollapsible(False)

        preview_container = QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(14)

        self.spectrum_panel = PanelGroup("Spectrum preview")
        spectrum_layout = QtWidgets.QVBoxLayout()
        spectrum_layout.setContentsMargins(12, 12, 12, 12)
        spectrum_layout.setSpacing(12)
        placeholder = QtWidgets.QLabel("Load a recording to view its spectrum.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setMinimumHeight(420)
        spectrum_layout.addWidget(placeholder)
        self.spectrum_panel.set_layout(spectrum_layout)
        preview_layout.addWidget(self.spectrum_panel, stretch=4)

        controls_container = QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(16, 0, 16, 16)
        controls_layout.setSpacing(14)

        self.spectrum_options_panel = SpectrumOptionsPanel(self.state)
        self.waterfall_options_panel = WaterfallOptionsPanel(self.state)
        controls_layout.addWidget(self.spectrum_options_panel)
        controls_layout.addWidget(self.waterfall_options_panel)
        controls_layout.addStretch(1)

        preview_splitter.addWidget(preview_container)
        preview_splitter.addWidget(controls_container)
        preview_splitter.setStretchFactor(0, 5)
        preview_splitter.setStretchFactor(1, 2)

        splitter.addWidget(preview_splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        self.setCentralWidget(splitter)
        self.main_splitter = splitter
        self.preview_splitter = preview_splitter

        self._connect_panel_signals()

    def _build_actions(self) -> None:
        toolbar = self.addToolBar("Interactive controls")
        toolbar.setObjectName("interactiveControlsToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        open_action = QtGui.QAction("Open…", self)
        open_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._on_browse)

        preview_action = QtGui.QAction("Preview", self)
        preview_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+P"))
        preview_action.triggered.connect(self._on_preview_clicked)

        confirm_action = QtGui.QAction("Confirm && Run", self)
        confirm_action.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        confirm_action.triggered.connect(self._on_confirm_clicked)

        stop_action = QtGui.QAction("Stop preview", self)
        stop_action.setShortcut(QtGui.QKeySequence("Esc"))
        stop_action.triggered.connect(self._on_stop_preview)

        close_action = QtGui.QAction("Close Session", self)
        close_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        close_action.triggered.connect(self._on_close_session)

        toolbar.addAction(open_action)
        toolbar.addAction(preview_action)
        toolbar.addAction(confirm_action)
        toolbar.addAction(stop_action)
        toolbar.addSeparator()
        toolbar.addAction(close_action)

        self.toolbar_widget = toolbar

    def _connect_panel_signals(self) -> None:
        if not self.recording_panel or not self.channel_panel or not self.targets_panel or not self.status_panel:
            return

        rp = self.recording_panel
        rp.file_entry.textChanged.connect(self._on_file_text_changed)
        rp.browse_button.clicked.connect(self._on_browse)
        rp.detect_button.clicked.connect(self._on_detect_center)
        rp.center_entry.editingFinished.connect(self._on_center_manual)
        rp.snapshot_entry.editingFinished.connect(self._on_snapshot_changed)
        rp.full_snapshot_check.stateChanged.connect(
            lambda state: self._on_full_snapshot_toggled(state == Qt.CheckState.Checked)
        )
        rp.load_fft_button.clicked.connect(lambda: self._schedule_snapshot(auto=False))
        rp.output_entry.editingFinished.connect(self._on_output_dir_changed)
        rp.output_browse_button.clicked.connect(self._on_output_dir_browse)
        rp.agc_check.stateChanged.connect(
            lambda state: self._on_agc_toggled(state == Qt.CheckState.Checked)
        )

        cp = self.channel_panel
        cp.bandwidth_entry.editingFinished.connect(self._on_bandwidth_changed)

        tp = self.targets_panel
        for index, entry in enumerate(tp.entries):
            entry.editingFinished.connect(lambda idx=index, field=entry: self._on_target_edit(idx, field.text()))

        sp = self.status_panel
        sp.preview_button.clicked.connect(self._on_preview_clicked)
        sp.confirm_button.clicked.connect(self._on_confirm_clicked)
        sp.stop_button.clicked.connect(self._on_stop_preview)
        sp.cancel_button.clicked.connect(self._on_close_session)

        sop = self.spectrum_options_panel
        sop.nfft_combo.currentTextChanged.connect(self._on_nfft_changed)
        sop.smoothing_spin.valueChanged.connect(self._on_smoothing_changed)
        sop.range_spin.valueChanged.connect(self._on_range_changed)
        sop.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        sop.refresh_button.clicked.connect(lambda: self._schedule_refresh(full=False))
        sop.reset_button.clicked.connect(self._reset_spectrum_defaults)

        wop = self.waterfall_options_panel
        wop.slice_spin.valueChanged.connect(self._on_waterfall_slices_changed)
        wop.floor_spin.valueChanged.connect(self._on_waterfall_floor_changed)
        wop.cmap_combo.currentTextChanged.connect(self._on_waterfall_cmap_changed)

    def _apply_initial_splitter_sizes(self) -> None:
        if self.main_splitter is None or self.preview_splitter is None:
            return
        total_width = max(self.width(), 1280)
        left = min(540, int(total_width * 0.33))
        right = total_width - left
        self.main_splitter.setSizes([left, right])
        total_height = max(self.height(), 720)
        top = int(total_height * 0.65)
        bottom = total_height - top
        self.preview_splitter.setSizes([top, bottom])

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _default_output_hint(self, path: Path) -> str:
        return f"Outputs default beside: {path.parent}"

    def _update_output_hint(self) -> None:
        if not self.recording_panel:
            return
        if self.state.selected_path is None:
            hint = "Select a recording to preview output location."
        else:
            resolved = self.state.resolved_output_dir()
            if resolved is None:
                hint = self._default_output_hint(self.state.selected_path)
            else:
                hint = f"Outputs will be written to: {resolved}"
        self.state.output_hint = hint
        self.recording_panel.output_hint_label.setText(hint)

    def _apply_status_message(self, message: str, highlight: bool) -> None:
        if not self.status_panel:
            return
        color = "red" if highlight else self.palette().color(QtGui.QPalette.ColorRole.WindowText).name()
        self.status_panel.status_label.setText(message)
        self.status_panel.status_label.setStyleSheet(f"color: {color};")
        self.statusBar().showMessage(message)

    def _apply_progress_ratio(self, ratio: float) -> None:
        if not self.status_panel:
            return
        progress = max(0.0, min(ratio, 1.0))
        if progress <= 0.0:
            self.status_panel.progress_bar.setVisible(False)
            self.status_panel.progress_bar.setValue(0)
        else:
            self.status_panel.progress_bar.setVisible(True)
            self.status_panel.progress_bar.setValue(int(progress * 1000.0))

    def _update_status_controls(self) -> None:
        if not self.status_panel:
            return
        preview_ready = self.state.snapshot_data is not None and self._active_preview_worker is None
        self.status_panel.preview_button.setEnabled(preview_ready)
        if self.toolbar_widget:
            for action in self.toolbar_widget.actions():
                if action.text().startswith("Preview"):
                    action.setEnabled(preview_ready)
        self.status_panel.confirm_button.setEnabled(self.state.selection is not None)
        self.status_panel.stop_button.setEnabled(self._active_preview_worker is not None)

    def _set_status(self, message: str, *, error: bool) -> None:
        self.status_update_signal.emit(message, error)

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        dialog = QFileDialog(self)
        dialog.setNameFilter("WAV files (*.wav)")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        files = dialog.selectedFiles()
        if not files:
            return
        path = Path(files[0])
        if self.recording_panel:
            self.recording_panel.file_entry.setText(path.as_posix())
        self._set_selected_path(path)
        self._schedule_snapshot(auto=True)

    def _set_selected_path(self, path: Path | None) -> None:
        self.state.selected_path = path
        if self.recording_panel:
            self.recording_panel.file_entry.setText(path.as_posix() if path else "")
        self._update_output_hint()

    def _on_file_text_changed(self, text: str) -> None:
        if not text:
            self._set_selected_path(None)
            return
        candidate = Path(text).expanduser()
        if candidate.exists():
            self._set_selected_path(candidate)
        else:
            self.state.selected_path = candidate
        self._update_output_hint()

    def _on_detect_center(self) -> None:
        if self.state.selected_path is None:
            self._set_status("Select an input recording before detecting center frequency.", error=True)
            return
        detection = detect_center_frequency(self.state.selected_path)
        if detection.value is None:
            self._set_status("Unable to detect center frequency from file metadata.", error=True)
            return
        self.state.update_center(detection.value, detection.source or "metadata")
        if self.recording_panel:
            self.recording_panel.center_entry.setText(f"{detection.value:.0f}")
            self.recording_panel.center_source_label.setText(
                f"Center source: {self._describe_center_source(self.state.center_source)}"
            )
        self._set_status(
            f"Center frequency detected ({self._describe_center_source(self.state.center_source)}).",
            error=False,
        )

    def _on_center_manual(self) -> None:
        if not self.recording_panel:
            return
        text = self.recording_panel.center_entry.text().strip()
        value = self._parse_float(text)
        if value is None or value <= 0:
            self._set_status("Center frequency must be a positive number.", error=True)
            self.recording_panel.center_entry.setText(
                f"{self.state.center_freq:.0f}" if self.state.center_freq else ""
            )
            return
        self.state.update_center(value, "manual")
        self.recording_panel.center_source_label.setText(
            f"Center source: {self._describe_center_source(self.state.center_source)}"
        )
        self._set_status("Center frequency updated.", error=False)

    def _on_snapshot_changed(self) -> None:
        if not self.recording_panel:
            return
        text = self.recording_panel.snapshot_entry.text().strip()
        value = self._parse_float(text)
        if value is None or value <= 0:
            self.recording_panel.snapshot_entry.setText(f"{self.state.snapshot_seconds:.2f}")
            self._set_status("Snapshot duration must be positive.", error=True)
            return
        self.state.update_snapshot_duration(value)
        self.recording_panel.snapshot_entry.setText(f"{self.state.snapshot_seconds:.2f}")

    def _on_full_snapshot_toggled(self, enabled: bool) -> None:
        self.state.full_snapshot = enabled
        if self.recording_panel:
            self.recording_panel.full_snapshot_check.setChecked(enabled)

    def _on_output_dir_changed(self) -> None:
        if not self.recording_panel:
            return
        text = self.recording_panel.output_entry.text().strip()
        if not text:
            self.state.output_dir = None
            self._set_status("Output directory cleared; using recording directory.", error=False)
        else:
            path = Path(text).expanduser()
            if not path.exists():
                self._set_status(f"Creating output directory {path}", error=False)
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    self._set_status(f"Failed to create directory: {exc}", error=True)
                    return
            if not path.is_dir():
                self._set_status("Output path must be a directory.", error=True)
                return
            self.state.output_dir = path
        if self.recording_panel:
            self.recording_panel.output_entry.setText(
                self.state.output_dir.as_posix() if self.state.output_dir else ""
            )
        self._update_output_hint()

    def _on_output_dir_browse(self) -> None:
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        dirs = dialog.selectedFiles()
        if not dirs:
            return
        directory = Path(dirs[0])
        self.state.output_dir = directory
        if self.recording_panel:
            self.recording_panel.output_entry.setText(directory.as_posix())
        self._update_output_hint()

    def _on_agc_toggled(self, enabled: bool) -> None:
        self.state.set_agc_enabled(enabled)
        if self.recording_panel:
            self.recording_panel.agc_check.setChecked(enabled)

    def _on_bandwidth_changed(self) -> None:
        if not self.channel_panel:
            return
        text = self.channel_panel.bandwidth_entry.text().strip()
        value = self._parse_float(text)
        if value is None or value <= 0:
            self._set_status("Bandwidth must be positive.", error=True)
            if self.state.bandwidth_hz:
                self.channel_panel.bandwidth_entry.setText(f"{self.state.bandwidth_hz:.0f}")
            return
        self.state.set_bandwidth(value)
        self.channel_panel.bandwidth_entry.setText(f"{self.state.bandwidth_hz:.0f}")
        self._set_status("Bandwidth updated.", error=False)
        if self.span_controller:
            self.span_controller.set_selection(
                self.state.selection.center_freq if self.state.selection else value,
                self.state.bandwidth_hz,
            )

    def _on_target_edit(self, index: int, text: str) -> None:
        text = text.strip()
        if index >= len(self.state.target_text):
            return
        self.state.target_text[index] = text
        freqs = []
        for raw in self.state.target_text:
            freq = self._parse_float(raw)
            if freq is not None and freq > 0:
                if any(math.isclose(freq, other, rel_tol=0.0, abs_tol=0.5) for other in freqs):
                    continue
                freqs.append(freq)
        self.state.set_target_frequencies(freqs)
        self._update_offset_label()

    def _on_nfft_changed(self, value: str) -> None:
        nfft = int(self._parse_float(value) or self.state.nfft)
        if nfft <= 0:
            return
        self.state.nfft = nfft

    def _on_smoothing_changed(self, value: int) -> None:
        self.state.smoothing = value
        if self.state.snapshot_data:
            self._render_snapshot(self.state.snapshot_data, remember=False)

    def _on_range_changed(self, value: int) -> None:
        self.state.dynamic_range = value
        if self.state.snapshot_data:
            self._render_snapshot(self.state.snapshot_data, remember=False)

    def _on_theme_changed(self, theme: str) -> None:
        if theme not in COLOR_THEMES:
            return
        self.state.theme = theme
        if self.state.snapshot_data:
            self._render_snapshot(self.state.snapshot_data, remember=False)

    def _on_waterfall_slices_changed(self, value: int) -> None:
        self.state.waterfall_slices = value
        if self.state.snapshot_data:
            self._update_waterfall_display(
                self.state.snapshot_data.sample_rate,
                self.state.snapshot_data.center_freq,
                self.state.snapshot_data.waterfall,
            )

    def _on_waterfall_floor_changed(self, value: int) -> None:
        self.state.waterfall_floor = value
        if self.state.snapshot_data and self.waterfall_window:
            self._update_waterfall_display(
                self.state.snapshot_data.sample_rate,
                self.state.snapshot_data.center_freq,
                self.state.snapshot_data.waterfall,
            )

    def _on_waterfall_cmap_changed(self, cmap: str) -> None:
        self.state.waterfall_cmap = cmap or "magma"
        if self.state.snapshot_data and self.waterfall_window:
            self._update_waterfall_display(
                self.state.snapshot_data.sample_rate,
                self.state.snapshot_data.center_freq,
                self.state.snapshot_data.waterfall,
            )

    def _reset_spectrum_defaults(self) -> None:
        self.state.nfft = 262_144
        self.state.smoothing = 3
        self.state.dynamic_range = 100
        self.state.theme = "contrast"
        if self.spectrum_options_panel:
            self.spectrum_options_panel.nfft_combo.setCurrentText(str(self.state.nfft))
            self.spectrum_options_panel.smoothing_spin.setValue(self.state.smoothing)
            self.spectrum_options_panel.range_spin.setValue(self.state.dynamic_range)
            self.spectrum_options_panel.theme_combo.setCurrentText(self.state.theme)
        if self.state.snapshot_data:
            self._render_snapshot(self.state.snapshot_data, remember=False)

    def _schedule_refresh(self, *, full: bool) -> None:
        if self.state.snapshot_data is None:
            self._set_status("Load an FFT snapshot before refreshing.", error=True)
            return
        self._render_snapshot(self.state.snapshot_data, remember=False)
        self._set_status("Preview refreshed.", error=False)

    # ------------------------------------------------------------------
    # Snapshot handling
    # ------------------------------------------------------------------

    def _schedule_snapshot(self, *, auto: bool) -> None:
        if self.state.selected_path is None:
            if not auto:
                self._set_status("Select an input recording first.", error=True)
            return
        if self._active_snapshot_worker is not None:
            self._set_status("Snapshot already loading; please wait.", error=False)
            return

        snapshot_seconds = self.state.snapshot_seconds
        if not self.state.full_snapshot:
            if snapshot_seconds <= 0:
                self._set_status("Snapshot duration must be positive.", error=True)
                return
        else:
            snapshot_seconds = max(snapshot_seconds, 2.0)

        try:
            config = self._build_preview_config(require_targets=False)
        except Exception as exc:
            LOG.error("Preview setup failed: %s", exc)
            self._set_status(f"Preview setup failed: {exc}", error=True)
            if not auto:
                QMessageBox.critical(self, "Preview failed", str(exc))
            return

        self._set_status(
            "Analyzing entire recording…" if self.state.full_snapshot else f"Gathering {snapshot_seconds:.2f} s snapshot…",
            error=False,
        )
        self.status_panel.progress_bar.setVisible(True)
        self.status_panel.progress_bar.setValue(0)

        job = SnapshotJob(
            config=config,
            seconds=snapshot_seconds,
            nfft=self.state.nfft,
            hop=max(1, self.state.nfft // 4),
            max_slices=self.state.waterfall_slices,
            fft_workers=self._fft_worker_count(),
            max_samples=self.state.max_preview_samples,
            full_capture=self.state.full_snapshot,
        )
        worker = SnapshotWorker(job)

        worker.signals.progress.connect(self._on_snapshot_progress, Qt.ConnectionType.QueuedConnection)
        worker.signals.failed.connect(self._on_snapshot_failed, Qt.ConnectionType.QueuedConnection)
        worker.signals.finished.connect(self._on_snapshot_finished, Qt.ConnectionType.QueuedConnection)

        self._active_snapshot_worker = worker
        self.thread_pool.start(worker)

    def _on_snapshot_progress(self, seconds_done: float, fraction: float) -> None:
        if not self.status_panel:
            return
        self.status_panel.progress_bar.setVisible(True)
        self.status_panel.progress_bar.setValue(int(max(0.0, min(fraction, 1.0)) * 1000.0))
        self._set_status(f"Gathered {seconds_done:.1f} s of preview ({fraction * 100:4.1f}%)…", error=False)

    def _on_snapshot_failed(self, exc: Exception) -> None:
        self._active_snapshot_worker = None
        if self.status_panel:
            self.status_panel.progress_bar.setVisible(False)
        LOG.error("Failed to gather preview: %s", exc)
        self._set_status(f"Preview failed: {exc}", error=True)
        QMessageBox.critical(self, "Preview failed", str(exc))

    def _on_snapshot_finished(self, snapshot: SnapshotData) -> None:
        self._active_snapshot_worker = None
        if self.status_panel:
            self.status_panel.progress_bar.setVisible(False)
        self.state.snapshot_data = snapshot
        self.state.sample_rate = snapshot.sample_rate
        self.state.probe = snapshot.probe
        self.state.center_freq = snapshot.center_freq
        if self.channel_panel:
            self.channel_panel.sample_rate_label.setText(f"Sample rate: {snapshot.sample_rate:,.0f} Hz")
        self._render_snapshot(snapshot, remember=True)
        self._set_status("Snapshot ready. Drag to select a channel.", error=False)
        self._update_status_controls()

    def _render_snapshot(self, snapshot: SnapshotData, *, remember: bool) -> None:
        if self.spectrum_panel is None:
            return
        freqs = snapshot.center_freq + snapshot.freqs
        psd_db = snapshot.psd_db
        if psd_db.size == 0 or freqs.size == 0:
            self._set_status("Snapshot empty; nothing to display.", error=True)
            return

        if remember:
            self.state.snapshot_data = snapshot

        if self.figure is None:
            self.figure = Figure(figsize=(9.0, 5.5))
        if self.canvas is None:
            self.canvas = FigureCanvasQTAgg(self.figure)
        if self.toolbar is None and NavigationToolbar2QT is not None:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setObjectName("spectrumToolbar")

        layout = self.spectrum_panel.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout()
            self.spectrum_panel.set_layout(layout)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if self.toolbar:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        theme = COLOR_THEMES.get(self.state.theme, COLOR_THEMES["contrast"])
        ax.plot(freqs, psd_db, color=theme.get("line", "#1f77b4"))
        ax.set_title(
            "Drag to highlight a channel. Scroll or double-click to zoom. Use Preview/Confirm buttons."
        )
        if FuncFormatter is not None:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x / 1e6:.3f}"))
        ax.set_xlabel("Absolute frequency (MHz)")
        ax.set_ylabel("Power (dBFS/Hz)")
        ax.grid(True, which="both", ls=theme.get("grid", ":"), color=theme.get("grid_color", "#cccccc"))
        ax.set_facecolor(theme.get("bg", "white"))
        self.figure.patch.set_facecolor(theme.get("face", "white"))
        for spine in ax.spines.values():
            spine.set_color(theme.get("fg", "black"))
        ax.tick_params(colors=theme.get("fg", "black"))
        ax.xaxis.label.set_color(theme.get("fg", "black"))
        ax.yaxis.label.set_color(theme.get("fg", "black"))
        ax.title.set_color(theme.get("fg", "black"))

        dynamic = max(20.0, float(self.state.dynamic_range))
        finite_vals = psd_db[np.isfinite(psd_db)]
        peak = float(np.max(finite_vals)) if finite_vals.size else 0.0
        ax.set_ylim(peak - dynamic, peak + 5.0)

        self.ax_main = ax
        self.canvas.draw()

        initial_center = (
            self.state.selection.center_freq
            if self.state.selection
            else (self.state.target_freqs[0] if self.state.target_freqs else snapshot.center_freq)
        )
        if initial_center is None or initial_center <= 0:
            initial_center = snapshot.center_freq
        initial_bw = (
            self.state.selection.bandwidth
            if self.state.selection
            else (self.state.bandwidth_hz or 12_500.0)
        )
        initial_bw = max(initial_bw, 100.0)

        self.span_controller = SpanController(
            ax=ax,
            canvas=self.canvas,
            initial=SelectionResult(initial_center, initial_bw),
            on_change=self._on_span_changed,
        )
        self._on_span_changed(self.span_controller.selection)

        if snapshot.waterfall is not None:
            self._update_waterfall_display(snapshot.sample_rate, snapshot.center_freq, snapshot.waterfall)

        self._update_status_controls()

    def _on_span_changed(self, selection: SelectionResult) -> None:
        self.state.selection = selection
        self.state.set_bandwidth(selection.bandwidth)
        if self.channel_panel:
            self.channel_panel.bandwidth_entry.setText(f"{selection.bandwidth:.0f}")
        self._update_offset_label()
        self._update_status_controls()

    def _update_offset_label(self) -> None:
        if not self.channel_panel:
            return
        if not self.state.center_freq or not self.state.selection:
            self.channel_panel.offset_label.setText("Offset: —")
            return
        offset = self.state.selection.center_freq - self.state.center_freq
        self.channel_panel.offset_label.setText(f"Offset: {offset:+.0f} Hz")

    def _update_waterfall_display(
        self,
        sample_rate: float,
        center_freq: float,
        waterfall: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> None:
        if waterfall is None:
            return
        if self.waterfall_window is None or not self.waterfall_window.alive:
            self.waterfall_window = WaterfallWindow(
                self,
                on_select=self._on_waterfall_pick,
                on_close=self._on_waterfall_closed,
            )
            self.waterfall_window.show()
        freqs, times, matrix = waterfall
        floor = float(self.state.waterfall_floor)
        cmap = self.state.waterfall_cmap or "magma"
        self.waterfall_window.update(
            freqs=freqs,
            times=times,
            matrix=matrix,
            center_freq=center_freq,
            sample_rate=sample_rate,
            floor_db=floor,
            cmap=cmap,
        )

    def _on_waterfall_pick(self, freq_hz: float) -> None:
        if self.span_controller is None:
            return
        bandwidth = self.state.selection.bandwidth if self.state.selection else (self.state.bandwidth_hz or 12_500.0)
        self.span_controller.set_selection(freq_hz, bandwidth)

    def _on_waterfall_closed(self) -> None:
        self.waterfall_window = None

    # ------------------------------------------------------------------
    # Preview handling
    # ------------------------------------------------------------------

    def _on_preview_clicked(self) -> None:
        if self._active_preview_worker is not None:
            self._set_status("Preview already running. Cancel it before starting a new one.", error=True)
            return
        if self.state.selection is None or self.state.selected_path is None:
            self._set_status("Select a frequency span before previewing.", error=True)
            return
        seconds = self.state.snapshot_seconds if self.state.snapshot_seconds > 0 else 2.0
        try:
            configs = self._build_configs(self.state.selected_path, self.state.center_freq)
        except Exception as exc:
            LOG.error("Preview setup failed: %s", exc)
            self._set_status(f"Preview setup failed: {exc}", error=True)
            QMessageBox.critical(self, "Preview failed", str(exc))
            return
        if not configs:
            self._set_status("Enter at least one target frequency before previewing.", error=True)
            return

        for cfg in configs:
            self._ensure_output_directory(cfg.output_path)

        sink = StatusProgressSink(
            lambda message, highlight: self.status_update_signal.emit(message, highlight),
            progress_update=lambda ratio: self.status_progress_signal.emit(ratio),
        )
        self._status_sink = sink
        if len(configs) > 1:
            self._set_status(f"Previewing {seconds:.2f} s… (1/{len(configs)})", error=True)
        else:
            self._set_status(f"Previewing {seconds:.2f} s…", error=True)
        self._update_status_controls()

        worker = PreviewWorker(
            configs,
            seconds,
            progress_sink=sink,
            register_pipeline=self._register_preview_pipeline,
        )
        worker.signals.started.connect(lambda: self._set_preview_enabled(False))
        worker.signals.failed.connect(self._on_preview_failed, Qt.ConnectionType.QueuedConnection)
        worker.signals.finished.connect(self._on_preview_finished, Qt.ConnectionType.QueuedConnection)

        self._active_preview_worker = worker
        self.thread_pool.start(worker)
        self._set_stop_enabled(True)

    def _on_preview_failed(self, exc: Exception) -> None:
        self._active_preview_worker = None
        self._status_sink = None
        self._set_stop_enabled(False)
        self._set_preview_enabled(True)
        if isinstance(exc, ProcessingCancelled):
            self._set_status("Preview cancelled.", error=False)
        else:
            LOG.error("Preview failed: %s", exc)
            self._set_status(f"Preview failed: {exc}", error=True)
            QMessageBox.critical(self, "Preview failed", str(exc))
        self._update_status_controls()

    def _on_preview_finished(self, outputs: list[Path]) -> None:
        self._active_preview_worker = None
        self._status_sink = None
        self._set_stop_enabled(False)
        self._set_preview_enabled(True)
        self._update_status_controls()

        if not outputs:
            self._set_status("Preview complete.", error=False)
            return
        if len(outputs) == 1:
            summary = f"Preview complete — {outputs[0].name}"
            detail = str(outputs[0])
        else:
            summary = f"Preview complete for {len(outputs)} targets."
            detail = "\n".join(f"{idx + 1}. {path}" for idx, path in enumerate(outputs))
        self._set_status(summary, error=False)
        QMessageBox.information(
            self,
            "Preview complete",
            f"Preview audio written to:\n{detail}",
        )

    def _set_preview_enabled(self, enabled: bool) -> None:
        if not self.status_panel:
            return
        self.status_panel.preview_button.setEnabled(enabled and self.state.snapshot_data is not None)
        if self.toolbar_widget:
            for action in self.toolbar_widget.actions():
                if action.text().startswith("Preview"):
                    action.setEnabled(enabled)

    def _set_stop_enabled(self, enabled: bool) -> None:
        if not self.status_panel:
            return
        self.status_panel.stop_button.setEnabled(enabled)
        if self.toolbar_widget:
            for action in self.toolbar_widget.actions():
                if action.text().startswith("Stop preview"):
                    action.setEnabled(enabled)

    def _on_stop_preview(self) -> None:
        if self._active_preview_worker is None:
            return
        self._set_status("Cancelling preview…", error=True)
        if self._status_sink:
            with contextlib.suppress(Exception):
                self._status_sink.cancel()
        self._set_stop_enabled(False)
        self._cancel_active_pipeline()

    def _register_preview_pipeline(self, pipeline) -> None:
        self._active_pipeline = pipeline

    def _cancel_active_pipeline(self) -> None:
        pipeline = self._active_pipeline
        if pipeline is not None:
            try:
                pipeline.cancel()
            except Exception as exc:
                LOG.debug("Failed to cancel pipeline cleanly: %s", exc)

    def _ensure_output_directory(self, output_path: Path | None) -> None:
        if output_path is None:
            return
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOG.warning("Failed to create output directory %s: %s", output_path.parent, exc)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _on_confirm_clicked(self) -> None:
        if self.state.selection is None or self.state.selected_path is None:
            self._set_status("Select a frequency span before confirming.", error=True)
            return
        configs = self._build_configs(self.state.selected_path, self.state.center_freq)
        if not configs:
            self._set_status("Enter at least one target frequency before running DSP.", error=True)
            return
        self.result_configs = configs
        self.progress_sink = self._status_sink
        app = QApplication.instance()
        if app:
            app.quit()

    def _on_close_session(self) -> None:
        self._cancel_active_pipeline()
        app = QApplication.instance()
        if app:
            app.quit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]  # noqa: N802
        self._on_close_session()
        event.accept()

    def run(self) -> InteractiveSessionResult:
        try:
            self.show()
            app = QApplication.instance()
            if app is None:
                raise RuntimeError("QApplication instance missing during run()")
            app.exec()
        finally:
            self._cancel_active_pipeline()
        if not self.result_configs:
            raise KeyboardInterrupt()
        LOG.info(
            "Interactive selection: center %.0f Hz, %d target(s), bandwidth %.0f Hz",
            self.state.center_freq or 0.0,
            len(self.result_configs),
            self.result_configs[0].bandwidth,
        )
        return InteractiveSessionResult(configs=self.result_configs, progress_sink=self.progress_sink)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _build_preview_config(self, *, require_targets: bool) -> ProcessingConfig:
        if self.state.selected_path is None:
            raise ValueError("Select an input recording first.")
        center_value = self.state.center_freq
        if self.recording_panel:
            entered = self._parse_float(self.recording_panel.center_entry.text())
            if entered:
                center_value = entered
        if center_value is None or center_value <= 0:
            raise ValueError("Center frequency must be provided before previewing.")
        bandwidth = self.state.bandwidth_hz or 12_500.0
        target_freqs = self.state.target_freqs or []
        if not target_freqs and require_targets:
            raise ValueError("Enter at least one target frequency before running DSP.")
        target = target_freqs[0] if target_freqs else center_value
        output_path = self._output_path_for_frequency(self.state.selected_path, target, 1)
        kwargs = self._processing_kwargs(
            center_freq=center_value,
            center_freq_source=self.state.center_source,
            bandwidth=bandwidth,
            target_freq=target,
            output_path=output_path,
            agc_enabled=self.state.agc_enabled,
        )
        return ProcessingConfig(in_path=self.state.selected_path, **kwargs)

    def _processing_kwargs(self, **overrides: Any) -> dict[str, Any]:
        merged = dict(self.state.base_kwargs)
        merged.update(overrides)
        filtered: dict[str, Any] = {}
        for key in self._PROCESSING_FIELDS:
            if key not in merged:
                continue
            value = merged[key]
            if value is None:
                continue
            filtered[key] = value
        return filtered

    def _build_configs(self, path: Path, center_override: float | None) -> list[ProcessingConfig]:
        kwargs = dict(self.state.base_kwargs)
        kwargs["agc_enabled"] = self.state.agc_enabled
        center_value = self.state.center_freq
        center_value = center_override
        if center_value is None or center_value <= 0:
            center_value = self.state.center_freq
        if self.recording_panel:
            manual = self._parse_float(self.recording_panel.center_entry.text())
            if manual:
                center_value = manual
        if center_value is None or center_value <= 0:
            raise ValueError("Center frequency must be provided before running DSP.")
        self.state.update_center(center_value, self.state.center_source)
        kwargs["center_freq"] = center_value
        kwargs["center_freq_source"] = self.state.center_source

        bandwidth = self.state.bandwidth_hz or (self.state.selection.bandwidth if self.state.selection else 12_500.0)
        kwargs["bandwidth"] = max(bandwidth, 100.0)
        self.state.set_bandwidth(kwargs["bandwidth"])

        frequencies = self.state.target_freqs
        if not frequencies:
            fallback = self.state.selection.center_freq if self.state.selection else center_value
            if fallback and fallback > 0:
                frequencies = [float(fallback)]
                if self.targets_panel:
                    entry = self.targets_panel.entries[0]
                    entry.blockSignals(True)
                    entry.setText(f"{fallback:.0f}")
                    entry.blockSignals(False)
                self.state.set_target_frequencies(frequencies)
            else:
                raise ValueError("Enter at least one target frequency before running DSP.")

        configs: list[ProcessingConfig] = []
        total = len(frequencies)
        for freq in frequencies:
            freq_kwargs = self._processing_kwargs(
                target_freq=freq,
                bandwidth=kwargs.get("bandwidth", self.state.bandwidth_hz or 12_500.0),
                center_freq=center_value,
                center_freq_source=self.state.center_source,
                output_path=self._output_path_for_frequency(path, freq, total),
                dump_iq_path=self._augment_path(self.state.base_kwargs.get("dump_iq_path"), freq, total),
                plot_stages_path=self._augment_path(self.state.base_kwargs.get("plot_stages_path"), freq, total),
                agc_enabled=self.state.agc_enabled,
            )
            config = ProcessingConfig(in_path=path, **freq_kwargs)
            configs.append(config)

        if configs:
            self.state.base_kwargs["target_freq"] = configs[0].target_freq
            if configs[0].output_path is not None:
                self.state.base_kwargs["output_path"] = configs[0].output_path
            elif "output_path" in self.state.base_kwargs:
                self.state.base_kwargs.pop("output_path")
        self._update_output_hint()
        return configs

    def _output_path_for_frequency(self, input_path: Path, target_freq: float, total: int) -> Path | None:
        base = self._resolve_output_path(input_path, target_freq)
        if total <= 1:
            return base
        if self.state.output_dir is not None:
            return base
        if base is None:
            return None
        freq_tag = int(round(target_freq))
        return base.with_name(f"{base.stem}_{freq_tag}{base.suffix}")

    def _resolve_output_path(self, input_path: Path, target_freq: float) -> Path | None:
        if self.state.output_dir:
            return self.state.output_dir / f"audio_{int(round(target_freq))}_48k.wav"
        default = input_path.with_name(f"audio_{int(round(target_freq))}_48k.wav")
        return default

    @staticmethod
    def _augment_path(path: Path | None, freq: float, total: int) -> Path | None:
        if path is None or total <= 1:
            return path
        freq_tag = int(round(freq))
        return path.with_name(f"{path.stem}_{freq_tag}{path.suffix}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_float(text: str) -> float | None:
        try:
            value = float(text.strip())
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    def _describe_center_source(self, source: str | None) -> str:
        if not source:
            return "—"
        if source == "manual":
            return "manual entry"
        if source == "metadata":
            return "metadata/filename"
        return source

    @staticmethod
    def _fft_worker_count() -> int | None:
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 1:
            return None
        return min(4, cpu_count)


def launch_interactive_session(
    *,
    input_path: Path | None,
    base_kwargs: dict[str, Any],
    snapshot_seconds: float,
) -> InteractiveSessionResult:
    """Launch the PySide6-based interactive GUI for full-session control."""
    ensure_matplotlib()
    if QtWidgets is None:
        raise RuntimeError(QT_DEPENDENCY_HINT)
    if FigureCanvasQTAgg is None or Figure is None:
        raise RuntimeError("matplotlib QtAgg backend is required for --interactive.")

    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication(sys.argv)

    sigint_relay: _SigintRelay | None = None
    try:
        previous_handler = signal.getsignal(signal.SIGINT)
    except Exception:
        previous_handler = None
    else:
        if previous_handler is None:
            previous_handler = signal.SIG_DFL
        try:
            sigint_relay = _SigintRelay(qapp, previous_handler)
            sigint_relay.install()
        except Exception as exc:
            LOG.debug("Unable to install SIGINT handler: %s", exc)
            sigint_relay = None

    app = InteractiveWindow(
        base_kwargs=base_kwargs,
        initial_path=input_path,
        snapshot_seconds=snapshot_seconds,
    )
    try:
        return app.run()
    finally:
        if sigint_relay is not None:
            try:
                sigint_relay.restore()
            except Exception:
                LOG.debug("Unable to restore previous SIGINT handler.")


def interactive_select(config: ProcessingConfig, seconds: float = 2.0) -> InteractiveOutcome:
    """Compatibility wrapper returning the older InteractiveOutcome structure."""
    session = launch_interactive_session(
        input_path=config.in_path,
        base_kwargs={
            "target_freq": config.target_freq,
            "target_freqs": [config.target_freq],
            "bandwidth": config.bandwidth,
            "center_freq": config.center_freq,
            "demod_mode": config.demod_mode,
            "fs_ch_target": config.fs_ch_target,
            "deemph_us": config.deemph_us,
            "agc_enabled": config.agc_enabled,
            "output_path": config.output_path,
            "dump_iq_path": config.dump_iq_path,
            "chunk_size": config.chunk_size,
            "filter_block": config.filter_block,
            "iq_order": config.iq_order,
            "probe_only": config.probe_only,
            "mix_sign_override": config.mix_sign_override,
            "plot_stages_path": config.plot_stages_path,
        },
        snapshot_seconds=seconds,
    )
    final_config = session.config
    probe = probe_sample_rate(final_config.in_path)
    center = final_config.center_freq
    if center is None:
        detection = detect_center_frequency(final_config.in_path)
        center = detection.value if detection.value is not None else final_config.target_freq
    return InteractiveOutcome(
        center_freq=center,
        target_freq=final_config.target_freq,
        bandwidth=final_config.bandwidth,
        probe=probe,
    )
