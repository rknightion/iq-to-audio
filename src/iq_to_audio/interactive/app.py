from __future__ import annotations

import contextlib
import logging
import math
import os

os.environ.setdefault("MPLBACKEND", "QtAgg")  # Prefer QtAgg during runtime and compilation

import signal
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QWidget,
)

from ..input_formats import InputFormatSpec, detect_input_format, list_supported_formats
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
    DemodPanel,
    RecordingPanel,
    SpectrumOptionsPanel,
    StatusPanel,
    TargetsPanel,
    WaterfallOptionsPanel,
)
from .state import COLOR_THEMES, InteractiveState
from .widgets import LockedSplitter, PanelGroup, SpanController, WaterfallWindow
from .workers import PreviewWorker, SnapshotJob, SnapshotWorker

LOG = logging.getLogger(__name__)

QT_DEPENDENCY_HINT = (
    "PySide6 is required for --interactive. Install it via `uv pip install PySide6 PySide6-Addons`."
)

LEFT_PANEL_WIDTH = 640
PREVIEW_PANEL_MIN_WIDTH = 720
CONTROLS_PANEL_MIN_WIDTH = 360
MIN_WINDOW_HEIGHT = 780
MIN_WINDOW_WIDTH = LEFT_PANEL_WIDTH + PREVIEW_PANEL_MIN_WIDTH


@lru_cache(maxsize=1)
def _format_spec_map() -> dict[str, InputFormatSpec]:
    return {spec.key: spec for spec in list_supported_formats()}


def _format_label(key: str) -> str:
    spec = _format_spec_map().get(key)
    return spec.label if spec else key


DEMOD_OPTIONS: list[tuple[str, str, str]] = [
    (
        "nfm",
        "NFM — Narrowband FM",
        "Voice scanners and public safety (≈12.5 kHz).",
    ),
    (
        "am",
        "AM — Amplitude Modulation",
        "Airband and legacy AM broadcasts.",
    ),
    (
        "usb",
        "USB — Upper Sideband",
        "Single-sideband (marine, ham voice).",
    ),
    (
        "lsb",
        "LSB — Lower Sideband",
        "HF single-sideband voice.",
    ),
    (
        "none",
        "No Demodulation (IQ slice)",
        "Write tuned complex IQ using the original format.",
    ),
]

_DEMOD_LOOKUP = {value: (label, description) for value, label, description in DEMOD_OPTIONS}


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
        "input_format",
        "input_container",
        "input_format_source",
        "input_sample_rate",
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
        self._active_pipeline: Any | None = None
        self._status_sink: StatusProgressSink | None = None
        self.progress_sink: ProgressSink | None = None
        self.result_configs: list[ProcessingConfig] | None = None
        self.waterfall_window: WaterfallWindow | None = None

        self.figure: Figure | None = None
        self.canvas: FigureCanvasQTAgg | None = None
        self.toolbar: NavigationToolbar2QT | None = None
        self.ax_main: Axes | None = None
        self.span_controller: SpanController | None = None
        self._hover_cid: int | None = None
        self._press_cid: int | None = None
        self._release_cid: int | None = None
        self._hover_line: Line2D | None = None
        self._hover_text: Text | None = None
        self._selection_text: Text | None = None
        self._hover_theme: dict[str, str] = {}
        self._press_event_data: dict[str, float] | None = None
        self._last_hover_freq: float | None = None
        self._scroll_cid: int | None = None
        self._freq_limits: tuple[float, float] | None = None

        self.recording_panel: RecordingPanel | None = None
        self.channel_panel: ChannelPanel | None = None
        self.demod_panel: DemodPanel | None = None
        self.targets_panel: TargetsPanel | None = None
        self.status_panel: StatusPanel | None = None
        self.spectrum_panel: PanelGroup | None = None
        self.spectrum_options_panel: SpectrumOptionsPanel | None = None
        self.waterfall_options_panel: WaterfallOptionsPanel | None = None

        self.toolbar_widget: QtWidgets.QToolBar | None = None
        self.nav_toolbar: QtWidgets.QToolBar | None = None
        self.page_stack: QStackedWidget | None = None
        self.nav_actions: dict[int, QtGui.QAction] = {}
        self.capture_page_index: int = 0
        self.audio_post_page_index: int = 0
        self.digital_post_page_index: int = 0
        self.audio_post_page: AudioPostPage | None = None
        self.digital_post_page: DigitalPostPage | None = None
        self.main_splitter: QSplitter | None = None
        self.preview_splitter: QSplitter | None = None

        self.status_update_signal.connect(self._apply_status_message)
        self.status_progress_signal.connect(self._apply_progress_ratio)

        self._configure_window()
        self._build_ui()
        self._build_actions()
        self._update_format_options()
        self._refresh_format_status_label()
        self._refresh_demod_description()
        self._apply_demod_constraints()
        self._refresh_sample_rate_label()
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
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

    def _build_ui(self) -> None:
        stack = QStackedWidget()
        stack.setObjectName("interactivePageStack")
        self.page_stack = stack

        capture_page = self._build_capture_page()
        self.capture_page_index = stack.addWidget(capture_page)

        self.audio_post_page = AudioPostPage(state=self.state)
        self.audio_post_page.process_requested.connect(self._on_audio_post_requested)
        self.audio_post_page_index = stack.addWidget(self.audio_post_page)

        self.digital_post_page = DigitalPostPage(state=self.state)
        self.digital_post_page_index = stack.addWidget(self.digital_post_page)

        stack.setCurrentIndex(self.capture_page_index)
        stack.currentChanged.connect(self._on_page_changed)
        self.setCentralWidget(stack)

    def _build_capture_page(self) -> QWidget:
        splitter = LockedSplitter(Qt.Orientation.Horizontal, locked_handles={1})
        splitter.setObjectName("interactiveMainSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)

        options_scroll = QtWidgets.QScrollArea()
        options_scroll.setWidgetResizable(True)
        options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        options_scroll.setMinimumWidth(LEFT_PANEL_WIDTH)
        options_scroll.setMaximumWidth(LEFT_PANEL_WIDTH)
        options_scroll.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        )

        options_container = QWidget()
        options_container.setMinimumWidth(LEFT_PANEL_WIDTH)
        options_container.setMaximumWidth(LEFT_PANEL_WIDTH)
        options_layout = QtWidgets.QVBoxLayout(options_container)
        options_layout.setContentsMargins(16, 16, 16, 16)
        options_layout.setSpacing(14)

        self.recording_panel = RecordingPanel(self.state)
        self.channel_panel = ChannelPanel(self.state)
        self.demod_panel = DemodPanel(self.state, DEMOD_OPTIONS)
        self.targets_panel = TargetsPanel(self.state)
        self.status_panel = StatusPanel()

        for panel in (
            self.recording_panel,
            self.channel_panel,
            self.demod_panel,
            self.targets_panel,
            self.status_panel,
        ):
            if panel is not None:
                panel.setMinimumWidth(LEFT_PANEL_WIDTH - 32)
                panel.setMaximumWidth(LEFT_PANEL_WIDTH - 32)
                panel.setSizePolicy(
                    QtWidgets.QSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Fixed,
                        QtWidgets.QSizePolicy.Policy.Preferred,
                    )
                )

        options_layout.addWidget(self.recording_panel)
        options_layout.addWidget(self.channel_panel)
        options_layout.addWidget(self.demod_panel)
        options_layout.addWidget(self.targets_panel)
        options_layout.addStretch(1)
        options_layout.addWidget(self.status_panel)

        options_scroll.setWidget(options_container)
        splitter.addWidget(options_scroll)

        preview_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_splitter.setObjectName("interactivePreviewSplitter")
        preview_splitter.setChildrenCollapsible(False)

        preview_container = QWidget()
        preview_container.setMinimumWidth(PREVIEW_PANEL_MIN_WIDTH)
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
        controls_container.setMinimumWidth(CONTROLS_PANEL_MIN_WIDTH)
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
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.main_splitter = splitter
        self.preview_splitter = preview_splitter

        self._connect_panel_signals()
        container = QWidget()
        container.setObjectName("interactiveCapturePage")
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(splitter)
        return container

    def _build_navigation(self) -> None:
        if self.page_stack is None:
            return
        nav_toolbar = QtWidgets.QToolBar("Workspace navigation")
        nav_toolbar.setObjectName("interactiveNavigationToolbar")
        nav_toolbar.setMovable(False)
        nav_toolbar.setFloatable(False)
        nav_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, nav_toolbar)
        self.nav_toolbar = nav_toolbar

        self.nav_actions.clear()
        action_group = QtGui.QActionGroup(nav_toolbar)
        action_group.setExclusive(True)

        def add_nav_action(label: str, index: int) -> None:
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action_group.addAction(action)
            nav_toolbar.addAction(action)
            action.triggered.connect(lambda checked, idx=index: self._set_active_page(idx) if checked else None)
            self.nav_actions[index] = action

        add_nav_action("Capture", self.capture_page_index)
        add_nav_action("Audio Post", self.audio_post_page_index)
        add_nav_action("Digital Post", self.digital_post_page_index)

        action = self.nav_actions.get(self.capture_page_index)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self._on_page_changed(self.capture_page_index)

    def _build_actions(self) -> None:
        toolbar = QtWidgets.QToolBar("Interactive controls")
        toolbar.setObjectName("interactiveControlsToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

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
        if (
            not self.recording_panel
            or not self.channel_panel
            or not self.demod_panel
            or not self.targets_panel
            or not self.status_panel
            or not self.spectrum_options_panel
            or not self.waterfall_options_panel
        ):
            return

        assert self.recording_panel is not None
        assert self.channel_panel is not None
        assert self.demod_panel is not None
        assert self.targets_panel is not None
        assert self.status_panel is not None
        assert self.spectrum_options_panel is not None
        assert self.waterfall_options_panel is not None

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
        rp.agc_check.toggled.connect(self._on_agc_toggled)
        rp.format_combo.currentIndexChanged.connect(self._on_format_changed)

        cp = self.channel_panel
        cp.bandwidth_entry.editingFinished.connect(self._on_bandwidth_changed)
        cp.sample_rate_entry.editingFinished.connect(self._on_sample_rate_override)

        dp = self.demod_panel
        dp.mode_combo.currentIndexChanged.connect(self._on_demod_changed)

        tp = self.targets_panel
        for index, entry in enumerate(tp.entries):
            entry.editingFinished.connect(lambda idx=index, field=entry: self._on_target_edit(idx, field.text()))
        tp.clear_button.clicked.connect(self._on_clear_targets)

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

    def _format_auto_text(self) -> str:
        if self.state.detected_format:
            label = _format_label(self.state.detected_format)
            return f"Auto (Detected: {label})"
        if self.state.input_format_error:
            return "Auto (manual selection required)"
        return "Auto (detect from file)"

    def _update_format_options(self) -> None:
        if not self.recording_panel:
            return
        combo = self.recording_panel.format_combo
        order = {"pcm_u8": 0, "pcm_s16le": 1, "pcm_f32le": 2}
        container_order = {"wav": 0, "raw": 1}
        specs = sorted(
            _format_spec_map().values(),
            key=lambda spec: (container_order.get(spec.container, 99), order.get(spec.codec, 99)),
        )
        desired = self.state.input_format_choice if self.state.input_format_choice != "auto" else "auto"
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._format_auto_text(), "auto")
        for spec in specs:
            combo.addItem(spec.label, spec.key)
        index = combo.findData(desired)
        if index < 0:
            index = 0
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _refresh_format_status_label(self) -> None:
        if not self.recording_panel:
            return
        label_text = self.state.input_format_message
        color = "red" if self.state.input_format_error else self.palette().color(QtGui.QPalette.ColorRole.WindowText).name()
        self.recording_panel.format_status_label.setText(label_text)
        self.recording_panel.format_status_label.setStyleSheet(f"color: {color};")

    def _refresh_sample_rate_hint(self) -> None:
        if not self.channel_panel:
            return
        if self.state.sample_rate_override:
            hint = "Manual override active. Raw recordings require a valid sample rate."
        elif self.state.sample_rate and self.state.sample_rate_source:
            hint = f"Detected via {self.state.sample_rate_source}."
        elif self.state.sample_rate:
            hint = "Sample rate detected from metadata."
        else:
            hint = "Provide a sample rate override if auto-detection fails (raw or malformed WAV recordings)."
        self.channel_panel.sample_rate_hint.setText(hint)

    def _refresh_sample_rate_label(self) -> None:
        if not self.channel_panel:
            return
        if self.state.sample_rate_override:
            label = f"Sample rate: {self.state.sample_rate_override:,.0f} Hz (manual override)"
        elif self.state.sample_rate:
            label = f"Sample rate: {self.state.sample_rate:,.0f} Hz"
            if self.state.sample_rate_source:
                label += f" ({self.state.sample_rate_source})"
        else:
            label = "Sample rate: —"
        self.channel_panel.sample_rate_label.setText(label)
        self._refresh_sample_rate_hint()

    def _clear_format_detection(self) -> None:
        self.state.set_detected_format(
            None,
            source="",
            message="Select a recording to detect input format.",
            error=None,
        )
        self._update_format_options()
        self._refresh_format_status_label()

    def _refresh_demod_description(self) -> None:
        if not self.demod_panel:
            return
        combo = self.demod_panel.mode_combo
        current = combo.findData(self.state.demod_mode)
        combo.blockSignals(True)
        if current >= 0 and combo.currentIndex() != current:
            combo.setCurrentIndex(current)
        combo.blockSignals(False)
        description = _DEMOD_LOOKUP.get(self.state.demod_mode, ("", ""))[1]
        self.demod_panel.description_label.setText(description)

    def _apply_demod_constraints(self) -> None:
        if not self.recording_panel:
            return
        checkbox = self.recording_panel.agc_check
        checkbox.blockSignals(True)
        if self.state.demod_mode.lower() == "none":
            self.state.apply_agc_override(False)
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
        else:
            checkbox.setEnabled(True)
            checkbox.setChecked(self.state.agc_enabled)
        checkbox.blockSignals(False)

    def _update_sample_rate_from_probe(self, path: Path) -> None:
        try:
            probe = probe_sample_rate(path)
        except Exception:  # pragma: no cover - defensive fallback
            return
        self.state.probe = probe
        value = None
        source = ""
        if probe.ffprobe:
            value = probe.ffprobe
            source = "ffprobe"
        elif probe.header:
            value = probe.header
            source = "header"
        elif probe.wave:
            value = probe.wave
            source = "wave"
        if value:
            self.state.set_sample_rate_detected(float(value), source)
            self._refresh_sample_rate_label()

    def _auto_detect_format(self, path: Path, *, announce: bool) -> None:
        detection = detect_input_format(path)
        if detection.spec:
            spec = detection.spec
            message = detection.message or f"Detected {spec.label}."
            self.state.set_detected_format(spec.key, source=detection.source, message=message, error=None)
            if spec.container == "wav":
                self._update_sample_rate_from_probe(path)
                if announce:
                    self._set_status(message, error=False)
            else:
                if announce:
                    self._set_status(
                        f"Detected {spec.label}. Provide a sample rate override if metadata is missing.",
                        error=False,
                    )
                if not self.state.sample_rate_override:
                    self._set_status(
                        "Enter the capture sample rate in the Channel panel when metadata is unavailable.",
                        error=True,
                    )
        else:
            error = detection.error or "Unable to detect input format."
            self.state.set_detected_format(
                None,
                source=detection.source,
                message=error,
                error=error,
            )
            if announce:
                self._set_status(error, error=True)
        self._update_format_options()
        self._refresh_format_status_label()
        self._refresh_sample_rate_label()

    def _on_demod_changed(self, index: int) -> None:
        if not self.demod_panel:
            return
        value = self.demod_panel.mode_combo.itemData(index)
        if value is None:
            return
        mode = str(value)
        if mode == self.state.demod_mode:
            self._refresh_demod_description()
            self._apply_demod_constraints()
            return
        previous_pref = self.state.preferred_agc
        self.state.set_demod_mode(mode)
        if mode.lower() == "none":
            if previous_pref is not None:
                self.state.preferred_agc = previous_pref
            self.state.apply_agc_override(False)
            if self.recording_panel:
                checkbox = self.recording_panel.agc_check
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
                checkbox.blockSignals(False)
            self._set_status("No demodulation: tuned IQ slices will be written.", error=False)
        else:
            desired = previous_pref if previous_pref is not None else True
            self.state.set_agc_enabled(desired)
            if self.recording_panel:
                checkbox = self.recording_panel.agc_check
                checkbox.blockSignals(True)
                checkbox.setEnabled(True)
                checkbox.setChecked(self.state.agc_enabled)
                checkbox.blockSignals(False)
        self._refresh_demod_description()
        self._apply_demod_constraints()

    def _on_format_changed(self, index: int) -> None:
        if not self.recording_panel:
            return
        value = self.recording_panel.format_combo.itemData(index)
        if value is None:
            return
        if value == "auto":
            if self.state.input_format_choice != "auto":
                self.state.clear_manual_format()
                if self.state.selected_path and self.state.selected_path.exists():
                    self._auto_detect_format(self.state.selected_path, announce=False)
                else:
                    self._clear_format_detection()
                self._update_format_options()
                self._refresh_format_status_label()
                self._set_status("Input format set to auto detection.", error=False)
            return
        if isinstance(value, str) and ":" in value:
            container, codec = value.split(":", 1)
            self.state.set_manual_format(container, codec)
            label = _format_label(value)
            self.state.input_format_message = f"Manual override: {label}"
            self._set_status(f"Manual input format override set to {label}.", error=False)
            self._update_format_options()
            self._refresh_format_status_label()

    def _on_sample_rate_override(self) -> None:
        if not self.channel_panel:
            return
        text = self.channel_panel.sample_rate_entry.text().strip()
        if not text:
            if self.state.sample_rate_override:
                self.state.set_sample_rate_override(None)
                self._set_status("Sample rate override cleared.", error=False)
                self._refresh_sample_rate_label()
            else:
                self.state.set_sample_rate_override(None)
                self._refresh_sample_rate_label()
            self.channel_panel.sample_rate_entry.setText("")
            return
        value = self._parse_float(text)
        if value is None or value <= 0:
            self._set_status("Sample rate override must be a positive number.", error=True)
            self.channel_panel.sample_rate_entry.setText(
                f"{self.state.sample_rate_override:.0f}" if self.state.sample_rate_override else ""
            )
            return
        self.state.set_sample_rate_override(value)
        self.state.sample_rate = value
        self.channel_panel.sample_rate_entry.setText(f"{value:.0f}")
        self._refresh_sample_rate_label()
        self._set_status("Sample rate override applied.", error=False)

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
        if self.state.input_format_choice == "auto" and self.state.input_format_error:
            preview_ready = False
        self.status_panel.preview_button.setEnabled(preview_ready)
        if self.toolbar_widget:
            for action in self.toolbar_widget.actions():
                if action.text().startswith("Preview"):
                    action.setEnabled(preview_ready)
        self.status_panel.confirm_button.setEnabled(self.state.selection is not None)
        self.status_panel.stop_button.setEnabled(self._active_preview_worker is not None)
        if self.recording_panel:
            detect_enabled = (
                self.state.selected_path is not None
                and self._active_preview_worker is None
                and self._active_pipeline is None
            )
            self.recording_panel.detect_button.setEnabled(detect_enabled)

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
        if path is None:
            self.state.probe = None
            if not self.state.sample_rate_override:
                self.state.sample_rate = None
                self.state.sample_rate_source = ""
            self._clear_format_detection()
            self._refresh_sample_rate_label()
            self._update_status_controls()
            return
        self._auto_detect_format(path, announce=False)
        self._auto_detect_center(path, announce=True, force=True, preserve_manual=False)
        self._update_status_controls()

    def _on_file_text_changed(self, text: str) -> None:
        if not text:
            self._set_selected_path(None)
            return
        candidate = Path(text).expanduser()
        if candidate.exists():
            self._set_selected_path(candidate)
            return
        self.state.selected_path = candidate
        self._update_output_hint()
        self._clear_format_detection()
        if not self.state.sample_rate_override:
            self.state.sample_rate = None
            self.state.sample_rate_source = ""
        self._refresh_sample_rate_label()

    def _on_detect_center(self) -> None:
        if self.state.selected_path is None:
            self._set_status("Select an input recording before detecting center frequency.", error=True)
            return
        if self._active_preview_worker is not None or self._active_pipeline is not None:
            self._set_status("Center detection is unavailable while processing is active.", error=True)
            return
        self._auto_detect_center(self.state.selected_path, announce=True, force=True, preserve_manual=True)

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
        self._set_status("Center frequency updated.", error=False)
        self._refresh_center_source_label()

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
        previous = self.state.agc_enabled
        self.state.set_agc_enabled(enabled)
        if previous == enabled:
            return
        status = "Automatic gain control enabled." if enabled else "Automatic gain control disabled."
        self._set_status(status, error=False)
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
        if self.span_controller and self.state.bandwidth_hz is not None:
            center = (
                self.state.selection.center_freq
                if self.state.selection is not None
                else value
            )
            self.span_controller.set_selection(
                center,
                self.state.bandwidth_hz,
            )

    def _on_target_edit(self, index: int, text: str) -> None:
        if index >= self.state.max_target_freqs:
            return
        self._sync_state_targets_from_entries()
        self._update_offset_label()

    def _sync_state_targets_from_entries(self) -> None:
        if not self.targets_panel:
            return
        texts = [entry.text().strip() for entry in self.targets_panel.entries]
        padded = list(texts)
        if len(padded) < self.state.max_target_freqs:
            padded.extend([""] * (self.state.max_target_freqs - len(padded)))
        self.state.target_text = padded[: self.state.max_target_freqs]
        freqs: list[float] = []
        for raw in texts:
            freq = self._parse_float(raw)
            if freq is None or freq <= 0:
                continue
            if any(abs(freq - other) < 0.5 for other in freqs):
                continue
            freqs.append(freq)
        self.state.set_target_frequencies(freqs)
        self._update_status_controls()

    def _insert_target_frequency(self, freq: float, *, announce: bool = False) -> bool:
        if not self.targets_panel or freq <= 0:
            return False
        rounded = float(round(freq))
        existing = [
            self._parse_float(entry.text()) for entry in self.targets_panel.entries if entry is not None
        ]
        if any(value is not None and abs(value - rounded) < 0.5 for value in existing):
            if announce:
                self._set_status(f"{rounded:.0f} Hz is already in the target list.", error=False)
            return False
        next_index: int | None = None
        for idx, value in enumerate(existing):
            if value is None or value <= 0:
                next_index = idx
                break
        if next_index is None:
            if announce:
                self._set_status("All target slots are filled. Clear targets to add more.", error=True)
            return False
        entry = self.targets_panel.entries[next_index]
        entry.blockSignals(True)
        entry.setText(f"{rounded:.0f}")
        entry.blockSignals(False)
        self._sync_state_targets_from_entries()
        if announce:
            self._set_status(f"Target {next_index + 1} set to {rounded:.0f} Hz.", error=False)
        return True

    def _on_clear_targets(self) -> None:
        if not self.targets_panel:
            return
        for entry in self.targets_panel.entries:
            entry.blockSignals(True)
            entry.clear()
            entry.blockSignals(False)
        self._sync_state_targets_from_entries()
        self._set_status("Cleared all target frequencies.", error=False)

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

        if self.status_panel is None:
            return
        assert self.status_panel is not None

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
        self.state.set_sample_rate_detected(snapshot.sample_rate, "preview")
        self.state.probe = snapshot.probe
        self.state.center_freq = snapshot.center_freq
        self._refresh_sample_rate_label()
        self._render_snapshot(snapshot, remember=True)
        self._set_status("Snapshot ready. Drag to select a channel.", error=False)
        self._update_status_controls()

    def _render_snapshot(self, snapshot: SnapshotData, *, remember: bool) -> None:
        if self.spectrum_panel is None:
            raise RuntimeError("Spectrum panel not initialized.")
        freqs = snapshot.center_freq + snapshot.freqs
        psd_db = snapshot.psd_db
        if psd_db.size == 0 or freqs.size == 0:
            self._set_status("Snapshot empty; nothing to display.", error=True)
            return
        finite_freqs = freqs[np.isfinite(freqs)]
        if finite_freqs.size == 0:
            self._set_status("Snapshot frequencies invalid; nothing to display.", error=True)
            return
        freq_min = float(finite_freqs.min())
        freq_max = float(finite_freqs.max())
        if not math.isfinite(freq_min) or not math.isfinite(freq_max):
            self._set_status("Snapshot frequencies invalid; nothing to display.", error=True)
            return
        self._freq_limits = (freq_min, freq_max)

        if remember:
            self.state.snapshot_data = snapshot

        if self.figure is None:
            self.figure = Figure(figsize=(9.0, 5.5))
        if self.canvas is None:
            self.canvas = FigureCanvasQTAgg(self.figure)
        assert self.figure is not None
        assert self.canvas is not None
        if self.toolbar is None and NavigationToolbar2QT is not None:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setObjectName("spectrumToolbar")

        if self.spectrum_panel is None:
            raise RuntimeError("Spectrum panel not initialized.")
        layout = self.spectrum_panel.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout()
            self.spectrum_panel.set_layout(layout)
        managed_widgets = tuple(widget for widget in (self.toolbar, self.canvas) if widget is not None)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget in managed_widgets:
                widget.setParent(None)
                continue
            widget.deleteLater()
        if self.toolbar:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._teardown_canvas_events()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        theme = COLOR_THEMES.get(self.state.theme, COLOR_THEMES["contrast"])
        ax.plot(freqs, psd_db, color=theme.get("line", "#1f77b4"))
        ax.set_title(
            "Click & Drag to select a channel & BW. Click again to save target frequency. Scroll or double-click to zoom."
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
        ax.set_xlim(freq_min, freq_max)
        self._freq_limits = (freq_min, freq_max)

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
        self._setup_canvas_interactions(ax, theme)
        self._on_span_changed(self.span_controller.selection.as_selection())

        if snapshot.waterfall is not None:
            self._update_waterfall_display(snapshot.sample_rate, snapshot.center_freq, snapshot.waterfall)

        self._update_status_controls()

    def _setup_canvas_interactions(self, ax, theme: dict[str, str]) -> None:
        if self.canvas is None:
            return
        self._hover_theme = {
            "fg": theme.get("fg", "black"),
            "line": theme.get("line", "#1f77b4"),
            "face": theme.get("face", "white"),
        }
        hover_color = self._hover_theme["line"]
        self._hover_line = ax.axvline(ax.get_xlim()[0], color=hover_color, ls="--", lw=1.0, alpha=0.6)
        self._hover_line.set_visible(False)
        bbox_args = {
            "boxstyle": "round,pad=0.3",
            "facecolor": self._hover_theme["face"],
            "edgecolor": "none",
            "alpha": 0.85,
        }
        label_kwargs = {
            "transform": ax.transAxes,
            "fontsize": 10,
            "bbox": bbox_args,
        }
        self._hover_text = ax.text(
            0.02,
            0.96,
            "",
            color=self._hover_theme["fg"],
            ha="left",
            va="top",
            **label_kwargs,
        )
        self._hover_text.set_visible(False)
        self._selection_text = ax.text(
            0.98,
            0.96,
            "",
            color=self._hover_theme["fg"],
            ha="right",
            va="top",
            **label_kwargs,
        )
        self._selection_text.set_visible(True)
        self._last_hover_freq = None
        if self.canvas:
            self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self._press_cid = self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
            self._release_cid = self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
            self._scroll_cid = self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)
        self._update_selection_label()

    def _teardown_canvas_events(self) -> None:
        if self.canvas is not None:
            for cid in (self._hover_cid, self._press_cid, self._release_cid):
                if cid is None:
                    continue
                with contextlib.suppress(Exception):
                    self.canvas.mpl_disconnect(cid)
            if self._scroll_cid is not None:
                with contextlib.suppress(Exception):
                    self.canvas.mpl_disconnect(self._scroll_cid)
        self._hover_cid = None
        self._press_cid = None
        self._release_cid = None
        self._scroll_cid = None
        self._hover_line = None
        self._hover_text = None
        self._selection_text = None
        self._hover_theme = {}
        self._press_event_data = None
        self._last_hover_freq = None

    def _on_canvas_motion(self, event) -> None:
        if event.inaxes != self.ax_main or event.xdata is None:
            self._last_hover_freq = None
            self._set_hover_visible(None)
            return
        freq = float(event.xdata)
        self._last_hover_freq = freq
        self._set_hover_visible(freq)

    def _on_canvas_press(self, event) -> None:
        if event.button != 1 or getattr(event, "dblclick", False):
            self._press_event_data = None
            return
        if event.inaxes != self.ax_main or event.xdata is None:
            self._press_event_data = None
            return
        self._press_event_data = {"freq": float(event.xdata)}

    def _on_canvas_release(self, event) -> None:
        if event.button != 1 or getattr(event, "dblclick", False):
            self._press_event_data = None
            return
        if self._press_event_data is None:
            return
        if event.inaxes != self.ax_main or event.xdata is None:
            self._press_event_data = None
            return
        start_freq = self._press_event_data.get("freq")
        self._press_event_data = None
        if start_freq is None:
            return
        end_freq = float(event.xdata)
        delta = abs(end_freq - start_freq)
        threshold = max(750.0, 0.001 * (self.state.sample_rate or 0.0))
        if delta > threshold:
            return
        self._insert_target_frequency(end_freq, announce=True)

    def _on_canvas_scroll(self, event) -> None:
        if self.ax_main is None or event.inaxes != self.ax_main or event.xdata is None:
            return
        step = float(event.step)
        if step == 0.0:
            return
        direction = 1.0 if step > 0 else -1.0
        magnitude = max(1.0, abs(step))
        base_scale = 1.2
        scale = base_scale ** (0.5 * magnitude)
        cur_xlim = self.ax_main.get_xlim()
        width = cur_xlim[1] - cur_xlim[0]
        if width <= 0:
            return
        if self._freq_limits is None:
            self._freq_limits = (float(cur_xlim[0]), float(cur_xlim[1]))
        new_width = width / scale if direction > 0 else width * scale
        center = float(event.xdata)
        left_fraction = (center - cur_xlim[0]) / width if width else 0.5
        right_fraction = (cur_xlim[1] - center) / width if width else 0.5
        new_left = center - left_fraction * new_width
        new_right = center + right_fraction * new_width
        if self._freq_limits is not None:
            freq_min, freq_max = self._freq_limits
            max_width = freq_max - freq_min
            if max_width <= 0:
                return
            if new_width >= max_width:
                new_left, new_right = freq_min, freq_max
            else:
                if new_left < freq_min:
                    shift = freq_min - new_left
                    new_left = freq_min
                    new_right = min(freq_max, new_right + shift)
                if new_right > freq_max:
                    shift = new_right - freq_max
                    new_right = freq_max
                    new_left = max(freq_min, new_left - shift)
                new_left = max(new_left, freq_min)
                new_right = min(new_right, freq_max)
        if math.isclose(new_left, new_right):
            return
        if new_right - new_left <= 0:
            return
        self.ax_main.set_xlim(new_left, new_right)
        if self.canvas:
            self.canvas.draw_idle()

    def _set_hover_visible(self, freq: float | None) -> None:
        if self._hover_line is None or self._hover_text is None:
            return
        if freq is None:
            if self._hover_line.get_visible() or self._hover_text.get_visible():
                self._hover_line.set_visible(False)
                self._hover_text.set_visible(False)
                if self.canvas:
                    self.canvas.draw_idle()
            return
        self._hover_line.set_xdata([freq, freq])
        label = f"Cursor: {freq / 1e6:.6f} MHz"
        if self._hover_text.get_text() != label:
            self._hover_text.set_text(label)
        if not self._hover_line.get_visible():
            self._hover_line.set_visible(True)
        if not self._hover_text.get_visible():
            self._hover_text.set_visible(True)
        if self.canvas:
            self.canvas.draw_idle()

    def _update_selection_label(self) -> None:
        if self._selection_text is None:
            return
        if not self.state.selection:
            self._selection_text.set_text("Selection: —")
        else:
            center = self.state.selection.center_freq
            bandwidth = self.state.selection.bandwidth
            label = f"Selection: {center / 1e6:.6f} MHz • BW {bandwidth / 1e3:.1f} kHz"
            if self._selection_text.get_text() != label:
                self._selection_text.set_text(label)
        if self.canvas:
            self.canvas.draw_idle()

    def _on_span_changed(self, selection: SelectionResult) -> None:
        self.state.selection = selection
        self.state.set_bandwidth(selection.bandwidth)
        if self.channel_panel:
            self.channel_panel.bandwidth_entry.setText(f"{selection.bandwidth:.0f}")
        self._update_offset_label()
        self._update_selection_label()
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
        self.waterfall_window.update_plot(
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

    def _register_preview_pipeline(self, pipeline: Any | None) -> None:
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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
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
        if self.state.input_format_choice == "auto" and self.state.input_format_error:
            raise ValueError("Input format not detected. Choose a format or fix the recording metadata before previewing.")
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
        if self.state.input_format_choice == "auto" and self.state.input_format_error:
            raise ValueError("Input format not detected. Choose a format before running DSP.")
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
                    self._sync_state_targets_from_entries()
                    frequencies = list(self.state.target_freqs)
                else:
                    self.state.set_target_frequencies(frequencies)
                    self.state.target_text = [f"{freq:.0f}" for freq in frequencies]
                    if len(self.state.target_text) < self.state.max_target_freqs:
                        self.state.target_text.extend([""] * (self.state.max_target_freqs - len(self.state.target_text)))
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

    def _auto_detect_center(
        self,
        path: Path,
        *,
        announce: bool,
        force: bool = False,
        preserve_manual: bool = False,
    ) -> None:
        previous_center = self.state.center_freq
        previous_source = self.state.center_source
        if (
            not force
            and previous_source in {"manual", "provided"}
            and previous_center
            and previous_center > 0
        ):
            return
        detection = detect_center_frequency(path)
        if detection.value is None:
            if (
                preserve_manual
                and previous_source in {"manual", "provided"}
                and previous_center
                and previous_center > 0
            ):
                if announce:
                    self._set_status(
                        "Unable to detect center frequency; keeping manual value.",
                        error=True,
                    )
                return
            self.state.center_freq = None
            self.state.center_source = "unavailable"
            self.state.base_kwargs.pop("center_freq", None)
            self.state.base_kwargs["center_freq_source"] = "unavailable"
            if self.recording_panel:
                entry = self.recording_panel.center_entry
                entry.blockSignals(True)
                entry.clear()
                entry.blockSignals(False)
            self._refresh_center_source_label()
            if announce:
                self._set_status(
                    "Unable to detect center frequency; enter a value before previewing.",
                    error=True,
                )
            return
        self.state.update_center(detection.value, detection.source or "metadata")
        if self.recording_panel:
            entry = self.recording_panel.center_entry
            entry.blockSignals(True)
            entry.setText(f"{detection.value:.0f}")
            entry.blockSignals(False)
        self._refresh_center_source_label()
        if announce:
            self._set_status(
                f"Center frequency detected ({self._describe_center_source(self.state.center_source)}).",
                error=False,
            )

    def _refresh_center_source_label(self) -> None:
        if not self.recording_panel:
            return
        label = self._describe_center_source(self.state.center_source)
        self.recording_panel.center_source_label.setText(f"Center source: {label}")

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

    if not isinstance(qapp, QApplication):
        raise RuntimeError("QApplication instance missing during launch.")

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
