from __future__ import annotations

import logging
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import SpanSelector

from .processing import ProcessingCancelled, ProcessingConfig, tune_chunk_size
from .preview import run_preview
from .probe import SampleRateProbe, probe_sample_rate
from .progress import ProgressSink
from .spectrum import WaterfallResult, compute_psd, streaming_waterfall
from .utils import detect_center_frequency
from .visualize import SelectionResult, ensure_matplotlib

LOG = logging.getLogger(__name__)

QT_DEPENDENCY_HINT = (
    "PySide6 is required for --interactive. Install it via "
    "`uv pip install PySide6 PySide6-Addons`."
)


@dataclass
class InteractiveOutcome:
    center_freq: float
    target_freq: float
    bandwidth: float
    probe: SampleRateProbe


@dataclass
class InteractiveSessionResult:
    configs: list[ProcessingConfig]
    progress_sink: Optional[ProgressSink]

    @property
    def config(self) -> ProcessingConfig:
        return self.configs[0]


MAX_PREVIEW_SAMPLES = 8_000_000  # Complex samples retained in memory for previews (~64 MB).
MAX_TARGET_FREQUENCIES = 5


@dataclass
class SnapshotData:
    path: Path
    sample_rate: float
    center_freq: float
    probe: SampleRateProbe
    seconds: float
    mode: str
    freqs: np.ndarray
    psd_db: np.ndarray
    waterfall: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]
    samples: Optional[np.ndarray]
    params: Dict[str, Any]
    fft_frames: int


class StatusProgressSink(ProgressSink):
    """Simple progress sink that reflects pipeline status in the status bar."""

    def __init__(self, update: Callable[[str, bool], None]):
        self._update = update
        self._status: Optional[str] = None
        self._overall_total = 0.0
        self._overall_completed = 0.0
        self._cancel_callback: Optional[Callable[[], None]] = None

    def start(self, phases, *, overall_total: float) -> None:
        self._overall_total = max(overall_total, 0.0)
        self._overall_completed = 0.0
        self._status = "Processing…"
        self._emit(highlight=True)

    def advance(
        self,
        phase,
        delta: float,
        *,
        overall_completed: float,
        overall_total: float,
    ) -> None:
        if delta <= 0:
            return
        self._overall_completed = max(0.0, overall_completed)
        self._overall_total = max(self._overall_total, overall_total)
        self._emit(highlight=True)

    def status(self, message: str) -> None:
        self._status = message
        self._emit(highlight=True)

    def close(self) -> None:
        self._update("Processing complete.", False)

    def cancel(self) -> None:
        self._update("Cancelling…", True)

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        self._cancel_callback = callback

    def trigger_cancel(self) -> None:
        if self._cancel_callback is not None:
            self._cancel_callback()

    def _emit(self, *, highlight: bool) -> None:
        message = self._status or "Processing…"
        if self._overall_total > 0 and self._overall_completed > 0:
            pct = 100.0 * min(self._overall_completed / self._overall_total, 1.0)
            message = f"{message} — {pct:4.1f}%"
        self._update(message, highlight)


def gather_snapshot(
    config: ProcessingConfig,
    seconds: float = 2.0,
    *,
    nfft: int,
    hop: Optional[int],
    max_slices: int,
    fft_workers: Optional[int],
    max_in_memory_samples: int = MAX_PREVIEW_SAMPLES,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> SnapshotData:
    """Stream a preview segment of IQ data for interactive spectrum display."""
    probe = probe_sample_rate(config.in_path)
    sample_rate = probe.value

    center_freq = config.center_freq
    if center_freq is None:
        detection = detect_center_frequency(config.in_path)
        if detection.value is None:
            raise ValueError(
                "Center frequency not provided and could not be inferred from WAV metadata or filename. "
                "Provide --fc or enter a value before previewing."
            )
        center_freq = detection.value

    total_samples = int(max(1, round(sample_rate * seconds)))
    hop = max(1, hop or nfft // 4)
    tuned_chunk = tune_chunk_size(sample_rate, config.chunk_size)
    chunk_size = max(tuned_chunk, nfft)
    retain = min(max_in_memory_samples, total_samples)
    retain_buffer = (
        np.empty(retain, dtype=np.complex64) if retain > 0 else None
    )
    retain_pos = 0
    consumed = 0
    last_report = -1.0

    from .processing import IQReader  # Local import to avoid circular refs.

    def _chunk_iter() -> Iterator[np.ndarray]:
        nonlocal retain_pos, consumed, last_report
        remaining = total_samples
        with IQReader(config.in_path, chunk_size, config.iq_order) as reader:
            for block in reader:
                if remaining <= 0:
                    break
                use = block
                if block.size > remaining:
                    use = block[:remaining]
                remaining -= use.size
                consumed += use.size
                if retain_buffer is not None and retain_pos < retain_buffer.size:
                    take = min(retain_buffer.size - retain_pos, use.size)
                    if take > 0:
                        retain_buffer[retain_pos : retain_pos + take] = use[:take]
                        retain_pos += take
                if progress_cb:
                    try:
                        frac = min(consumed / total_samples, 1.0)
                        if frac - last_report >= 0.02 or frac >= 0.999:
                            seconds_done = consumed / sample_rate
                            progress_cb(seconds_done, frac)
                            last_report = frac
                    except Exception:
                        pass
                yield use
                if remaining <= 0:
                    break

    freqs, avg_psd, waterfall, frames = streaming_waterfall(
        _chunk_iter(),
        sample_rate,
        nfft=nfft,
        hop=hop,
        max_slices=max_slices,
        fft_workers=fft_workers,
    )

    samples = None
    if retain_buffer is not None and retain_pos > 0:
        samples = retain_buffer[:retain_pos].copy()

    params: Dict[str, Any] = {
        "nfft": nfft,
        "hop": hop,
        "max_slices": max_slices,
        "fft_workers": fft_workers,
        "seconds": seconds,
        "full_capture": False,
        "max_in_memory_samples": max_in_memory_samples,
    }
    snapshot = SnapshotData(
        path=config.in_path,
        sample_rate=sample_rate,
        center_freq=center_freq,
        probe=probe,
        seconds=consumed / sample_rate,
        mode="samples" if samples is not None else "precomputed",
        freqs=freqs,
        psd_db=avg_psd,
        waterfall=_waterfall_to_tuple(waterfall),
        samples=samples,
        params=params,
        fft_frames=frames,
    )
    if progress_cb:
        try:
            progress_cb(snapshot.seconds, 1.0)
        except Exception:
            pass
    LOG.info(
        "Snapshot complete: %.2f s processed (%d FFT frames, mode=%s).",
        snapshot.seconds,
        frames,
        snapshot.mode,
    )
    return snapshot


def _waterfall_to_tuple(
    waterfall: Optional[WaterfallResult],
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if waterfall is None:
        return None
    return (
        np.asarray(waterfall.freqs, dtype=np.float64),
        np.asarray(waterfall.times, dtype=np.float32),
        np.asarray(waterfall.matrix, dtype=np.float32),
    )


class _WaterfallWindow(QMainWindow):
    """Secondary window hosting the waterfall plot."""

    def __init__(self, parent: QMainWindow, on_select, on_close) -> None:
        super().__init__(parent)
        self._on_select = on_select
        self._on_close = on_close

        self.setWindowTitle("Waterfall (time vs frequency)")
        self.resize(900, 700)

        central = QWidget()
        layout = QVBoxLayout(central)

        self.figure = Figure(figsize=(8.5, 5.5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.draw()
        layout.addWidget(self.canvas)

        self.setCentralWidget(central)

        self.cid = self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.freqs_hz: Optional[np.ndarray] = None
        self.center_freq = 0.0
        self.sample_rate = 0.0
        self.image = None
        self.alive = True
        self._colorbar = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._handle_close()
        event.accept()

    def _handle_close(self) -> None:
        self.alive = False
        try:
            if self.cid is not None:
                self.figure.canvas.mpl_disconnect(self.cid)
        except Exception:
            pass
        if self._on_close:
            self._on_close()

    def update(
        self,
        *,
        freqs: np.ndarray,
        times: np.ndarray,
        matrix: np.ndarray,
        center_freq: float,
        sample_rate: float,
        floor_db: float,
        cmap: str,
    ) -> None:
        if freqs.size == 0 or matrix.size == 0:
            return
        self.freqs_hz = center_freq + freqs
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None
        self.ax.clear()
        peak = float(np.max(matrix[np.isfinite(matrix)])) if np.isfinite(matrix).any() else 0.0
        vmin = peak - max(20.0, floor_db)
        vmax = peak
        freq_mhz = self.freqs_hz / 1e6
        extent = [freq_mhz.min(), freq_mhz.max(), times.max(), times.min()]
        self.image = self.ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        self.ax.set_xlabel("Frequency (MHz)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title("Waterfall (newest at bottom)")
        self._colorbar = self.figure.colorbar(
            self.image, ax=self.ax, orientation="vertical", label="Power (dB)"
        )
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        freq_mhz = float(event.xdata)
        freq_hz = freq_mhz * 1e6
        if self._on_select:
            self._on_select(freq_hz)


class _SpanController:
    """Track matplotlib SpanSelector selections and update overlays."""

    def __init__(
        self,
        *,
        ax,
        canvas: FigureCanvasQTAgg,
        initial: SelectionResult,
        on_change,
    ):
        self.ax = ax
        self.canvas = canvas
        self.on_change = on_change
        self.selection = SelectionResult(initial.center_freq, initial.bandwidth)

        half_bw = max(self.selection.bandwidth / 2.0, 1.0)
        self.center_line = self.ax.axvline(
            self.selection.center_freq, color="C3", ls="--", lw=1.2, label="Center"
        )
        self.low_line = self.ax.axvline(
            self.selection.center_freq - half_bw, color="C2", ls=":", lw=1.0
        )
        self.high_line = self.ax.axvline(
            self.selection.center_freq + half_bw, color="C2", ls=":", lw=1.0
        )
        self.ax.legend(loc="upper right")

        self.selector = SpanSelector(
            self.ax,
            onselect=self._on_select,
            direction="horizontal",
            useblit=True,
        )
        self._emit()

    def _on_select(self, xmin: float, xmax: float) -> None:
        if xmin == xmax:
            return
        lo, hi = sorted((xmin, xmax))
        center = 0.5 * (lo + hi)
        bw = max(hi - lo, 10.0)
        self.selection = SelectionResult(center, bw)
        self._update_lines()
        self._emit()

    def _update_lines(self) -> None:
        half_bw = max(self.selection.bandwidth / 2.0, 1.0)
        self.center_line.set_xdata(
            [self.selection.center_freq, self.selection.center_freq]
        )
        self.low_line.set_xdata(
            [self.selection.center_freq - half_bw] * 2
        )
        self.high_line.set_xdata(
            [self.selection.center_freq + half_bw] * 2
        )
        self.canvas.draw_idle()

    def _emit(self) -> None:
        if self.on_change:
            self.on_change(self.selection)

    def set_selection(self, center: float, bandwidth: float) -> None:
        self.selection = SelectionResult(center, max(bandwidth, 10.0))
        self._update_lines()
        self._emit()

class _InteractiveApp(QMainWindow):
    """PySide6-based interactive spectrum viewer for IQ to Audio."""

    status_update_signal = Signal(str, bool)
    snapshot_ready_signal = Signal(object, object, object)  # snapshot, path, previous_path
    snapshot_failed_signal = Signal(Exception, bool)
    snapshot_finished_signal = Signal()
    preview_complete_signal = Signal(object, bool)  # result/error flag

    def __init__(
        self,
        *,
        base_kwargs: dict,
        initial_path: Optional[Path],
        snapshot_seconds: float,
    ):
        super().__init__()
        self.base_kwargs = dict(base_kwargs)
        self.initial_path = initial_path
        self.default_snapshot = max(snapshot_seconds, 0.25)

        # Session state
        self.selected_path: Optional[Path] = initial_path
        self.selection: Optional[SelectionResult] = None
        self.sample_rate: Optional[float] = None
        self.center_freq: Optional[float] = self.base_kwargs.get("center_freq")
        self.probe: Optional[SampleRateProbe] = None
        self.snapshot_seconds = self.default_snapshot
        self.snapshot_data: Optional[SnapshotData] = None
        self._preview_thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None
        self._preview_running = False
        self._status_sink: Optional[StatusProgressSink] = None
        self.progress_sink: Optional[ProgressSink] = None
        self.waterfall_window: Optional[_WaterfallWindow] = None
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasQTAgg] = None
        self.toolbar: Optional[NavigationToolbar2QT] = None
        self.ax_main = None
        self.span_controller: Optional[_SpanController] = None
        self._freq_min_hz: Optional[float] = None
        self._freq_max_hz: Optional[float] = None
        self.plot_layout: Optional[QVBoxLayout] = None
        self.status_bar: Optional[QStatusBar] = None
        self._preview_lock = threading.Lock()
        self._active_pipeline = None
        self._refresh_timer: Optional[QtCore.QTimer] = None
        self._refresh_pending: tuple[bool, bool] = (False, False)

        # UI state values (replacing Tkinter Variable instances)
        self.file_value = str(initial_path) if initial_path else ""
        self.center_value = self._format_float(self.base_kwargs.get("center_freq"))
        self.center_source_value = "Center source: —"
        initial_source = self.base_kwargs.get("center_freq_source")
        self.center_source = initial_source or "unavailable"
        if initial_source:
            self.center_source_value = f"Center source: {initial_source}"
        self.snapshot_text = f"{self.default_snapshot:.2f}"
        provided_output = self.base_kwargs.get("output_path")
        self._cli_output_path: Optional[Path] = Path(provided_output) if provided_output else None
        self.output_dir_value = (
            str(self._cli_output_path.parent) if self._cli_output_path else ""
        )
        self.output_hint = "Select a recording to preview output location."
        demod_mode = (self.base_kwargs.get("demod_mode") or "nfm").lower()
        self.demod_options = ("nfm", "am", "usb", "lsb", "ssb")
        if demod_mode not in self.demod_options:
            demod_mode = "nfm"
        self.demod_value = demod_mode
        self.squelch_enabled = bool(self.base_kwargs.get("squelch_enabled", True))
        self.trim_enabled = bool(self.base_kwargs.get("silence_trim", False))
        self.agc_enabled = bool(self.base_kwargs.get("agc_enabled", True))
        self._preferred_agc = self.agc_enabled
        threshold = self.base_kwargs.get("squelch_dbfs")
        self.squelch_threshold_value = "" if threshold is None else f"{threshold:.1f}"
        self.full_snapshot = False
        self.nfft_value = "262144"
        self.smooth_value = 3
        self.range_value = 100
        self.theme_value = "contrast"
        self.color_themes: dict[str, dict[str, str]] = {
            "default": {
                "bg": "white",
                "face": "white",
                "line": "#1f77b4",
                "fg": "black",
                "grid": ":",
                "grid_color": "#d0d0d0",
            },
            "contrast": {
                "bg": "#101010",
                "face": "#101010",
                "line": "#ff7600",
                "fg": "white",
                "grid": "--",
                "grid_color": "#444444",
            },
            "night": {
                "bg": "#0b1a2a",
                "face": "#0b1a2a",
                "line": "#7fffd4",
                "fg": "#f0f4ff",
                "grid": ":",
                "grid_color": "#223347",
            },
        }
        self.waterfall_cmap_value = "magma"
        self.waterfall_slices_value = "400"
        self.waterfall_floor_value = "110"
        bandwidth = self.base_kwargs.get("bandwidth")
        self.bandwidth_value = "—" if bandwidth is None else f"{bandwidth:.0f}"
        raw_targets = self.base_kwargs.get("target_freqs") or []
        if not raw_targets:
            initial_target = self.base_kwargs.get("target_freq")
            if initial_target:
                raw_targets = [initial_target]
        defaults: list[str] = []
        for value in list(raw_targets)[:MAX_TARGET_FREQUENCIES]:
            try:
                freq_val = float(value)
            except (TypeError, ValueError):
                freq_val = 0.0
            defaults.append(f"{freq_val:.0f}" if freq_val > 0 else "")
        while len(defaults) < MAX_TARGET_FREQUENCIES:
            defaults.append("")
        self.target_freq_values: list[str] = defaults
        self.offset_value = "Offset: —"
        self.sample_rate_value = "Sample rate: —"
        self.status_message = "Select a recording to begin."

        # Widget references
        self.file_entry: Optional[QLineEdit] = None
        self.center_entry: Optional[QLineEdit] = None
        self.center_source_label: Optional[QLabel] = None
        self.snapshot_entry: Optional[QLineEdit] = None
        self.load_preview_button: Optional[QPushButton] = None
        self.full_snapshot_check: Optional[QCheckBox] = None
        self.output_dir_entry: Optional[QLineEdit] = None
        self.output_hint_label: Optional[QLabel] = None
        self.demod_combo: Optional[QComboBox] = None
        self.squelch_check: Optional[QCheckBox] = None
        self.trim_check: Optional[QCheckBox] = None
        self.agc_check: Optional[QCheckBox] = None
        self.squelch_threshold_entry: Optional[QLineEdit] = None
        self.plot_group: Optional[QGroupBox] = None
        self.placeholder_label: Optional[QLabel] = None
        self.spectrum_nfft_combo: Optional[QComboBox] = None
        self.spectrum_theme_combo: Optional[QComboBox] = None
        self.spectrum_refresh_button: Optional[QPushButton] = None
        self.spectrum_smooth_spin: Optional[QSpinBox] = None
        self.spectrum_range_spin: Optional[QSpinBox] = None
        self.waterfall_slices_spin: Optional[QSpinBox] = None
        self.waterfall_floor_spin: Optional[QSpinBox] = None
        self.waterfall_cmap_combo: Optional[QComboBox] = None
        self.bandwidth_entry: Optional[QLineEdit] = None
        self.sample_rate_label: Optional[QLabel] = None
        self.offset_label: Optional[QLabel] = None
        self.target_entries: list[QLineEdit] = []
        self.status_label: Optional[QLabel] = None
        self.stop_btn: Optional[QPushButton] = None
        self.preview_btn: Optional[QPushButton] = None
        self.confirm_btn: Optional[QPushButton] = None

        # Signal wiring
        self.status_update_signal.connect(self._set_status)
        self.snapshot_ready_signal.connect(self._on_snapshot_ready)
        self.snapshot_failed_signal.connect(self._on_snapshot_failed)
        self.snapshot_finished_signal.connect(self._on_snapshot_thread_finished)
        self.preview_complete_signal.connect(self._on_preview_completed)

        self._apply_styles()

        # Window setup
        self.setWindowTitle("IQ to Audio — Interactive Mode")
        self._configure_main_window()
        self._build_ui()
        self._update_option_state()
        self._update_output_path_hint()
        self._set_status(self.status_message, error=False)

        if initial_path:
            self._schedule_preview(auto=True)

    def run(self) -> InteractiveSessionResult:
        try:
            self.show()
            app = QApplication.instance()
            if app is None:
                raise RuntimeError("QApplication instance missing during run()")
            app.exec()
        finally:
            self._cancel_active_pipeline()
            self._join_threads()
        if not self.selection or not self.selected_path:
            raise KeyboardInterrupt()
        self.base_kwargs["silence_trim"] = self.trim_enabled
        self.base_kwargs["squelch_enabled"] = self.squelch_enabled
        self.base_kwargs["agc_enabled"] = self.agc_enabled
        self.base_kwargs["demod_mode"] = self.demod_value
        configs = self._build_configs(self.selected_path, self.center_freq)
        if not configs:
            raise ValueError("Enter at least one target frequency before running DSP.")
        LOG.info(
            "Interactive selection: center %.0f Hz, %d target(s), bandwidth %.0f Hz",
            self.center_freq or 0.0,
            len(configs),
            configs[0].bandwidth,
        )
        return InteractiveSessionResult(configs=configs, progress_sink=self.progress_sink)

    def _apply_styles(self) -> None:
        palette = self.palette()
        border = palette.color(QtGui.QPalette.ColorRole.Mid).name()
        bg = palette.color(QtGui.QPalette.ColorRole.Base).name()
        self.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {border};
                border-radius: 6px;
                margin-top: 8px;
                background-color: {bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }}
            """
        )

    def _configure_main_window(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            geometry = screen.availableGeometry()
            width = min(1150, max(1000, geometry.width() - 160))
            height = min(max(780, geometry.height() - 200), geometry.height() - 60)
            height = max(680, height)
        else:
            width, height = 1150, 780
        self.resize(int(width), int(height))
        self.setMinimumSize(960, 680)

    def _build_ui(self) -> None:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        self._build_recording_section(main_layout)
        self._build_demod_options_section(main_layout)
        self._build_plot_section(main_layout)
        self._build_spectrum_options_section(main_layout)
        self._build_waterfall_options_section(main_layout)
        self._build_selection_section(main_layout)
        self._build_targets_section(main_layout)
        self._build_status_section(main_layout)
        self._build_button_row(main_layout)

        content.setLayout(main_layout)
        scroll_area.setWidget(content)
        self.setCentralWidget(scroll_area)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.status_message)

    def _build_recording_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Recording")
        layout = QVBoxLayout()

        # Input file row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Input WAV:"))
        self.file_entry = QLineEdit(self.file_value)
        self.file_entry.setPlaceholderText("Select a baseband WAV recording…")
        self.file_entry.textChanged.connect(self._on_file_text_changed)
        file_row.addWidget(self.file_entry, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Center frequency row
        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("Center freq (Hz):"))
        self.center_entry = QLineEdit(self.center_value)
        self.center_entry.setMaximumWidth(180)
        self.center_entry.editingFinished.connect(self._on_center_manual)
        center_row.addWidget(self.center_entry)
        detect_btn = QPushButton("Detect from file")
        detect_btn.clicked.connect(self._parse_center_from_name)
        center_row.addWidget(detect_btn)
        self.center_source_label = QLabel(self.center_source_value)
        self.center_source_label.setStyleSheet("color: #708090;")
        center_row.addWidget(self.center_source_label)
        center_row.addStretch()
        layout.addLayout(center_row)

        # Snapshot row
        snapshot_row = QHBoxLayout()
        snapshot_row.addWidget(QLabel("Snapshot (seconds):"))
        self.snapshot_entry = QLineEdit(self.snapshot_text)
        self.snapshot_entry.setMaximumWidth(100)
        self.snapshot_entry.editingFinished.connect(self._on_snapshot_changed)
        snapshot_row.addWidget(self.snapshot_entry)
        self.load_preview_button = QPushButton("Load FFT")
        self.load_preview_button.clicked.connect(lambda: self._schedule_preview(auto=False))
        snapshot_row.addWidget(self.load_preview_button)
        self.full_snapshot_check = QCheckBox("Analyze entire recording")
        self.full_snapshot_check.setChecked(self.full_snapshot)
        self.full_snapshot_check.stateChanged.connect(self._on_full_snapshot_toggled)
        snapshot_row.addWidget(self.full_snapshot_check)
        snapshot_row.addStretch()
        layout.addLayout(snapshot_row)

        # Output directory row
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output directory:"))
        self.output_dir_entry = QLineEdit(self.output_dir_value)
        self.output_dir_entry.setPlaceholderText("Optional override – defaults beside input WAV")
        self.output_dir_entry.editingFinished.connect(self._on_output_dir_changed)
        output_row.addWidget(self.output_dir_entry, stretch=1)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._on_output_dir_browse)
        output_row.addWidget(out_browse)
        layout.addLayout(output_row)

        self.output_hint_label = QLabel(self.output_hint)
        self.output_hint_label.setWordWrap(True)
        self.output_hint_label.setStyleSheet("color: #708090;")
        layout.addWidget(self.output_hint_label)

        # Demod row
        demod_row = QHBoxLayout()
        demod_row.addWidget(QLabel("Demodulator:"))
        self.demod_combo = QComboBox()
        self.demod_combo.addItems(self.demod_options)
        self.demod_combo.setCurrentText(self.demod_value)
        self.demod_combo.currentTextChanged.connect(self._on_demod_changed)
        demod_row.addWidget(self.demod_combo)
        demod_help = QLabel("Choose AM/NFM/USB/LSB demodulation")
        demod_help.setStyleSheet("color: #708090;")
        demod_row.addWidget(demod_help)
        demod_row.addStretch()
        layout.addLayout(demod_row)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_plot_section(self, parent_layout: QVBoxLayout) -> None:
        self.plot_group = QGroupBox("Spectrum preview")
        layout = QVBoxLayout()
        self.plot_layout = layout
        self.placeholder_label = QLabel("Load a recording to view its spectrum.")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setMinimumHeight(420)
        layout.addWidget(self.placeholder_label)
        self.plot_group.setLayout(layout)
        parent_layout.addWidget(self.plot_group, stretch=1)

    def _build_spectrum_options_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Spectrum options")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("FFT size"))
        self.spectrum_nfft_combo = QComboBox()
        self.spectrum_nfft_combo.addItems(["65536", "131072", "262144", "524288"])
        self.spectrum_nfft_combo.setCurrentText(self.nfft_value)
        layout.addWidget(self.spectrum_nfft_combo)

        layout.addWidget(QLabel("Smoothing"))
        self.spectrum_smooth_spin = QSpinBox()
        self.spectrum_smooth_spin.setRange(1, 20)
        self.spectrum_smooth_spin.setValue(self.smooth_value)
        layout.addWidget(self.spectrum_smooth_spin)

        layout.addWidget(QLabel("Dynamic range (dB)"))
        self.spectrum_range_spin = QSpinBox()
        self.spectrum_range_spin.setRange(20, 140)
        self.spectrum_range_spin.setValue(self.range_value)
        layout.addWidget(self.spectrum_range_spin)

        layout.addWidget(QLabel("Theme"))
        self.spectrum_theme_combo = QComboBox()
        self.spectrum_theme_combo.addItems(list(self.color_themes.keys()))
        if self.theme_value in self.color_themes:
            self.spectrum_theme_combo.setCurrentText(self.theme_value)
        self.spectrum_theme_combo.currentTextChanged.connect(self._on_theme_changed)
        layout.addWidget(self.spectrum_theme_combo)

        reset_btn = QPushButton("Reset defaults")
        reset_btn.clicked.connect(self._reset_spectrum_defaults)
        layout.addWidget(reset_btn)

        self.spectrum_refresh_button = QPushButton("Refresh preview")
        self.spectrum_refresh_button.clicked.connect(self._refresh_preview_manual)
        layout.addWidget(self.spectrum_refresh_button)

        layout.addStretch()
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_waterfall_options_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Waterfall options")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Max slices"))
        self.waterfall_slices_spin = QSpinBox()
        self.waterfall_slices_spin.setRange(50, 800)
        self.waterfall_slices_spin.setValue(int(self.waterfall_slices_value))
        layout.addWidget(self.waterfall_slices_spin)

        layout.addWidget(QLabel("Range (dB)"))
        self.waterfall_floor_spin = QSpinBox()
        self.waterfall_floor_spin.setRange(20, 140)
        self.waterfall_floor_spin.setValue(int(self.waterfall_floor_value))
        layout.addWidget(self.waterfall_floor_spin)

        layout.addWidget(QLabel("Colormap"))
        self.waterfall_cmap_combo = QComboBox()
        self.waterfall_cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis"])
        self.waterfall_cmap_combo.setCurrentText(self.waterfall_cmap_value)
        layout.addWidget(self.waterfall_cmap_combo)

        layout.addStretch()
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_selection_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Channel selection")
        layout = QVBoxLayout()

        self.sample_rate_label = QLabel(self.sample_rate_value)
        layout.addWidget(self.sample_rate_label)

        bandwidth_row = QHBoxLayout()
        bandwidth_row.addWidget(QLabel("Bandwidth (Hz):"))
        self.bandwidth_entry = QLineEdit(self.bandwidth_value)
        self.bandwidth_entry.setMaximumWidth(140)
        self.bandwidth_entry.editingFinished.connect(self._on_bandwidth_edit)
        bandwidth_row.addWidget(self.bandwidth_entry)
        bandwidth_row.addStretch()
        layout.addLayout(bandwidth_row)

        self.offset_label = QLabel(self.offset_value)
        layout.addWidget(self.offset_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _build_targets_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Target frequencies (Hz)")
        grid = QtWidgets.QGridLayout()
        self.target_entries = []
        per_row = 3
        for idx, value in enumerate(self.target_freq_values):
            row = idx // per_row
            col = (idx % per_row) * 2
            grid.addWidget(QLabel(f"Target {idx + 1}:"), row, col, alignment=Qt.AlignmentFlag.AlignLeft)
            entry = QLineEdit(value)
            entry.setMaximumWidth(160)
            entry.editingFinished.connect(lambda i=idx, line=entry: self._on_target_entry_edit(i, line.text()))
            grid.addWidget(entry, row, col + 1)
            self.target_entries.append(entry)
        group.setLayout(grid)
        parent_layout.addWidget(group)

    def _build_status_section(self, parent_layout: QVBoxLayout) -> None:
        self.status_label = QLabel(self.status_message)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #708090;")
        parent_layout.addWidget(self.status_label)

    def _build_button_row(self, parent_layout: QVBoxLayout) -> None:
        row = QHBoxLayout()

        self.stop_btn = QPushButton("Stop DSP")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_processing)
        row.addWidget(self.stop_btn)

        row.addStretch()

        self.preview_btn = QPushButton("Preview DSP")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._on_preview)
        row.addWidget(self.preview_btn)

        self.confirm_btn = QPushButton("Confirm && Run")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self._on_confirm)
        row.addWidget(self.confirm_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(cancel_btn)

        parent_layout.addLayout(row)

    @Slot(str, bool)
    def _set_status(self, message: str, error: bool) -> None:
        self.status_message = message
        color = "#B22222" if error else "#708090"
        if self.status_label:
            self.status_label.setText(message)
            self.status_label.setStyleSheet(f"color: {color};")
        if self.status_bar:
            self.status_bar.showMessage(message)

    def _join_threads(self) -> None:
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=5.0)
        self._preview_thread = None
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=5.0)
        self._snapshot_thread = None

    def _cancel_active_pipeline(self) -> None:
        with self._preview_lock:
            pipeline = self._active_pipeline
        if pipeline is not None:
            try:
                pipeline.cancel()
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("Failed to cancel pipeline cleanly: %s", exc)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._on_cancel()
        event.accept()

    def _set_center_source(self, source: Optional[str]) -> None:
        resolved = source or "unavailable"
        self.center_source = resolved
        self.base_kwargs["center_freq_source"] = resolved
        if resolved == "unavailable":
            label = "—"
        else:
            label = self._describe_center_source(resolved)
        self.center_source_value = f"Center source: {label}"
        if self.center_source_label:
            self.center_source_label.setText(self.center_source_value)

    def _describe_center_source(self, source: Optional[str]) -> str:
        resolved = source or "unavailable"
        if resolved in {"unavailable", "manual"}:
            return "manual entry"
        if resolved.startswith("metadata:"):
            detail = resolved.split(":", 1)[1] or "metadata"
            return f"WAV metadata ({detail})"
        if resolved.startswith("filename"):
            return "filename pattern"
        if resolved == "config":
            return "configuration"
        return resolved

    def _current_output_dir(self) -> Optional[Path]:
        text = self.output_dir_entry.text().strip() if self.output_dir_entry else self.output_dir_value.strip()
        if not text:
            return None
        return Path(text).expanduser()

    def _default_output_filename(self, target_freq: float) -> str:
        finest = int(round(target_freq)) if target_freq else 0
        return f"audio_{finest}_48k.wav"

    def _resolve_output_path(self, input_path: Path, target_freq: float) -> Optional[Path]:
        directory = self._current_output_dir()
        if directory:
            return directory / self._default_output_filename(target_freq)
        if self._cli_output_path is not None:
            return self._cli_output_path
        existing = self.base_kwargs.get("output_path")
        if existing:
            return Path(existing)
        return None

    def _update_output_path_hint(self) -> None:
        target = self.selection.center_freq if self.selection else self.base_kwargs.get("target_freq", 0.0)
        input_path = self.selected_path or (Path(self.file_value) if self.file_value else None)
        resolved: Optional[Path] = None
        if input_path:
            try:
                resolved = self._resolve_output_path(input_path, target)
            except Exception:
                resolved = None
        if resolved is None and not input_path and self._cli_output_path is not None:
            resolved = self._cli_output_path
        if resolved is None and input_path:
            resolved = input_path.with_name(self._default_output_filename(target))
        if resolved is not None:
            hint = f"Preview/output files: {resolved}"
        else:
            filename = self._default_output_filename(target)
            hint = f"Preview/output files: {filename} (default)"
        self.output_hint = hint
        if self.output_hint_label:
            self.output_hint_label.setText(hint)

    def _on_file_text_changed(self, text: str) -> None:
        self.file_value = text.strip()
        if self.file_value:
            self.selected_path = Path(self.file_value)
        else:
            self.selected_path = None
        self._update_output_path_hint()
        self._update_option_state()

    def _on_browse(self) -> None:
        initial_dir = str(Path(self.file_value).parent) if self.file_value else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SDR++ baseband WAV",
            initial_dir,
            "WAV files (*.wav);;All files (*.*)",
        )
        if not file_path:
            return
        self.file_value = file_path
        if self.file_entry:
            self.file_entry.setText(file_path)
        self.selected_path = Path(file_path)
        self._update_output_path_hint()
        self._parse_center_from_name(silent=True)
        self._schedule_preview(auto=True)

    def _on_snapshot_changed(self) -> None:
        if not self.snapshot_entry:
            return
        text = self.snapshot_entry.text().strip()
        value = self._parse_float(text)
        if value is None or value <= 0:
            self.snapshot_entry.setText(self.snapshot_text)
            self._set_status("Snapshot duration must be a positive number.", error=True)
            return
        self.snapshot_seconds = value
        self.snapshot_text = f"{value:.2f}"
        self.snapshot_entry.setText(self.snapshot_text)

    def _on_full_snapshot_toggled(self, state: int) -> None:
        self.full_snapshot = state == Qt.CheckState.Checked

    def _on_output_dir_browse(self) -> None:
        current = self._current_output_dir()
        if current is None and self.selected_path:
            current = self.selected_path.parent
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            str(current) if current else "",
        )
        if directory:
            self.output_dir_value = directory
            if self.output_dir_entry:
                self.output_dir_entry.setText(directory)
            self._on_output_dir_changed()
        else:
            if self.output_dir_entry:
                self.output_dir_entry.setText(self.output_dir_value)

    def _on_output_dir_changed(self) -> None:
        if self.output_dir_entry:
            self.output_dir_value = self.output_dir_entry.text().strip()
        directory = self._current_output_dir()
        if directory:
            self._set_status(f"Output directory set to {directory}", error=False)
        else:
            self._set_status(
                "Output directory cleared; previews and outputs will default beside the recording.",
                error=False,
            )
        self._update_output_path_hint()

    def _on_demod_changed(self, value: str) -> None:
        self.demod_value = value.lower()
        self._update_option_state()

    def _on_toggle_squelch(self, state: int) -> None:
        self.squelch_enabled = state == Qt.CheckState.Checked
        self._update_option_state()

    def _on_toggle_trim(self, state: int) -> None:
        self.trim_enabled = state == Qt.CheckState.Checked

    def _on_toggle_agc(self, state: int) -> None:
        self.agc_enabled = state == Qt.CheckState.Checked
        self._update_option_state()

    def _on_squelch_threshold_edit(self) -> None:
        if not self.squelch_threshold_entry:
            return
        text = self.squelch_threshold_entry.text().strip()
        if not text:
            self.squelch_threshold_value = ""
            self.base_kwargs["squelch_dbfs"] = None
            return
        try:
            value = float(text)
        except ValueError:
            self.squelch_threshold_entry.setText(self.squelch_threshold_value)
            self._set_status("Invalid squelch threshold; enter a numeric dBFS value.", error=True)
            return
        self.squelch_threshold_value = f"{value:.1f}"
        self.squelch_threshold_entry.setText(self.squelch_threshold_value)
        self.base_kwargs["squelch_dbfs"] = value

    def _on_theme_changed(self, name: str) -> None:
        if name not in self.color_themes:
            return
        self.theme_value = name
        if self.snapshot_data:
            self._render_snapshot(self.snapshot_data, remember=False)

    def _reset_spectrum_defaults(self) -> None:
        self.nfft_value = "262144"
        self.smooth_value = 3
        self.range_value = 100
        self.theme_value = "contrast"
        if self.spectrum_nfft_combo:
            self.spectrum_nfft_combo.setCurrentText(self.nfft_value)
        if self.spectrum_smooth_spin:
            self.spectrum_smooth_spin.setValue(self.smooth_value)
        if self.spectrum_range_spin:
            self.spectrum_range_spin.setValue(self.range_value)
        if self.spectrum_theme_combo:
            self.spectrum_theme_combo.setCurrentText(self.theme_value)
        if self.snapshot_data:
            self._render_snapshot(self.snapshot_data, remember=False)

    def _refresh_preview_manual(self) -> None:
        self._schedule_preview(auto=False)

    def _on_bandwidth_edit(self) -> None:
        if not self.bandwidth_entry:
            return
        text = self.bandwidth_entry.text().strip()
        try:
            bandwidth = float(text)
            if bandwidth <= 0:
                raise ValueError()
        except ValueError:
            self._set_status("Bandwidth must be positive.", error=True)
            self.bandwidth_entry.setText(self.bandwidth_value)
            return
        self.bandwidth_value = f"{bandwidth:.0f}"
        self.bandwidth_entry.setText(self.bandwidth_value)
        freqs = self._current_target_frequencies()
        target = freqs[0] if freqs else (self.selection.center_freq if self.selection else None)
        if self.span_controller and target:
            self.span_controller.set_selection(target, bandwidth)
        elif target is not None:
            self.selection = SelectionResult(target, bandwidth)
            if self.center_freq is not None and self.offset_label:
                offset = target - self.center_freq
                self.offset_label.setText(f"Offset: {offset:+.0f} Hz")
        elif self.selection:
            self.selection = SelectionResult(self.selection.center_freq, bandwidth)
        self.base_kwargs["bandwidth"] = bandwidth
        self._set_status(
            "Bandwidth updated. Use Preview DSP or Refresh preview to apply.",
            error=False,
        )
        self._update_output_path_hint()

    def _current_target_frequencies(self) -> list[float]:
        freqs: list[float] = []
        for idx, value in enumerate(self.target_entries):
            text = value.text().strip()
            freq = self._parse_float(text)
            self.target_freq_values[idx] = text
            if freq is None or freq <= 0:
                continue
            if any(math.isclose(freq, other, rel_tol=0.0, abs_tol=0.5) for other in freqs):
                continue
            freqs.append(freq)
        return freqs[:MAX_TARGET_FREQUENCIES]

    def _update_target_freq_state(self) -> list[float]:
        freqs = self._current_target_frequencies()
        if freqs:
            primary = freqs[0]
            self.base_kwargs["target_freq"] = primary
            self.base_kwargs["target_freqs"] = freqs
            if self.center_freq is not None and self.offset_label:
                offset = primary - self.center_freq
                self.offset_label.setText(f"Offset: {offset:+.0f} Hz")
            elif self.offset_label:
                self.offset_label.setText("Offset: —")
        else:
            self.base_kwargs["target_freq"] = 0.0
            self.base_kwargs["target_freqs"] = []
            if self.offset_label:
                self.offset_label.setText("Offset: —")
        return freqs

    def _on_target_entry_edit(self, index: int, text: str) -> None:
        if index < len(self.target_freq_values):
            self.target_freq_values[index] = text.strip()
        freqs = self._update_target_freq_state()
        if not freqs:
            self._set_status(
                "Enter at least one target frequency before running DSP.",
                error=True,
            )
            return
        primary = freqs[0]
        if self.selection is None:
            bandwidth = self.base_kwargs.get("bandwidth", 12_500.0)
            self.selection = SelectionResult(primary, bandwidth)
        if self.span_controller:
            self.span_controller.set_selection(primary, self.selection.bandwidth)
        self._update_option_state()

    def _on_center_manual(self) -> None:
        if not self.center_entry:
            return
        text = self.center_entry.text().strip()
        self.center_value = text
        value = self._parse_float(text)
        self._set_center_source("manual")
        if value is not None and value > 0:
            self.base_kwargs["center_freq"] = value
            self.center_freq = value
            self.center_entry.setText(f"{value:.0f}")
            freqs = self._update_target_freq_state()
            primary = freqs[0] if freqs else (self.selection.center_freq if self.selection else None)
            if primary and self.span_controller:
                self.span_controller.set_selection(primary, self.selection.bandwidth if self.selection else self.base_kwargs.get("bandwidth", 12_500.0))
        else:
            self._set_status("Enter a valid center frequency (Hz).", error=True)

    def _parse_center_from_name(self, *, silent: bool = False) -> None:
        if not self.file_value:
            if not silent:
                self._set_status("Browse to a recording before parsing.", error=True)
            return
        path = Path(self.file_value)
        detection = detect_center_frequency(path)
        if detection.value is None:
            self._set_center_source("unavailable")
            if not silent:
                QMessageBox.information(
                    self,
                    "Center frequency",
                    "Could not derive a center frequency from WAV metadata or filename. Enter it manually.",
                )
                self._set_status("Enter center frequency manually.", error=True)
            return
        self.center_freq = detection.value
        self.center_value = f"{detection.value:.0f}"
        if self.center_entry:
            self.center_entry.setText(self.center_value)
        self.base_kwargs["center_freq"] = detection.value
        self._set_center_source(detection.source or "filename")
        friendly = self._describe_center_source(detection.source)
        self._set_status(f"Center frequency populated from {friendly}.", error=False)
        self._update_target_freq_state()

    def _update_option_state(self) -> None:
        demod = (self.demod_value or "nfm").lower()
        self.base_kwargs["demod_mode"] = demod
        is_ssb = demod in {"usb", "lsb", "ssb"}
        if self.agc_check:
            self.agc_check.blockSignals(True)
            self.agc_check.setEnabled(is_ssb)
            if is_ssb:
                if not self.agc_enabled:
                    self.agc_enabled = self._preferred_agc
                self.agc_check.setChecked(self.agc_enabled)
            else:
                self.agc_enabled = False
                self.agc_check.setChecked(False)
            self.agc_check.blockSignals(False)
        else:
            self.agc_enabled = self.agc_enabled if is_ssb else False
        if self.squelch_check:
            self.squelch_check.blockSignals(True)
            self.squelch_check.setChecked(self.squelch_enabled)
            self.squelch_check.blockSignals(False)

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None

    @staticmethod
    def _parse_int(text: str, *, default: int) -> int:
        try:
            value = int(text.strip())
        except Exception:
            return default
        return max(1024, value)

    @staticmethod
    def _format_float(value: Optional[float]) -> str:
        if value is None or value <= 0:
            return ""
        return f"{value:.0f}"

    def _current_nfft(self) -> int:
        text = (
            self.spectrum_nfft_combo.currentText()
            if self.spectrum_nfft_combo
            else self.nfft_value
        )
        value = self._parse_int(text, default=262_144)
        self.nfft_value = f"{value}"
        return value

    def _current_smoothing(self) -> int:
        if self.spectrum_smooth_spin:
            self.smooth_value = int(self.spectrum_smooth_spin.value())
        return int(self.smooth_value)

    def _current_dynamic_range(self) -> int:
        if self.spectrum_range_spin:
            self.range_value = int(self.spectrum_range_spin.value())
        return int(self.range_value)

    def _current_waterfall_slices(self) -> int:
        if self.waterfall_slices_spin:
            self.waterfall_slices_value = str(int(self.waterfall_slices_spin.value()))
        return int(self.waterfall_slices_value or "400")

    def _current_waterfall_floor(self) -> float:
        if self.waterfall_floor_spin:
            self.waterfall_floor_value = str(int(self.waterfall_floor_spin.value()))
        return float(self.waterfall_floor_value or "110")

    def _current_waterfall_cmap(self) -> str:
        if self.waterfall_cmap_combo:
            self.waterfall_cmap_value = self.waterfall_cmap_combo.currentText()
        return self.waterfall_cmap_value or "magma"

    def _schedule_preview(self, *, auto: bool) -> None:
        if not self.file_value:
            if not auto:
                self._set_status("Select an input recording first.", error=True)
            return
        path = Path(self.file_value)
        previous_path = self.selected_path
        if not path.exists():
            self._set_status(f"File not found: {path}", error=True)
            if not auto:
                QMessageBox.critical(
                    self,
                    "Preview failed",
                    f"File not found: {path}",
                )
            return

        if self.snapshot_entry:
            entered = self.snapshot_entry.text().strip()
        else:
            entered = self.snapshot_text
        if entered:
            self.snapshot_text = entered
        snapshot_seconds = self._parse_float(self.snapshot_text)
        full_capture = self.full_snapshot
        if not full_capture:
            if snapshot_seconds is None or snapshot_seconds <= 0:
                self._set_status("Snapshot duration must be a positive number.", error=True)
                if not auto:
                    QMessageBox.critical(
                        self,
                        "Invalid snapshot",
                        "Snapshot duration must be a positive number of seconds.",
                    )
                if self.snapshot_entry:
                    self.snapshot_entry.setText(self.snapshot_text or f"{self.snapshot_seconds:.2f}")
                return
        else:
            snapshot_seconds = self.snapshot_seconds if self.snapshot_seconds > 0 else 2.0
        if snapshot_seconds is None or snapshot_seconds <= 0:
            snapshot_seconds = 2.0
        self.snapshot_seconds = snapshot_seconds
        self.snapshot_text = f"{snapshot_seconds:.2f}"
        if self.snapshot_entry:
            self.snapshot_entry.setText(self.snapshot_text)

        center_override = self._parse_float(self.center_value)
        if center_override is None and self.center_entry:
            center_override = self._parse_float(self.center_entry.text())

        try:
            configs = self._build_configs(path, center_override, require_targets=False)
            if not configs:
                raise ValueError("No target frequencies configured for preview.")
            config = configs[0]
        except Exception as exc:
            LOG.error("Preview setup failed: %s", exc)
            self._set_status(f"Preview setup failed: {exc}", error=True)
            if not auto:
                QMessageBox.critical(self, "Preview failed", str(exc))
            return

        nfft = self._current_nfft()
        hop = max(1, nfft // 4)
        max_slices = self._current_waterfall_slices()
        fft_workers = self._fft_worker_count()
        status_msg = (
            "Computing full-record FFT/waterfall…"
            if full_capture
            else f"Gathering {snapshot_seconds:.2f} s FFT snapshot…"
        )
        self._set_status(status_msg, error=False)
        self._start_snapshot_thread(
            config=config,
            path=path,
            previous_path=previous_path,
            auto=auto,
            full_capture=full_capture,
            snapshot_seconds=float(snapshot_seconds),
            nfft=nfft,
            hop=hop,
            max_slices=max_slices,
            fft_workers=fft_workers,
            max_in_memory_samples=MAX_PREVIEW_SAMPLES,
        )

    def _start_snapshot_thread(
        self,
        *,
        config: ProcessingConfig,
        path: Path,
        previous_path: Optional[Path],
        auto: bool,
        full_capture: bool,
        snapshot_seconds: float,
        nfft: int,
        hop: int,
        max_slices: int,
        fft_workers: Optional[int],
        max_in_memory_samples: int,
    ) -> None:
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._set_status("Preview already loading; please wait.", error=False)
            return
        if self.load_preview_button:
            self.load_preview_button.setEnabled(False)
        if full_capture:
            self.status_update_signal.emit(
                "Analyzing entire recording… this may take a moment.", False
            )
        else:
            self.status_update_signal.emit(
                f"Preparing preview (~{snapshot_seconds:.2f} s)…", False
            )

        def worker() -> None:
            try:
                if full_capture:
                    snapshot = self._compute_full_psd(
                        config,
                        nfft=nfft,
                        hop=hop,
                        max_slices=max_slices,
                        fft_workers=fft_workers,
                        status_cb=lambda msg: self.status_update_signal.emit(msg, False),
                    )
                else:
                    snapshot = gather_snapshot(
                        config,
                        seconds=snapshot_seconds,
                        nfft=nfft,
                        hop=hop,
                        max_slices=max_slices,
                        fft_workers=fft_workers,
                        max_in_memory_samples=max_in_memory_samples,
                        progress_cb=lambda seconds_done, frac: self.status_update_signal.emit(
                            f"Gathered {seconds_done:.1f} s of preview ({frac * 100:4.1f}%)…",
                            False,
                        ),
                    )
            except Exception as exc:
                LOG.error("Failed to gather preview: %s", exc)
                self.snapshot_failed_signal.emit(exc, auto)
            else:
                self.snapshot_ready_signal.emit(snapshot, path, previous_path)
            finally:
                self.snapshot_finished_signal.emit()

        self._snapshot_thread = threading.Thread(target=worker, daemon=True)
        self._snapshot_thread.start()

    @Slot(object, object, object)
    def _on_snapshot_ready(
        self,
        snapshot: SnapshotData,
        path: Path,
        previous_path: Optional[Path],
    ) -> None:
        same_file = previous_path is not None and Path(previous_path) == path
        self.selected_path = path
        self.snapshot_data = snapshot
        self.center_freq = snapshot.center_freq
        self.probe = snapshot.probe
        self.sample_rate = snapshot.sample_rate
        self.snapshot_seconds = snapshot.seconds
        self.center_value = f"{snapshot.center_freq:.0f}"
        if self.center_entry:
            self.center_entry.setText(self.center_value)
        self.base_kwargs["center_freq"] = snapshot.center_freq
        self.full_snapshot = bool(snapshot.params.get("full_capture", False))
        if self.full_snapshot_check:
            self.full_snapshot_check.blockSignals(True)
            self.full_snapshot_check.setChecked(self.full_snapshot)
            self.full_snapshot_check.blockSignals(False)
        if self.sample_rate_label:
            self.sample_rate_label.setText(
                f"Sample rate: {snapshot.sample_rate:,.0f} Hz"
            )
        if not same_file:
            self.selection = None
            if self.confirm_btn:
                self.confirm_btn.setEnabled(False)
            if self.preview_btn:
                self.preview_btn.setEnabled(False)
        self._update_output_path_hint()
        self._render_snapshot(snapshot, remember=True)
        self._set_status(
            "Drag over the channel of interest, then Confirm && Run.",
            error=False,
        )

    @Slot(Exception, bool)
    def _on_snapshot_failed(self, error: Exception, auto: bool) -> None:
        self._set_status(f"Preview failed: {error}", error=True)
        if not auto:
            QMessageBox.critical(self, "Preview failed", str(error))

    @Slot()
    def _on_snapshot_thread_finished(self) -> None:
        self._snapshot_thread = None
        if self.load_preview_button:
            self.load_preview_button.setEnabled(True)
        if self.snapshot_data and self.preview_btn:
            self.preview_btn.setEnabled(True)

    def _render_snapshot(self, snapshot: SnapshotData, *, remember: bool) -> None:
        precomputed = (snapshot.freqs, snapshot.psd_db)
        self._render_plot(
            snapshot.samples,
            snapshot.sample_rate,
            snapshot.center_freq,
            remember=remember,
            snapshot=snapshot,
            precomputed=precomputed,
            waterfall=snapshot.waterfall,
        )

    def _render_plot(
        self,
        samples: Optional[np.ndarray],
        sample_rate: float,
        center_freq: float,
        *,
        remember: bool,
        snapshot: Optional[SnapshotData],
        precomputed: Optional[tuple[np.ndarray, np.ndarray]],
        waterfall: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        layout = self.plot_layout
        if layout is None:
            return
        if self.placeholder_label:
            self.placeholder_label.deleteLater()
            self.placeholder_label = None
        if self.toolbar:
            self.toolbar.setParent(None)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.canvas:
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

        nfft = self._current_nfft()
        smooth = max(1, int(self._current_smoothing()))
        if precomputed is not None:
            freqs, psd_db = precomputed
        else:
            if samples is None:
                raise ValueError("Samples must be provided when no precomputed PSD is supplied.")
            freqs, psd_db = compute_psd(
                samples,
                sample_rate,
                nfft=nfft,
                fft_workers=self._fft_worker_count(),
            )
        psd_db = np.asarray(psd_db, dtype=np.float64)
        if smooth > 1 and psd_db.size >= smooth:
            kernel = np.ones(smooth) / smooth
            psd_db = np.convolve(psd_db, kernel, mode="same")
        abs_freqs = center_freq + freqs
        self._freq_min_hz = float(np.min(abs_freqs))
        self._freq_max_hz = float(np.max(abs_freqs))

        self.figure = Figure(figsize=(9.5, 5.2))
        ax = self.figure.add_subplot(111)
        theme = self.color_themes.get(self.theme_value, self.color_themes["default"])
        line_color = theme["line"]
        ax.plot(abs_freqs, psd_db, lw=0.9, color=line_color)
        ax.set_title(
            "Drag to highlight a channel. Scroll or double-click to zoom. "
            "Use Preview/Confirm buttons below."
        )
        if FuncFormatter is not None:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _pos: f"{x/1e6:.3f}")
            )
        ax.set_xlabel("Absolute frequency (MHz)")
        ax.set_ylabel("Power (dBFS/Hz)")
        ax.grid(
            True,
            which="both",
            ls=theme.get("grid", ":"),
            color=theme.get("grid_color", "#cccccc"),
        )
        ax.set_facecolor(theme.get("bg", "white"))
        self.figure.patch.set_facecolor(theme.get("face", "white"))
        for spine in ax.spines.values():
            spine.set_color(theme.get("fg", "black"))
        ax.tick_params(colors=theme.get("fg", "black"))
        ax.xaxis.label.set_color(theme.get("fg", "black"))
        ax.yaxis.label.set_color(theme.get("fg", "black"))
        ax.title.set_color(theme.get("fg", "black"))

        dynamic = max(20.0, float(self._current_dynamic_range()))
        if psd_db.size:
            finite_vals = psd_db[np.isfinite(psd_db)]
            peak = float(np.max(finite_vals)) if finite_vals.size else 0.0
        else:
            peak = 0.0
        ax.set_ylim(peak - dynamic, peak + 5.0)

        self.ax_main = ax
        self.canvas = FigureCanvasQTAgg(self.figure)
        if NavigationToolbar2QT is not None:
            self.toolbar = NavigationToolbar2QT(self.canvas, self.plot_group)
            self.toolbar.setObjectName("spectrumToolbar")
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.canvas.draw()

        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("key_press_event", self._on_canvas_key)
        self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)

        initial_center = (
            self.selection.center_freq
            if self.selection and self.selected_path
            else self.base_kwargs.get("target_freq", 0.0)
        )
        if not initial_center or initial_center <= 0:
            initial_center = center_freq
        initial_bw = (
            self.selection.bandwidth
            if self.selection and self.selected_path
            else self.base_kwargs.get("bandwidth", 12_500.0)
        )
        initial_bw = max(initial_bw, 100.0)

        self.span_controller = _SpanController(
            ax=ax,
            canvas=self.canvas,
            initial=SelectionResult(initial_center, initial_bw),
            on_change=self._on_span_change,
        )
        self._on_span_change(self.span_controller.selection)

        if snapshot is not None and not remember:
            self.snapshot_data = snapshot
        if self.bandwidth_entry and self.selection:
            self.bandwidth_value = f"{self.selection.bandwidth:.0f}"
            self.bandwidth_entry.setText(self.bandwidth_value)
        if self.sample_rate_label:
            self.sample_rate_label.setText(
                f"Sample rate: {sample_rate:,.0f} Hz"
            )
        if waterfall is not None:
            self._update_waterfall_display(sample_rate, center_freq, waterfall)
        elif snapshot and snapshot.waterfall is not None:
            self._update_waterfall_display(sample_rate, center_freq, snapshot.waterfall)

        if self.preview_btn:
            self.preview_btn.setEnabled(True)
        if self.confirm_btn and self.selection is not None:
            self.confirm_btn.setEnabled(True)

    def _update_waterfall_display(
        self,
        sample_rate: float,
        center_freq: float,
        waterfall: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        if waterfall is None:
            return
        freqs, times, matrix = waterfall
        if matrix.size == 0:
            return
        if self.waterfall_window is None or not self.waterfall_window.alive:
            self.waterfall_window = _WaterfallWindow(
                self,
                on_select=self._on_waterfall_pick,
                on_close=self._on_waterfall_closed,
            )
            self.waterfall_window.show()
        floor = float(self._current_waterfall_floor())
        cmap = self._current_waterfall_cmap()
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
        bandwidth = (
            self.selection.bandwidth if self.selection is not None else 12_500.0
        )
        self.span_controller.set_selection(freq_hz, bandwidth)

    def _on_waterfall_closed(self) -> None:
        self.waterfall_window = None

    def _schedule_refresh(self, *, full: bool = False, waterfall_only: bool = False) -> None:
        if self.snapshot_data is None:
            self._set_status("Load an FFT snapshot before refreshing.", error=True)
            return
        self._refresh_pending = (full, waterfall_only)
        if self._refresh_timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_refresh)
            self._refresh_timer = timer
        else:
            self._refresh_timer.stop()
        self._refresh_timer.start(0)

    def _run_refresh(self) -> None:
        if self._refresh_timer is None:
            return
        full, waterfall_only = self._refresh_pending
        self._refresh_timer.stop()
        self._refresh_plot(full=full, waterfall_only=waterfall_only)

    def _refresh_preview_manual(self) -> None:
        if self.snapshot_data is None:
            self._set_status("Load an FFT snapshot before refreshing.", error=True)
            return
        full_capture = bool(self.snapshot_data.params.get("full_capture", False))
        if self.full_snapshot_check:
            self.full_snapshot_check.blockSignals(True)
            self.full_snapshot_check.setChecked(full_capture)
            self.full_snapshot_check.blockSignals(False)
        self._set_status("Refreshing preview…", error=False)
        self._schedule_refresh(full=True)

    def _refresh_plot(self, *, full: bool = False, waterfall_only: bool = False) -> None:
        snapshot = self.snapshot_data
        if snapshot is None:
            return
        if not full and snapshot.waterfall is not None and waterfall_only:
            self._update_waterfall_display(
                snapshot.sample_rate, snapshot.center_freq, snapshot.waterfall
            )
            return

        nfft = self._current_nfft()
        hop = max(1, nfft // 4)
        max_slices = self._current_waterfall_slices()
        fft_workers = self._fft_worker_count()

        try:
            if snapshot.mode == "samples" and snapshot.samples is not None:
                freqs, psd, waterfall_res, frames = streaming_waterfall(
                    [snapshot.samples],
                    snapshot.sample_rate,
                    nfft=nfft,
                    hop=hop,
                    max_slices=max_slices,
                    fft_workers=fft_workers,
                )
                refreshed = SnapshotData(
                    path=snapshot.path,
                    sample_rate=snapshot.sample_rate,
                    center_freq=snapshot.center_freq,
                    probe=snapshot.probe,
                    seconds=snapshot.seconds,
                    mode="samples",
                    freqs=freqs,
                    psd_db=psd,
                    waterfall=_waterfall_to_tuple(waterfall_res),
                    samples=snapshot.samples,
                    params={
                        **snapshot.params,
                        "nfft": nfft,
                        "hop": hop,
                        "max_slices": max_slices,
                        "fft_workers": fft_workers,
                    },
                    fft_frames=frames,
                )
            else:
                configs = self._build_configs(
                    snapshot.path,
                    snapshot.center_freq,
                    require_targets=False,
                )
                if not configs:
                    raise ValueError("Enter at least one target frequency before refreshing.")
                config = configs[0]
                seconds = float(snapshot.params.get("seconds", snapshot.seconds))
                max_samples = int(
                    snapshot.params.get("max_in_memory_samples", MAX_PREVIEW_SAMPLES)
                )
                full_capture_flag = bool(snapshot.params.get("full_capture", False))
                message = (
                    "Recomputing full-record preview…"
                    if full_capture_flag
                    else "Refreshing snapshot preview…"
                )
                self._set_status(message, error=False)
                self._start_snapshot_thread(
                    config=config,
                    path=snapshot.path,
                    previous_path=snapshot.path,
                    auto=False,
                    full_capture=full_capture_flag,
                    snapshot_seconds=seconds,
                    nfft=nfft,
                    hop=hop,
                    max_slices=max_slices,
                    fft_workers=fft_workers,
                    max_in_memory_samples=max_samples,
                )
                return
        except Exception as exc:
            LOG.error("Preview refresh failed: %s", exc)
            self._set_status(f"Preview refresh failed: {exc}", error=True)
            return

        self.snapshot_data = refreshed
        self.snapshot_seconds = refreshed.seconds
        self._render_plot(
            refreshed.samples,
            refreshed.sample_rate,
            refreshed.center_freq,
            remember=True,
            snapshot=refreshed,
            precomputed=(refreshed.freqs, refreshed.psd_db),
            waterfall=refreshed.waterfall,
        )

    def _on_span_change(self, selection: SelectionResult) -> None:
        self.selection = selection
        if self.target_entries:
            entry = self.target_entries[0]
            entry.blockSignals(True)
            entry.setText(f"{selection.center_freq:.0f}")
            entry.blockSignals(False)
        freqs = self._update_target_freq_state()
        self.bandwidth_value = f"{selection.bandwidth:.0f}"
        if self.bandwidth_entry:
            self.bandwidth_entry.setText(self.bandwidth_value)
        if self.center_freq is not None and freqs:
            offset = freqs[0] - self.center_freq
            if self.offset_label:
                self.offset_label.setText(f"Offset: {offset:+.0f} Hz")
        elif self.offset_label:
            self.offset_label.setText("Offset: —")
        if self.confirm_btn:
            self.confirm_btn.setEnabled(True)
        if self.preview_btn:
            self.preview_btn.setEnabled(True)
        self._update_output_path_hint()

    def _on_canvas_click(self, event) -> None:
        if self.ax_main is None or event.inaxes != self.ax_main or event.xdata is None:
            return
        if event.dblclick:
            if (
                self.selection
                and abs(event.xdata - self.selection.center_freq)
                <= self.selection.bandwidth / 2.0
            ):
                self._zoom_to_selection()
            else:
                self._zoom_at(event.xdata, factor=0.5)

    def _on_canvas_key(self, event) -> None:
        if event.key == "escape":
            self._on_cancel()

    def _on_canvas_scroll(self, event) -> None:
        if self.ax_main is None or event.xdata is None:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            button = getattr(event, "button", None)
            step = 1 if button == "up" else -1
        if step > 0:
            self._zoom_at(event.xdata, factor=0.8)
        else:
            self._zoom_at(event.xdata, factor=1.25)

    def _zoom_to_selection(self) -> None:
        if self.ax_main is None or self.selection is None:
            return
        bandwidth = max(self.selection.bandwidth, 100.0)
        center = self.selection.center_freq
        self._set_xlim(center - bandwidth / 2.0, center + bandwidth / 2.0)

    def _zoom_at(self, center_hz: float, factor: float) -> None:
        if self.ax_main is None:
            return
        xmin, xmax = self.ax_main.get_xlim()
        width = xmax - xmin
        if factor <= 0:
            factor = 1.0
        new_width = width * factor
        total = None
        if self._freq_min_hz is not None and self._freq_max_hz is not None:
            total = self._freq_max_hz - self._freq_min_hz
        if total is not None:
            new_width = min(new_width, total)
        min_width = max(100.0, self.selection.bandwidth if self.selection else 100.0)
        if new_width < min_width:
            new_width = min_width
        new_min = center_hz - new_width / 2.0
        new_max = center_hz + new_width / 2.0
        self._set_xlim(new_min, new_max)

    def _set_xlim(self, xmin: float, xmax: float) -> None:
        if self.ax_main is None or self.canvas is None:
            return
        if self._freq_min_hz is not None and self._freq_max_hz is not None:
            total = self._freq_max_hz - self._freq_min_hz
            if total <= 0:
                xmin, xmax = self._freq_min_hz, self._freq_max_hz
            else:
                width = xmax - xmin
                if width >= total:
                    xmin, xmax = self._freq_min_hz, self._freq_max_hz
                else:
                    if xmin < self._freq_min_hz:
                        xmin = self._freq_min_hz
                    if xmax > self._freq_max_hz:
                        xmax = self._freq_max_hz
        self.ax_main.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def _set_stop_enabled(self, enabled: bool) -> None:
        if self.stop_btn:
            self.stop_btn.setEnabled(enabled)

    def _on_stop_processing(self) -> None:
        if not self._preview_running:
            return
        self._set_status("Cancelling preview…", error=True)
        if self._status_sink is not None:
            try:
                self._status_sink.cancel()
            except Exception:
                pass
        self._set_stop_enabled(False)
        self._cancel_active_pipeline()

    def _register_preview_pipeline(self, pipeline) -> None:
        with self._preview_lock:
            self._active_pipeline = pipeline

    def _ensure_output_directory(self, output_path: Optional[Path]) -> None:
        if output_path is None:
            return
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOG.warning(
                "Failed to create output directory %s: %s",
                output_path.parent,
                exc,
            )

    def _on_preview(self) -> None:
        if self._preview_thread and self._preview_thread.is_alive():
            self._set_status(
                "Preview already running. Cancel it before starting a new one.",
                error=True,
            )
            return
        if not self.selection or not self.selected_path:
            self._set_status("Select a frequency span before previewing.", error=True)
            return
        seconds = self.snapshot_seconds if self.snapshot_seconds > 0 else 2.0
        try:
            configs = self._build_configs(self.selected_path, self.center_freq)
            if not configs:
                raise ValueError("Enter at least one target frequency before previewing.")
        except Exception as exc:
            LOG.error("Preview setup failed: %s", exc)
            self._set_status(f"Preview setup failed: {exc}", error=True)
            QMessageBox.critical(self, "Preview failed", str(exc))
            return

        config = configs[0]
        total = len(configs)
        for cfg in configs:
            self._ensure_output_directory(cfg.output_path)
        self._preview_running = True
        sink = StatusProgressSink(
            lambda message, highlight: self.status_update_signal.emit(
                message, highlight
            )
        )
        self._status_sink = sink
        if self.preview_btn:
            self.preview_btn.setEnabled(False)
        if self.confirm_btn:
            self.confirm_btn.setEnabled(False)
        self._set_stop_enabled(True)
        if total > 1:
            self._set_status(f"Previewing {seconds:.2f} s… (1/{total})", error=True)
        else:
            self._set_status(f"Previewing {seconds:.2f} s…", error=True)

        def worker() -> None:
            outputs: list[Path] = []
            try:
                for index, cfg in enumerate(configs, start=1):
                    if total > 1:
                        sink.status(f"Preview {index}/{total}")
                    try:
                        _result, preview_path = run_preview(
                            cfg,
                            seconds,
                            progress_sink=sink,
                            on_pipeline=self._register_preview_pipeline,
                        )
                    except ProcessingCancelled:
                        raise
                    except Exception as exc:
                        LOG.error(
                            "Preview failed for %.0f Hz: %s", cfg.target_freq, exc
                        )
                        raise
                    else:
                        outputs.append(preview_path)
            except ProcessingCancelled:
                LOG.info("Preview cancelled by user.")
                self.preview_complete_signal.emit(None, True)
            except Exception as exc:
                if not outputs:
                    LOG.error("Preview failed: %s", exc)
                self.preview_complete_signal.emit(exc, True)
            else:
                self.preview_complete_signal.emit(outputs, False)
            finally:
                self._register_preview_pipeline(None)

        self._preview_thread = threading.Thread(target=worker, daemon=True)
        self._preview_thread.start()

    @Slot(object, bool)
    def _on_preview_completed(self, payload, is_error: bool) -> None:
        if isinstance(payload, Exception):
            self._handle_preview_failed(payload)
            return
        if is_error:
            if payload is None:
                self._handle_preview_cancelled()
            elif isinstance(payload, Exception):
                self._handle_preview_failed(payload)
            else:
                self._handle_preview_failed(Exception("Preview failed"))
            return
        if isinstance(payload, list):
            self._handle_preview_complete(payload)
        else:
            self._handle_preview_cancelled()

    def _preview_finished(self) -> None:
        self._preview_running = False
        self._preview_thread = None
        with self._preview_lock:
            self._active_pipeline = None
        self._set_stop_enabled(False)
        self._status_sink = None
        if self.preview_btn:
            self.preview_btn.setEnabled(True)
        if self.confirm_btn:
            self.confirm_btn.setEnabled(self.selection is not None)

    def _handle_preview_complete(self, preview_paths: list[Path]) -> None:
        self._preview_finished()
        if not preview_paths:
            self._set_status("Preview complete.", error=False)
            return
        if len(preview_paths) == 1:
            summary = f"Preview complete (output: {preview_paths[0].name})"
            detail = str(preview_paths[0])
        else:
            summary = f"Preview complete for {len(preview_paths)} targets."
            detail = "\n".join(
                f"{idx + 1}. {path}" for idx, path in enumerate(preview_paths)
            )
        self._set_status(summary, error=False)
        QMessageBox.information(
            self,
            "Preview complete",
            f"Preview audio written to:\n{detail}",
        )

    def _handle_preview_failed(self, error: Exception) -> None:
        self._preview_finished()
        self._set_status(f"Preview failed: {error}", error=True)
        QMessageBox.critical(self, "Preview failed", str(error))

    def _handle_preview_cancelled(self) -> None:
        self._preview_finished()
        self._set_status("Preview cancelled.", error=False)

    def _on_cancel(self) -> None:
        self._cancel_active_pipeline()
        self._set_stop_enabled(False)
        self.selection = None
        app = QApplication.instance()
        if app:
            app.quit()

    def _on_confirm(self) -> None:
        if not self.selection or not self.selected_path:
            self._set_status("Select a frequency span before confirming.", error=True)
            return
        self.progress_sink = None
        app = QApplication.instance()
        if app:
            app.quit()

    @staticmethod
    def _fft_worker_count() -> Optional[int]:
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 1:
            return None
        return min(4, cpu_count)

    @staticmethod
    def _augment_path(path: Optional[Path], freq: float, total: int) -> Optional[Path]:
        if path is None or total <= 1:
            return path
        freq_tag = int(round(freq))
        return path.with_name(f"{path.stem}_{freq_tag}{path.suffix}")

    def _output_path_for_frequency(
        self, input_path: Path, target_freq: float, total: int
    ) -> Optional[Path]:
        base = self._resolve_output_path(input_path, target_freq)
        if total <= 1:
            return base
        if self._current_output_dir() is not None:
            return base
        if base is None:
            return None
        freq_tag = int(round(target_freq))
        return base.with_name(f"{base.stem}_{freq_tag}{base.suffix}")

    def _build_configs(
        self,
        path: Path,
        center_override: Optional[float],
        *,
        require_targets: bool = True,
    ) -> list[ProcessingConfig]:
        kwargs = dict(self.base_kwargs)
        for deprecated_key in ("squelch_dbfs", "silence_trim", "squelch_enabled", "target_freqs"):
            kwargs.pop(deprecated_key, None)
        center_value = center_override
        if center_value is None or center_value <= 0:
            source_text = (
                self.center_entry.text() if self.center_entry else self.center_value
            )
            parsed = self._parse_float(source_text or "")
            if parsed is not None and parsed > 0:
                center_value = parsed
        kwargs["center_freq"] = center_value
        kwargs["center_freq_source"] = self.center_source
        demod = (self.demod_value or kwargs.get("demod_mode", "nfm")).lower()
        kwargs["demod_mode"] = demod
        silence_trim = self.trim_enabled
        squelch_enabled = self.squelch_enabled
        kwargs["agc_enabled"] = self.agc_enabled
        kwargs["silence_trim"] = silence_trim
        kwargs["squelch_enabled"] = squelch_enabled
        self.base_kwargs.update(
            {
                "center_freq": center_value,
                "demod_mode": demod,
                "silence_trim": silence_trim,
                "squelch_enabled": squelch_enabled,
                "agc_enabled": kwargs["agc_enabled"],
            }
        )
        if center_value is not None and center_value > 0:
            self.center_freq = center_value
        bandwidth = (
            self.selection.bandwidth
            if self.selection is not None
            else self.base_kwargs.get("bandwidth", kwargs.get("bandwidth", 12_500.0))
        )
        kwargs["bandwidth"] = bandwidth
        self.base_kwargs["bandwidth"] = bandwidth
        self.base_kwargs["center_freq_source"] = self.center_source
        frequencies = self._update_target_freq_state()
        if not frequencies:
            fallback = (
                self.selection.center_freq
                if self.selection is not None
                else self.base_kwargs.get("target_freq", kwargs.get("target_freq", 0.0))
            )
            if fallback and fallback > 0:
                frequencies = [float(fallback)]
                if self.target_entries:
                    entry = self.target_entries[0]
                    entry.blockSignals(True)
                    entry.setText(f"{float(fallback):.0f}")
                    entry.blockSignals(False)
                self._update_target_freq_state()
            elif require_targets:
                raise ValueError("Enter at least one target frequency before running DSP.")
            else:
                frequencies = [max(float(fallback), 0.0)]
        total = len(frequencies)
        base_dump = kwargs.get("dump_iq_path")
        base_plot = kwargs.get("plot_stages_path")
        configs: list[ProcessingConfig] = []
        for freq in frequencies:
            freq_kwargs = dict(kwargs)
            freq_kwargs["target_freq"] = freq
            freq_kwargs["output_path"] = self._output_path_for_frequency(path, freq, total)
            freq_kwargs["dump_iq_path"] = self._augment_path(base_dump, freq, total)
            freq_kwargs["plot_stages_path"] = self._augment_path(base_plot, freq, total)
            config = ProcessingConfig(in_path=path, **freq_kwargs)
            configs.append(config)
        if configs:
            self.base_kwargs["target_freq"] = configs[0].target_freq
            if configs[0].output_path is not None:
                self.base_kwargs["output_path"] = configs[0].output_path
            elif "output_path" in self.base_kwargs:
                self.base_kwargs.pop("output_path")
        else:
            self.base_kwargs["target_freq"] = 0.0
            self.base_kwargs.pop("output_path", None)
        self._update_output_path_hint()
        return configs

    def _compute_full_psd(
        self,
        config: ProcessingConfig,
        *,
        nfft: int,
        hop: int,
        max_slices: int,
        fft_workers: Optional[int],
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> SnapshotData:
        probe = probe_sample_rate(config.in_path)
        sample_rate = probe.value
        center_freq = config.center_freq
        if center_freq is None:
            detection = detect_center_frequency(config.in_path)
            if detection.value is None:
                raise ValueError(
                    "Center frequency not provided and could not be inferred from WAV metadata or filename. Enter a value before using full-record preview."
                )
            center_freq = detection.value

        chunk_samples = max(config.chunk_size, nfft)
        consumed = 0
        try:
            file_size = config.in_path.stat().st_size
        except OSError:
            file_size = 0
        header_bytes = 44
        frame_bytes = 4
        payload_bytes = max(file_size - header_bytes, 0)
        estimated_total_samples = (
            payload_bytes // frame_bytes if payload_bytes > 0 else 0
        )
        estimated_chunks = (
            int(math.ceil(estimated_total_samples / chunk_samples))
            if estimated_total_samples > 0
            else 0
        )
        status_stride = max(1, estimated_chunks // 25) if estimated_chunks else 4

        from .processing import IQReader

        if status_cb:
            try:
                status_cb("Reading full recording for spectrum analysis…")
            except Exception:
                pass

        def _chunk_iter() -> Iterator[np.ndarray]:
            nonlocal consumed
            chunk_index = 0
            with IQReader(config.in_path, chunk_samples, config.iq_order) as reader:
                for block in reader:
                    if block is None or block.size == 0:
                        break
                    consumed += block.size
                    chunk_index += 1
                    if status_cb and (
                        chunk_index == 1
                        or chunk_index % status_stride == 0
                        or (estimated_chunks and chunk_index >= estimated_chunks)
                    ):
                        try:
                            if estimated_chunks:
                                pct = min(chunk_index / estimated_chunks, 1.0) * 100.0
                                seconds_done = (
                                    consumed / sample_rate if sample_rate > 0 else 0.0
                                )
                                total_seconds = (
                                    estimated_total_samples / sample_rate
                                    if sample_rate > 0
                                    else 0.0
                                )
                                status_cb(
                                    f"Averaging PSD chunk {chunk_index}/{estimated_chunks} "
                                    f"({pct:4.1f}% ≈ {seconds_done:.1f}s/{total_seconds:.1f}s)"
                                )
                            else:
                                status_cb(f"Averaging PSD chunk {chunk_index}…")
                        except Exception:
                            pass
                    yield block

        freqs, avg_psd, waterfall, frames = streaming_waterfall(
            _chunk_iter(),
            sample_rate,
            nfft=nfft,
            hop=hop,
            max_slices=max_slices,
            fft_workers=fft_workers,
        )

        params: Dict[str, Any] = {
            "nfft": nfft,
            "hop": hop,
            "max_slices": max_slices,
            "fft_workers": fft_workers,
            "seconds": consumed / sample_rate if sample_rate > 0 else 0.0,
            "full_capture": True,
        }
        snapshot = SnapshotData(
            path=config.in_path,
            sample_rate=sample_rate,
            center_freq=center_freq,
            probe=probe,
            seconds=consumed / sample_rate if sample_rate > 0 else 0.0,
            mode="precomputed",
            freqs=freqs,
            psd_db=avg_psd,
            waterfall=_waterfall_to_tuple(waterfall),
            samples=None,
            params=params,
            fft_frames=frames,
        )
        return snapshot

    def _build_demod_options_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("Demod options")
        layout = QVBoxLayout()

        checks_row = QHBoxLayout()
        self.squelch_check = QCheckBox("Adaptive squelch")
        self.squelch_check.setChecked(self.squelch_enabled)
        self.squelch_check.stateChanged.connect(self._on_toggle_squelch)
        checks_row.addWidget(self.squelch_check)

        self.trim_check = QCheckBox("Trim silences")
        self.trim_check.setChecked(self.trim_enabled)
        self.trim_check.stateChanged.connect(self._on_toggle_trim)
        checks_row.addWidget(self.trim_check)

        self.agc_check = QCheckBox("Automatic gain control")
        self.agc_check.setChecked(self.agc_enabled)
        self.agc_check.stateChanged.connect(self._on_toggle_agc)
        checks_row.addWidget(self.agc_check)
        checks_row.addStretch()
        layout.addLayout(checks_row)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Manual threshold (dBFS):"))
        self.squelch_threshold_entry = QLineEdit(self.squelch_threshold_value)
        self.squelch_threshold_entry.setMaximumWidth(110)
        self.squelch_threshold_entry.editingFinished.connect(self._on_squelch_threshold_edit)
        threshold_row.addWidget(self.squelch_threshold_entry)
        threshold_row.addStretch()
        layout.addLayout(threshold_row)

        help_label = QLabel(
            "Adaptive squelch tracks the noise floor, opening when the signal rises ~4 dB above it.\n"
            "Leave the threshold blank to auto-track, or enter a negative dBFS value to pin the gate."
        )
        help_label.setStyleSheet("color: #708090;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)


def launch_interactive_session(
    *,
    input_path: Optional[Path],
    base_kwargs: dict,
    snapshot_seconds: float,
) -> InteractiveSessionResult:
    """Launch the PySide6-based interactive GUI for full-session control."""
    ensure_matplotlib()
    if QtWidgets is None:
        raise RuntimeError(QT_DEPENDENCY_HINT)
    if FigureCanvasQTAgg is None or Figure is None or SpanSelector is None:
        raise RuntimeError("matplotlib QtAgg backend is required for --interactive.")

    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication(sys.argv)

    app = _InteractiveApp(
        base_kwargs=base_kwargs,
        initial_path=input_path,
        snapshot_seconds=snapshot_seconds,
    )
    return app.run()


def interactive_select(
    config: ProcessingConfig, seconds: float = 2.0
) -> InteractiveOutcome:
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
            "squelch_dbfs": getattr(config, "squelch_dbfs", None),
            "silence_trim": getattr(config, "silence_trim", False),
            "squelch_enabled": getattr(config, "squelch_enabled", False),
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
        if detection.value is None:
            center = final_config.target_freq
        else:
            center = detection.value
    # Reuse selection info from final_config and the fresh probe.
    return InteractiveOutcome(
        center_freq=center,
        target_freq=final_config.target_freq,
        bandwidth=final_config.bandwidth,
        probe=probe,
    )
