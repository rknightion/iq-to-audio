from __future__ import annotations

import logging
import math
import queue
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .processing import ProcessingCancelled, ProcessingConfig
from .preview import run_preview
from .probe import SampleRateProbe, probe_sample_rate
from .utils import detect_center_frequency
from .visualize import SelectionResult, compute_psd, ensure_matplotlib
from .progress import PhaseState, ProgressSink

LOG = logging.getLogger(__name__)

try:  # GUI widgets are optional at runtime.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - runtime guard
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

try:  # Matplotlib embedding for Tk.
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from matplotlib.widgets import SpanSelector
    from matplotlib.ticker import FuncFormatter
except Exception:  # pragma: no cover - runtime guard
    FigureCanvasTkAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    SpanSelector = None  # type: ignore[assignment]
    NavigationToolbar2Tk = None  # type: ignore[assignment]
    FuncFormatter = None  # type: ignore[assignment]

TK_DEPENDENCY_HINT = (
    "Tkinter is required for --interactive. Install system Tk packages "
    "(e.g., `sudo apt install python3-tk` on Debian/Ubuntu, `brew install python-tk@3.14` on macOS, "
    "or `sudo pacman -S tk` on Arch)."
)

@dataclass
class InteractiveOutcome:
    center_freq: float
    target_freq: float
    bandwidth: float
    probe: SampleRateProbe


@dataclass
class InteractiveSessionResult:
    config: ProcessingConfig
    progress_sink: Optional[ProgressSink]


def gather_snapshot(
    config: ProcessingConfig, seconds: float = 2.0
) -> tuple[np.ndarray, float, float, SampleRateProbe]:
    """Read a short segment of IQ data for spectrum display."""
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

    block_samples = int(sample_rate * seconds)
    LOG.info(
        "Gathering interactive snapshot: %.2f s (~%d complex samples).",
        seconds,
        block_samples,
    )

    from .processing import IQReader  # Local import to avoid circular refs.

    with IQReader(config.in_path, block_samples, config.iq_order) as reader:
        block = reader.read_block()

    if block is None or block.size == 0:
        raise RuntimeError("Failed to read samples for interactive snapshot.")

    return block, sample_rate, center_freq, probe


def launch_interactive_session(
    *,
    input_path: Optional[Path],
    base_kwargs: dict,
    snapshot_seconds: float,
) -> InteractiveSessionResult:
    """Launch the Tk-based interactive TUI/GUI for full-session control."""
    ensure_matplotlib()
    if tk is None or ttk is None:
        raise RuntimeError(
            TK_DEPENDENCY_HINT
        )
    if FigureCanvasTkAgg is None or Figure is None or SpanSelector is None:
        raise RuntimeError(
            "matplotlib TkAgg backend is required for --interactive."
        )

    app = _InteractiveApp(
        base_kwargs=base_kwargs,
        initial_path=input_path,
        snapshot_seconds=snapshot_seconds,
    )
    return app.run()


class _InteractiveApp:
    """Tk + matplotlib workflow for selecting file, frequency span, and running."""

    def __init__(
        self,
        *,
        base_kwargs: dict,
        initial_path: Optional[Path],
        snapshot_seconds: float,
    ):
        self.base_kwargs = dict(base_kwargs)
        self.initial_path = initial_path
        self.default_snapshot = snapshot_seconds

        self.root = tk.Tk()
        self.root.title("IQ to Audio — Interactive Mode")
        self.root.geometry("1150x1210")

        self.selected_path: Optional[Path] = initial_path
        self.selection: Optional[SelectionResult] = None
        self.sample_rate: Optional[float] = None
        self.center_freq: Optional[float] = self.base_kwargs.get("center_freq")
        self.probe: Optional[SampleRateProbe] = None
        self.snapshot_seconds = max(snapshot_seconds, 0.25)

        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.span_controller: Optional[_SpanController] = None
        self.progress_sink: Optional[ProgressSink] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None
        self.plot_container: Optional[ttk.Frame] = None
        self.snapshot_data: Optional[dict[str, object]] = None
        self._refresh_job: Optional[str] = None
        self.demod_options = ("nfm", "am", "usb", "lsb", "ssb")
        self.color_themes: dict[str, dict[str, str]] = {
            "default": {"bg": "white", "face": "white", "line": "#1f77b4", "fg": "black", "grid": ":", "grid_color": "#d0d0d0"},
            "contrast": {"bg": "#101010", "face": "#101010", "line": "#ff7600", "fg": "white", "grid": "--", "grid_color": "#444444"},
            "night": {"bg": "#0b1a2a", "face": "#0b1a2a", "line": "#7fffd4", "fg": "#f0f4ff", "grid": ":", "grid_color": "#223347"},
        }
        self.ax_main = None
        self._freq_min_hz: Optional[float] = None
        self._freq_max_hz: Optional[float] = None

        self.file_var = tk.StringVar(
            value=str(initial_path) if initial_path else ""
        )
        self.center_var = tk.StringVar(
            value=self._format_float(self.base_kwargs.get("center_freq"))
        )
        self.center_source_var = tk.StringVar(value="Center source: —")
        self.center_source: str = "unavailable"
        self.snapshot_var = tk.StringVar(
            value=f"{self.snapshot_seconds:.2f}"
        )
        self.demod_var = tk.StringVar(
            value=(self.base_kwargs.get("demod_mode") or "nfm").lower()
        )
        self.squelch_var = tk.BooleanVar(
            value=self.base_kwargs.get("squelch_enabled", True)
        )
        self.trim_var = tk.BooleanVar(
            value=self.base_kwargs.get("silence_trim", False)
        )
        self.agc_var = tk.BooleanVar(
            value=self.base_kwargs.get("agc_enabled", True)
        )
        self._preferred_agc = self.agc_var.get()
        threshold = self.base_kwargs.get("squelch_dbfs")
        self.squelch_threshold_var = tk.StringVar(
            value="" if threshold is None else f"{threshold:.1f}"
        )
        self.squelch_threshold_entry: Optional[ttk.Entry] = None
        self.full_snapshot_var = tk.BooleanVar(value=False)
        self.nfft_var = tk.StringVar(value="262144")
        self.smooth_var = tk.IntVar(value=3)
        self.range_var = tk.IntVar(value=100)
        self.theme_var = tk.StringVar(value="contrast")
        self.waterfall_cmap_var = tk.StringVar(value="magma")
        self.waterfall_slices_var = tk.StringVar(value="400")
        self.waterfall_floor_var = tk.StringVar(value="110")
        self.target_var = tk.StringVar(value="—")
        self.bandwidth_var = tk.StringVar(value="—")
        self.offset_var = tk.StringVar(value="Offset: —")
        self.sample_rate_var = tk.StringVar(value="Sample rate: —")
        self.status_var = tk.StringVar(value="Select a recording to begin.")

        self.preview_btn: Optional[ttk.Button] = None
        self.confirm_btn: Optional[ttk.Button] = None
        self.status_label: Optional[ttk.Label] = None
        self.plot_frame: Optional[ttk.LabelFrame] = None
        self.placeholder_label: Optional[ttk.Label] = None
        self.trim_check: Optional[ttk.Checkbutton] = None
        self.agc_check: Optional[ttk.Checkbutton] = None
        self.squelch_check: Optional[ttk.Checkbutton] = None

        self.waterfall_window: Optional[_WaterfallWindow] = None
        self._preview_thread: Optional[threading.Thread] = None
        self._preview_lock = threading.Lock()
        self._active_pipeline = None
        self._preview_running = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        if self.initial_path and self.initial_path.exists():
            # Attempt to auto-populate center frequency from metadata/filename and preload snapshot.
            detection = detect_center_frequency(self.initial_path)
            if detection.value is not None:
                self.center_var.set(f"{detection.value:.0f}")
                self._set_center_source(detection.source)
                self.base_kwargs["center_freq"] = detection.value
            else:
                self._set_center_source("unavailable")
            self.root.after(150, lambda: self._load_preview(auto=True))

    def run(self) -> InteractiveSessionResult:
        try:
            self.root.mainloop()
        finally:
            self._cancel_active_pipeline()
            if self._preview_thread and self._preview_thread.is_alive():
                self._preview_thread.join(timeout=5.0)
            self._preview_thread = None
            try:
                self.root.destroy()
            except Exception:  # pragma: no cover - safe cleanup
                pass
        if not self.selection or not self.selected_path:
            raise KeyboardInterrupt()
        kwargs = dict(self.base_kwargs)
        kwargs["center_freq"] = self.center_freq
        kwargs["target_freq"] = self.selection.center_freq
        kwargs["bandwidth"] = self.selection.bandwidth
        kwargs["demod_mode"] = (self.demod_var.get() or kwargs.get("demod_mode", "nfm")).lower()
        self.base_kwargs["demod_mode"] = kwargs["demod_mode"]
        kwargs["silence_trim"] = self.trim_var.get()
        kwargs["squelch_enabled"] = self.squelch_var.get()
        kwargs["agc_enabled"] = self.agc_var.get()
        self.base_kwargs["silence_trim"] = kwargs["silence_trim"]
        self.base_kwargs["squelch_enabled"] = kwargs["squelch_enabled"]
        self.base_kwargs["agc_enabled"] = kwargs["agc_enabled"]
        config = ProcessingConfig(in_path=self.selected_path, **kwargs)
        LOG.info(
            "Interactive selection: center %.0f Hz, target %.0f Hz, bandwidth %.0f Hz",
            self.center_freq or 0.0,
            self.selection.center_freq,
            self.selection.bandwidth,
        )
        return InteractiveSessionResult(config=config, progress_sink=self.progress_sink)

    # --- UI Construction -------------------------------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        file_frame = ttk.LabelFrame(main, text="Recording")
        file_frame.pack(fill="x", expand=False)
        file_frame.columnconfigure(1, weight=1)
        file_frame.columnconfigure(3, weight=0)

        ttk.Label(file_frame, text="Input WAV:").grid(
            row=0, column=0, sticky="w", pady=4
        )
        entry = ttk.Entry(
            file_frame,
            textvariable=self.file_var,
            width=64,
        )
        entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(
            file_frame,
            text="Browse…",
            command=self._on_browse,
        ).grid(row=0, column=2, sticky="w", padx=4, pady=4)

        ttk.Label(file_frame, text="Center freq (Hz):").grid(
            row=1, column=0, sticky="w", pady=4
        )
        center_entry = ttk.Entry(
            file_frame,
            textvariable=self.center_var,
            width=24,
        )
        center_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        center_entry.bind("<FocusOut>", self._on_center_manual)
        center_entry.bind("<Return>", self._on_center_manual)
        ttk.Button(
            file_frame,
            text="Detect from file",
            command=self._parse_center_from_name,
        ).grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Label(
            file_frame,
            textvariable=self.center_source_var,
            foreground="SlateGray",
        ).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(file_frame, text="Snapshot (seconds):").grid(
            row=2, column=0, sticky="w", pady=4
        )
        ttk.Entry(
            file_frame,
            textvariable=self.snapshot_var,
            width=12,
        ).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(
            file_frame,
            text="Load Preview",
            command=self._load_preview,
        ).grid(row=2, column=2, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(
            file_frame,
            text="Analyze entire recording",
            variable=self.full_snapshot_var,
            command=self._schedule_refresh,
        ).grid(row=2, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(file_frame, text="Demodulator:").grid(
            row=3, column=0, sticky="w", pady=4
        )
        demod_combo = ttk.Combobox(
            file_frame,
            textvariable=self.demod_var,
            values=self.demod_options,
            state="readonly",
            width=16,
        )
        demod_combo.grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(
            file_frame,
            text="Choose AM/NFM/USB/LSB demodulation",
            foreground="SlateGray",
        ).grid(row=3, column=2, sticky="w", padx=4, pady=4)
        demod_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_option_state())

        options_frame = ttk.LabelFrame(main, text="Demod options")
        options_frame.pack(fill="x", expand=False, pady=(8, 8))
        options_frame.columnconfigure(0, weight=1)

        self.squelch_check = ttk.Checkbutton(
            options_frame,
            text="Adaptive squelch",
            variable=self.squelch_var,
            command=self._on_toggle_squelch,
        )
        self.squelch_check.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.trim_check = ttk.Checkbutton(
            options_frame,
            text="Trim silences",
            variable=self.trim_var,
            command=self._on_toggle_trim,
        )
        self.trim_check.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self.agc_check = ttk.Checkbutton(
            options_frame,
            text="Automatic gain control",
            variable=self.agc_var,
            command=self._on_toggle_agc,
        )
        self.agc_check.grid(row=0, column=2, sticky="w", padx=4, pady=2)
        ttk.Label(
            options_frame,
            text="Manual threshold (dBFS):",
        ).grid(row=1, column=0, sticky="w", padx=4, pady=(0, 2))
        self.squelch_threshold_entry = ttk.Entry(
            options_frame,
            textvariable=self.squelch_threshold_var,
            width=10,
        )
        self.squelch_threshold_entry.grid(row=1, column=1, sticky="w", padx=4, pady=(0, 2))
        self.squelch_threshold_entry.bind(
            "<FocusOut>", lambda _e: self._on_squelch_threshold_edit()
        )
        self.squelch_threshold_entry.bind(
            "<Return>", lambda _e: self._on_squelch_threshold_edit()
        )
        ttk.Label(
            options_frame,
            text="Adaptive squelch tracks the noise floor and opens when the signal rises ~6 dB above it.\nLeave the threshold blank to auto-track, or enter a negative dBFS value to pin the gate.",
            foreground="SlateGray",
            wraplength=580,
            justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))

        self.plot_frame = ttk.LabelFrame(main, text="Spectrum preview")
        self.plot_frame.pack(fill="both", expand=True, pady=(12, 8))
        self.placeholder_label = ttk.Label(
            self.plot_frame,
            text="Load a recording to view its spectrum.",
            anchor="center",
            justify="center",
        )
        self.placeholder_label.pack(fill="both", expand=True, padx=12, pady=12)

        spectrum_options = ttk.LabelFrame(main, text="Spectrum options")
        spectrum_options.pack(fill="x", expand=False, pady=(0, 8))

        ttk.Label(spectrum_options, text="FFT size").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        nfft_combo = ttk.Combobox(
            spectrum_options,
            textvariable=self.nfft_var,
            values=["65536", "131072", "262144", "524288"],
            state="readonly",
            width=10,
        )
        nfft_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        nfft_combo.bind("<<ComboboxSelected>>", lambda _event: None)

        ttk.Label(spectrum_options, text="Smoothing").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        smooth_spin = ttk.Spinbox(
            spectrum_options,
            from_=1,
            to=20,
            textvariable=self.smooth_var,
            width=5,
            command=lambda: None,
        )
        smooth_spin.grid(row=0, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(spectrum_options, text="Dynamic range (dB)").grid(row=0, column=4, sticky="w", padx=4, pady=2)
        range_spin = ttk.Spinbox(
            spectrum_options,
            from_=20,
            to=140,
            textvariable=self.range_var,
            width=5,
            command=lambda: None,
        )
        range_spin.grid(row=0, column=5, sticky="w", padx=4, pady=2)

        ttk.Label(spectrum_options, text="Theme").grid(row=0, column=6, sticky="w", padx=4, pady=2)
        theme_combo = ttk.Combobox(
            spectrum_options,
            textvariable=self.theme_var,
            values=list(self.color_themes.keys()),
            state="readonly",
            width=10,
        )
        theme_combo.grid(row=0, column=7, sticky="w", padx=4, pady=2)
        theme_combo.bind("<<ComboboxSelected>>", lambda _event: None)
        ttk.Button(
            spectrum_options,
            text="Reset defaults",
            command=self._reset_spectrum_defaults,
        ).grid(row=0, column=8, sticky="w", padx=4, pady=2)

        ttk.Button(
            spectrum_options,
            text="Refresh preview",
            command=self._refresh_preview_manual,
        ).grid(row=0, column=9, sticky="w", padx=4, pady=2)

        waterfall_options = ttk.LabelFrame(main, text="Waterfall options")
        waterfall_options.pack(fill="x", expand=False, pady=(0, 8))

        ttk.Label(waterfall_options, text="Max slices").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        slices_spin = ttk.Spinbox(
            waterfall_options,
            from_=50,
            to=800,
            textvariable=self.waterfall_slices_var,
            width=6,
            command=lambda: None,
        )
        slices_spin.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(waterfall_options, text="Range (dB)").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        waterfall_range_spin = ttk.Spinbox(
            waterfall_options,
            from_=20,
            to=140,
            textvariable=self.waterfall_floor_var,
            width=6,
            command=lambda: None,
        )
        waterfall_range_spin.grid(row=0, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(waterfall_options, text="Colormap").grid(row=0, column=4, sticky="w", padx=4, pady=2)
        waterfall_cmap_combo = ttk.Combobox(
            waterfall_options,
            textvariable=self.waterfall_cmap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis"],
            state="readonly",
            width=10,
        )
        waterfall_cmap_combo.grid(row=0, column=5, sticky="w", padx=4, pady=2)
        waterfall_cmap_combo.bind("<<ComboboxSelected>>", lambda _event: None)

        selection_frame = ttk.LabelFrame(main, text="Channel selection")
        selection_frame.pack(fill="x", expand=False, pady=(0, 8))
        selection_frame.columnconfigure(1, weight=1)

        ttk.Label(selection_frame, textvariable=self.sample_rate_var).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(4, 8)
        )
        ttk.Label(selection_frame, text="Target freq (Hz):").grid(
            row=1, column=0, sticky="w", pady=4
        )
        target_entry = ttk.Entry(
            selection_frame,
            textvariable=self.target_var,
            width=24,
        )
        target_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        target_entry.bind("<FocusOut>", lambda _e: self._on_target_bandwidth_edit())
        target_entry.bind("<Return>", lambda _e: self._on_target_bandwidth_edit())
        ttk.Label(selection_frame, text="Bandwidth (Hz):").grid(
            row=2, column=0, sticky="w", pady=4
        )
        bandwidth_entry = ttk.Entry(
            selection_frame,
            textvariable=self.bandwidth_var,
            width=24,
        )
        bandwidth_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)
        bandwidth_entry.bind("<FocusOut>", lambda _e: self._on_target_bandwidth_edit())
        bandwidth_entry.bind("<Return>", lambda _e: self._on_target_bandwidth_edit())
        ttk.Label(selection_frame, textvariable=self.offset_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

        self.status_label = ttk.Label(
            main,
            textvariable=self.status_var,
            foreground="SlateGray",
        )
        self.status_label.pack(anchor="w", pady=(6, 8))

        button_frame = ttk.Frame(main)
        button_frame.pack(fill="x", expand=False)
        button_frame.columnconfigure(0, weight=1)
        self.preview_btn = ttk.Button(
            button_frame,
            text="Preview DSP",
            command=self._on_preview,
        )
        self.preview_btn.grid(row=0, column=1, sticky="e", padx=6)
        self.preview_btn.configure(state="disabled")
        self.confirm_btn = ttk.Button(
            button_frame,
            text="Confirm & Run",
            command=self._on_confirm,
        )
        self.confirm_btn.grid(row=0, column=2, sticky="e", padx=6)
        self.confirm_btn.configure(state="disabled")
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
        ).grid(row=0, column=3, sticky="e")

        self._update_option_state()

    # --- Event handlers --------------------------------------------------
    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select SDR++ baseband WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)
            self.selected_path = Path(path)
            self._parse_center_from_name()

    def _parse_center_from_name(self) -> None:
        path_text = self.file_var.get().strip()
        if not path_text:
            self._set_status("Browse to a recording before parsing.", error=True)
            return
        path = Path(path_text)
        detection = detect_center_frequency(path)
        if detection.value is None:
            messagebox.showinfo(
                "Center frequency",
                "Could not derive a center frequency from WAV metadata or filename. Enter it manually.",
            )
            self._set_center_source("unavailable")
            self._set_status("Enter center frequency manually.", error=True)
            return
        self.center_var.set(f"{detection.value:.0f}")
        self._set_center_source(detection.source or "filename")
        self.base_kwargs["center_freq"] = detection.value
        friendly = self._describe_center_source(detection.source)
        self._set_status(f"Center frequency populated from {friendly}.", error=False)

    def _on_center_manual(self, _event=None) -> None:
        self._set_center_source("manual")
        value = self._parse_float(self.center_var.get())
        if value is not None and value > 0:
            self.base_kwargs["center_freq"] = value

    def _load_preview(self, auto: bool = False) -> None:
        path_text = self.file_var.get().strip()
        if not path_text:
            if not auto:
                self._set_status("Select an input recording first.", error=True)
            return
        path = Path(path_text)
        previous_path = self.selected_path
        if not path.exists():
            self._set_status(f"File not found: {path}", error=True)
            if not auto:
                messagebox.showerror("Preview failed", f"File not found: {path}")
            return

        snapshot = self._parse_float(self.snapshot_var.get())
        full_capture = self.full_snapshot_var.get()
        if not full_capture:
            if snapshot is None or snapshot <= 0:
                self._set_status("Snapshot duration must be a positive number.", error=True)
                if not auto:
                    messagebox.showerror(
                        "Invalid snapshot",
                        "Snapshot duration must be a positive number of seconds.",
                    )
                self.snapshot_var.set(f"{self.snapshot_seconds:.2f}")
                return
        else:
            snapshot = self.snapshot_seconds  # preserve last value for reuse even though unused

        center_override = self._parse_float(self.center_var.get())

        config = self._build_config(path, center_override)
        try:
            if full_capture:
                self._set_status("Computing spectrum for entire recording…", error=False)
                freqs, psd_db, sample_rate, center_freq, probe, wf_times, wf_matrix = self._compute_full_psd(config)
                block = None
                precomputed = (freqs, psd_db)
                waterfall = (freqs, wf_times, wf_matrix)
            else:
                block, sample_rate, center_freq, probe = gather_snapshot(
                    config, seconds=snapshot
                )
                precomputed = None
                waterfall = self._compute_waterfall_from_samples(block, sample_rate)
        except Exception as exc:
            LOG.error("Failed to gather preview: %s", exc)
            self._set_status(f"Preview failed: {exc}", error=True)
            if not auto:
                messagebox.showerror("Preview failed", str(exc))
            return
        same_file = previous_path is not None and Path(previous_path) == path
        self.selected_path = path
        if not same_file:
            self.selection = None
            if self.confirm_btn:
                self.confirm_btn.configure(state="disabled")
            if self.preview_btn:
                self.preview_btn.configure(state="disabled")
        self.center_freq = center_freq
        self.probe = probe
        self.sample_rate = sample_rate
        self.snapshot_seconds = snapshot
        self.center_var.set(f"{center_freq:.0f}")
        self._render_plot(
            block,
            sample_rate,
            center_freq,
            remember=True,
            precomputed=precomputed,
            waterfall=waterfall,
        )
        self._set_status(
            "Drag over the channel of interest, then Confirm & Run.",
            error=False,
        )

    def _render_plot(
        self,
        samples: Optional[np.ndarray],
        sample_rate: float,
        center_freq: float,
        *,
        remember: bool,
        precomputed: Optional[tuple[np.ndarray, np.ndarray]] = None,
        waterfall: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> None:
        if self.placeholder_label:
            self.placeholder_label.destroy()
            self.placeholder_label = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.plot_container:
            self.plot_container.destroy()
            self.plot_container = None

        nfft = self._parse_int(self.nfft_var.get(), default=131072)
        smooth = max(1, int(self.smooth_var.get()))
        if precomputed is not None:
            freqs, psd_db = precomputed
        else:
            if samples is None:
                raise ValueError("Samples must be provided when no precomputed PSD is supplied.")
            freqs, psd_db = compute_psd(samples, sample_rate, nfft=nfft)
        psd_db = np.asarray(psd_db, dtype=np.float64)
        if smooth > 1 and psd_db.size >= smooth:
            kernel = np.ones(smooth) / smooth
            psd_db = np.convolve(psd_db, kernel, mode="same")
        abs_freqs = center_freq + freqs
        self._freq_min_hz = float(np.min(abs_freqs))
        self._freq_max_hz = float(np.max(abs_freqs))

        self.figure = Figure(figsize=(9.5, 5.2))
        ax = self.figure.add_subplot(111)
        theme = self.color_themes.get(self.theme_var.get(), self.color_themes["default"])
        line_color = theme["line"]
        ax.plot(abs_freqs, psd_db, lw=0.9, color=line_color)
        ax.set_title("Drag to highlight a channel. Scroll or double-click to zoom. Use Preview/Confirm buttons below.")
        if FuncFormatter is not None:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x/1e6:.3f}"))
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

        dynamic = max(20.0, self._get_float(self.range_var, 80.0))
        if psd_db.size:
            finite_vals = psd_db[np.isfinite(psd_db)]
            peak = float(np.max(finite_vals)) if finite_vals.size else 0.0
        else:
            peak = 0.0
        ax.set_ylim(peak - dynamic, peak + 5.0)

        self.plot_container = ttk.Frame(self.plot_frame)
        self.plot_container.pack(fill="both", expand=True)
        self.ax_main = ax
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_container)
        self.canvas.draw()
        if NavigationToolbar2Tk is not None:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
            self.toolbar.update()
            self.toolbar.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
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
        self.sample_rate_var.set(f"Sample rate: {sample_rate:,.2f} Hz")
        if remember:
            if precomputed is not None:
                self.snapshot_data = {
                    "mode": "precomputed",
                    "freqs": freqs.copy(),
                    "psd": psd_db.copy(),
                    "sample_rate": sample_rate,
                    "center_freq": center_freq,
                    "waterfall": waterfall,
                }
            else:
                self.snapshot_data = {
                    "mode": "samples",
                    "samples": samples.copy() if samples is not None else np.empty(0, dtype=np.complex64),
                    "sample_rate": sample_rate,
                    "center_freq": center_freq,
                    "waterfall": waterfall,
                }
        self._update_waterfall_display(sample_rate, center_freq, waterfall)

    def _on_span_change(self, selection: SelectionResult) -> None:
        self.selection = selection
        self.target_var.set(f"{selection.center_freq:.0f}")
        self.bandwidth_var.set(f"{selection.bandwidth:.0f}")
        if self.center_freq is not None:
            offset = selection.center_freq - self.center_freq
            self.offset_var.set(f"Offset: {offset:+.0f} Hz")
        else:
            self.offset_var.set("Offset: —")
        if self.confirm_btn:
            self.confirm_btn.configure(state="normal")
        if self.preview_btn:
            self.preview_btn.configure(state="normal")

    def _on_canvas_click(self, event) -> None:
        if self.ax_main is None or event.inaxes != self.ax_main or event.xdata is None:
            return
        if event.dblclick:
            if self.selection and abs(event.xdata - self.selection.center_freq) <= self.selection.bandwidth / 2.0:
                self._zoom_to_selection()
            else:
                self._zoom_at(event.xdata, factor=0.5)

    def _on_canvas_key(self, event) -> None:
        if event.key == "escape":
            self._on_cancel()

    def _on_confirm(self) -> None:
        if not self.selection or not self.selected_path:
            self._set_status("Select a frequency span before confirming.", error=True)
            return
        self.progress_sink = None
        self.root.quit()

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
        seconds = self._parse_float(self.snapshot_var.get())
        if seconds is None or seconds <= 0:
            seconds = self.snapshot_seconds if self.snapshot_seconds > 0 else 2.0
        try:
            config = self._build_config(self.selected_path, self.center_freq)
            config = replace(
                config,
                target_freq=self.selection.center_freq,
                bandwidth=self.selection.bandwidth,
            )
        except Exception as exc:  # pragma: no cover - setup validation
            LOG.error("Preview setup failed: %s", exc)
            self._set_status(f"Preview setup failed: {exc}", error=True)
            messagebox.showerror("Preview failed", str(exc))
            return

        self._preview_running = True
        try:
            sink: Optional[ProgressSink] = TkProgressSink(self.root)
        except RuntimeError as exc:
            LOG.warning("Progress UI unavailable: %s", exc)
            sink = None
        if self.preview_btn:
            self.preview_btn.configure(state="disabled")
        if self.confirm_btn:
            self.confirm_btn.configure(state="disabled")
        self._set_status(f"Previewing {seconds:.2f} s…", error=False)

        def worker() -> None:
            try:
                result, preview_path = run_preview(
                    config,
                    seconds,
                    progress_sink=sink,
                    on_pipeline=self._register_preview_pipeline,
                )
            except ProcessingCancelled:
                LOG.info("Preview cancelled by user.")
                try:
                    self.root.after(0, self._handle_preview_cancelled)
                except Exception:  # pragma: no cover - root already closed
                    pass
            except Exception as exc:
                LOG.error("Preview failed: %s", exc)
                try:
                    self.root.after(0, lambda err=exc: self._handle_preview_failed(err))
                except Exception:  # pragma: no cover - root already closed
                    pass
            else:
                try:
                    self.root.after(
                        0,
                        lambda res=result, path=preview_path: self._handle_preview_complete(
                            res, path
                        ),
                    )
                except Exception:  # pragma: no cover - root already closed
                    pass
            finally:
                self._register_preview_pipeline(None)

        self._preview_thread = threading.Thread(target=worker, daemon=True)
        self._preview_thread.start()

    def _on_cancel(self) -> None:
        self._cancel_active_pipeline()
        self.selection = None
        self.root.quit()

    def _register_preview_pipeline(self, pipeline) -> None:
        with self._preview_lock:
            self._active_pipeline = pipeline

    def _cancel_active_pipeline(self) -> None:
        with self._preview_lock:
            pipeline = self._active_pipeline
        if pipeline is not None:
            try:
                pipeline.cancel()
            except Exception as exc:
                LOG.debug("Failed to cancel pipeline cleanly: %s", exc)

    def _preview_finished(self) -> None:
        self._preview_running = False
        self._preview_thread = None
        with self._preview_lock:
            self._active_pipeline = None
        if self.preview_btn:
            self.preview_btn.configure(state="normal")
        if self.confirm_btn:
            self.confirm_btn.configure(
                state="normal" if self.selection is not None else "disabled"
            )

    def _handle_preview_complete(self, _result, preview_path: Path) -> None:
        self._preview_finished()
        self._set_status(
            f"Preview complete (output: {preview_path.name})", error=False
        )
        if messagebox:
            messagebox.showinfo(
                "Preview complete",
                f"Preview audio written to:\n{preview_path}",
            )

    def _handle_preview_failed(self, error: Exception) -> None:
        self._preview_finished()
        self._set_status(f"Preview failed: {error}", error=True)
        if messagebox:
            messagebox.showerror("Preview failed", str(error))

    def _handle_preview_cancelled(self) -> None:
        self._preview_finished()
        self._set_status("Preview cancelled.", error=False)

    # --- Helpers ---------------------------------------------------------
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

    def _schedule_refresh(self, *, full: bool = False, waterfall_only: bool = False) -> None:
        if self.snapshot_data is None:
            self._set_status("Load a preview before refreshing.", error=True)
            return
        if self._refresh_job is not None:
            try:
                self.root.after_cancel(self._refresh_job)
            except Exception:  # pragma: no cover - safe guard
                pass
        self._refresh_job = self.root.after(
            0, lambda: self._refresh_plot(full=full, waterfall_only=waterfall_only)
        )

    def _refresh_preview_manual(self) -> None:
        if self.snapshot_data is None:
            self._set_status("Load a preview before refreshing.", error=True)
            return
        mode = self.snapshot_data.get("mode")
        if mode == "samples":
            # For snapshot mode, recompute with current settings.
            self._set_status("Refreshing preview (snapshot)…", error=False)
            self._schedule_refresh(full=True)
        else:
            # For precomputed mode, recompute waterfall but reuse PSD.
            self._set_status("Refreshing waterfall…", error=False)
            self._schedule_refresh(full=False, waterfall_only=True)

    def _refresh_plot(self, *, full: bool = False, waterfall_only: bool = False) -> None:
        self._refresh_job = None
        if not self.snapshot_data:
            return
        mode = self.snapshot_data.get("mode")
        sample_rate = float(self.snapshot_data.get("sample_rate", 0.0))
        center_freq = float(self.snapshot_data.get("center_freq", 0.0))

        if mode == "precomputed":
            freqs = np.asarray(self.snapshot_data.get("freqs"), dtype=np.float64)
            psd = np.asarray(self.snapshot_data.get("psd"), dtype=np.float64)
            waterfall_stored = self.snapshot_data.get("waterfall")
            if waterfall_only and waterfall_stored is not None:
                self._update_waterfall_display(sample_rate, center_freq, waterfall_stored)
            else:
                self._render_plot(
                    None,
                    sample_rate,
                    center_freq,
                    remember=False,
                    precomputed=(freqs, psd),
                    waterfall=waterfall_stored,
                )
            return

        # Snapshot mode uses raw samples.
        samples = np.asarray(self.snapshot_data.get("samples"), dtype=np.complex64)
        if full:
            waterfall = self._compute_waterfall_from_samples(samples, sample_rate)
            self._render_plot(
                samples,
                sample_rate,
                center_freq,
                remember=True,
                precomputed=None,
                waterfall=waterfall,
            )
        elif waterfall_only:
            waterfall = self._compute_waterfall_from_samples(samples, sample_rate)
            self._update_waterfall_display(sample_rate, center_freq, waterfall)
        else:
            waterfall = self._compute_waterfall_from_samples(samples, sample_rate)
            self._render_plot(
                samples,
                sample_rate,
                center_freq,
                remember=False,
                precomputed=None,
                waterfall=waterfall,
            )

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
                        xmax = xmin + width
                    if xmax > self._freq_max_hz:
                        xmax = self._freq_max_hz
                        xmin = xmax - width
        if xmin >= xmax:
            xmin, xmax = self.ax_main.get_xlim()
        self.ax_main.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def _on_toggle_squelch(self) -> None:
        enabled = self.squelch_var.get()
        self.base_kwargs["squelch_enabled"] = enabled
        if not enabled:
            self.trim_var.set(False)
            if self.trim_check:
                self.trim_check.state(["disabled"])
        else:
            if self.trim_check:
                self.trim_check.state(["!disabled"])
        if self.squelch_threshold_entry:
            state = "normal" if enabled else "disabled"
            self.squelch_threshold_entry.configure(state=state)
        self.base_kwargs["silence_trim"] = self.trim_var.get()
        message = (
            "Adaptive squelch enabled. Leave threshold blank to auto-track noise."
            if enabled
            else "Squelch disabled; audio will remain open regardless of level."
        )
        self._set_status(message, error=False)

    def _on_squelch_threshold_edit(self) -> None:
        text = self.squelch_threshold_var.get().strip()
        if not text:
            self.base_kwargs["squelch_dbfs"] = None
            self._set_status(
                "Adaptive squelch will auto-track the noise floor.",
                error=False,
            )
            return
        try:
            value = float(text)
        except ValueError:
            self._set_status(
                "Squelch threshold must be a numeric dBFS value.",
                error=True,
            )
            return
        self.base_kwargs["squelch_dbfs"] = value
        self._set_status(
            f"Manual squelch threshold set to {value:.1f} dBFS.",
            error=False,
        )

    def _reset_spectrum_defaults(self) -> None:
        self.nfft_var.set("131072")
        self.smooth_var.set(1)
        self.range_var.set(80)
        self.theme_var.set("default")
        self.waterfall_slices_var.set("400")
        self.waterfall_floor_var.set("110")
        self.waterfall_cmap_var.set("magma")
        self._set_status("Spectrum defaults restored. Click Refresh preview to update.", error=False)

    def _on_toggle_trim(self) -> None:
        self.base_kwargs["silence_trim"] = self.trim_var.get()
        # No automatic refresh to avoid recompute surprises.
        self._set_status("Updated silence trim setting. Use Preview/Refresh to apply.", error=False)

    def _on_toggle_agc(self) -> None:
        value = self.agc_var.get()
        demod = (self.demod_var.get() or "nfm").lower()
        if demod in {"usb", "lsb", "ssb"}:
            self._preferred_agc = value
        self.base_kwargs["agc_enabled"] = value
        # No automatic refresh; manual preview needed.
        self._set_status("Updated AGC preference. Use Preview/Refresh to apply.", error=False)

    def _on_target_bandwidth_edit(self) -> None:
        try:
            target = float(self.target_var.get())
            if target <= 0:
                raise ValueError()
        except ValueError:
            self._set_status("Target frequency must be positive.", error=True)
            return
        try:
            bandwidth = float(self.bandwidth_var.get())
            if bandwidth <= 0:
                raise ValueError()
        except ValueError:
            self._set_status("Bandwidth must be positive.", error=True)
            return
        if self.span_controller:
            self.span_controller.set_selection(target, bandwidth)
        else:
            self.selection = SelectionResult(target, bandwidth)
            if self.center_freq is not None:
                offset = target - self.center_freq
                self.offset_var.set(f"Offset: {offset:+.0f} Hz")
        self.base_kwargs["target_freq"] = target
        self.base_kwargs["bandwidth"] = bandwidth
        if self.confirm_btn:
            self.confirm_btn.configure(state="normal")
        if self.preview_btn:
            self.preview_btn.configure(state="normal")
        self._set_status("Edited target/bandwidth applied. Use Preview DSP or Refresh preview to update outputs.", error=False)

    def _update_option_state(self) -> None:
        demod = (self.demod_var.get() or "nfm").lower()
        self.base_kwargs["demod_mode"] = demod
        is_ssb = demod in {"usb", "lsb", "ssb"}
        if self.agc_check:
            if is_ssb:
                self.agc_check.state(["!disabled"])
            else:
                self.agc_check.state(["disabled"])
        if is_ssb:
            self.agc_var.set(self._preferred_agc)
            self._on_toggle_agc()
        else:
            self.agc_var.set(False)
            self._on_toggle_agc()
        self._on_toggle_squelch()

    @staticmethod
    def _parse_int(text: str, default: int) -> int:
        try:
            value = int(text)
        except (TypeError, ValueError):
            return default
        return max(1024, value)

    def _get_float(self, var: tk.Variable, default: float) -> float:
        try:
            value = float(var.get())
        except (TypeError, ValueError):
            var.set(f"{default}")
            return default
        return value

    def _compute_waterfall_from_samples(
        self, samples: np.ndarray, sample_rate: float
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if samples.size == 0:
            return None
        nfft = self._parse_int(self.nfft_var.get(), default=131072)
        if samples.size < nfft:
            freqs, psd = compute_psd(samples, sample_rate, nfft=nfft)
            return freqs, np.array([0.0], dtype=np.float32), psd[None, :]
        hop = max(nfft // 4, 1)
        slices: list[np.ndarray] = []
        times: list[float] = []
        start = 0
        idx = 0
        while start + nfft <= samples.size:
            window = samples[start : start + nfft]
            freqs, psd = compute_psd(window, sample_rate, nfft=nfft)
            slices.append(psd)
            times.append(start / sample_rate)
            start += hop
            idx += 1
        if not slices:
            freqs, psd = compute_psd(samples, sample_rate, nfft=nfft)
            return freqs, np.array([0.0], dtype=np.float32), psd[None, :]
        matrix = np.vstack(slices)
        max_slices = self._parse_int(self.waterfall_slices_var.get(), default=400)
        if matrix.shape[0] > max_slices:
            step = math.ceil(matrix.shape[0] / max_slices)
            reduced = []
            reduced_times = []
            for idx in range(0, matrix.shape[0], step):
                reduced.append(np.mean(matrix[idx : idx + step], axis=0))
                reduced_times.append(times[idx])
            matrix = np.array(reduced)
            times = reduced_times
        return freqs, np.array(times, dtype=np.float32), matrix

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
                self.root,
                on_select=self._on_waterfall_pick,
                on_close=self._on_waterfall_closed,
            )
        floor = self._get_float(self.waterfall_floor_var, 90.0)
        cmap = self.waterfall_cmap_var.get() or "viridis"
        max_slices = self._parse_int(self.waterfall_slices_var.get(), default=400)
        freq_arr, times_arr, matrix_arr = waterfall
        times_arr = np.asarray(times_arr, dtype=np.float32)
        matrix_arr = np.asarray(matrix_arr, dtype=np.float64)
        if matrix_arr.shape[0] > max_slices:
            step = math.ceil(matrix_arr.shape[0] / max_slices)
            reduced = []
            reduced_times = []
            for idx in range(0, matrix_arr.shape[0], step):
                reduced.append(np.mean(matrix_arr[idx : idx + step], axis=0))
                reduced_times.append(times_arr[idx])
            matrix_arr = np.array(reduced)
            times_arr = np.array(reduced_times, dtype=np.float32)
        self.waterfall_window.update(
            freqs=freq_arr,
            times=times_arr,
            matrix=matrix_arr,
            center_freq=center_freq,
            sample_rate=sample_rate,
            floor_db=floor,
            cmap=cmap,
        )

    def _set_center_source(self, source: Optional[str]) -> None:
        resolved = source or "unavailable"
        self.center_source = resolved
        if resolved == "unavailable":
            label = "—"
        else:
            label = self._describe_center_source(resolved)
        self.center_source_var.set(f"Center source: {label}")

    def _describe_center_source(self, source: Optional[str]) -> str:
        resolved = source or "unavailable"
        if resolved == "unavailable":
            return "manual entry"
        if resolved == "manual":
            return "manual entry"
        if resolved.startswith("metadata:"):
            detail = resolved.split(":", 1)[1] or "metadata"
            return f"WAV metadata ({detail})"
        if resolved.startswith("filename"):
            return "filename pattern"
        if resolved == "config":
            return "configuration"
        return resolved

    def _set_status(self, message: str, *, error: bool) -> None:
        self.status_var.set(message)
        if self.status_label:
            self.status_label.configure(
                foreground="Firebrick" if error else "SlateGray"
            )

    @staticmethod
    def _format_float(value: Optional[float]) -> str:
        if value is None or value <= 0:
            return ""
        return f"{value:.0f}"

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None

    def _build_config(self, path: Path, center_override: Optional[float]) -> ProcessingConfig:
        kwargs = dict(self.base_kwargs)
        kwargs["center_freq"] = center_override
        demod = (self.demod_var.get() or kwargs.get("demod_mode", "nfm")).lower()
        kwargs["demod_mode"] = demod
        kwargs["silence_trim"] = self.trim_var.get()
        kwargs["squelch_enabled"] = self.squelch_var.get()
        kwargs["agc_enabled"] = self.agc_var.get()
        self.base_kwargs.update(
            {
                "center_freq": center_override,
                "demod_mode": demod,
                "silence_trim": kwargs["silence_trim"],
                "squelch_enabled": kwargs["squelch_enabled"],
                "agc_enabled": kwargs["agc_enabled"],
            }
        )
        target_freq = (
            self.selection.center_freq
            if self.selection is not None
            else self.base_kwargs.get("target_freq", kwargs.get("target_freq", 0.0))
        )
        bandwidth = (
            self.selection.bandwidth
            if self.selection is not None
            else self.base_kwargs.get("bandwidth", kwargs.get("bandwidth", 12_500.0))
        )
        kwargs["target_freq"] = target_freq
        kwargs["bandwidth"] = bandwidth
        self.base_kwargs["target_freq"] = target_freq
        self.base_kwargs["bandwidth"] = bandwidth
        return ProcessingConfig(in_path=path, **kwargs)

    def _compute_full_psd(
        self, config: ProcessingConfig
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        SampleRateProbe,
        np.ndarray,
        np.ndarray,
    ]:
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

        nfft = self._parse_int(self.nfft_var.get(), default=131072)
        chunk_samples = max(config.chunk_size, nfft)
        try:
            from .processing import IQReader
        except ImportError as exc:  # pragma: no cover - safety
            raise RuntimeError(f"Interactive mode missing IQReader: {exc}") from exc

        acc_psd: Optional[np.ndarray] = None
        freqs_ref: Optional[np.ndarray] = None
        slices: list[np.ndarray] = []
        times: list[float] = []
        chunks = 0
        with IQReader(config.in_path, chunk_samples, config.iq_order) as reader:
            for block in reader:
                if block is None or block.size == 0:
                    break
                freqs, psd = compute_psd(block, sample_rate, nfft=nfft)
                if acc_psd is None:
                    acc_psd = psd
                    freqs_ref = freqs
                else:
                    acc_psd += psd
                slices.append(psd)
                times.append(block.size * chunks / sample_rate)
                chunks += 1
                if chunks % 4 == 0:
                    self._set_status(f"Averaging PSD chunk {chunks}…", error=False)
                    self._pump_events()

        if acc_psd is None or freqs_ref is None or chunks == 0:
            raise RuntimeError("Recording did not contain any IQ samples to analyze.")

        avg_psd = acc_psd / float(chunks)
        waterfall_matrix = np.vstack(slices)
        max_slices = self._parse_int(self.waterfall_slices_var.get(), default=200)
        if waterfall_matrix.shape[0] > max_slices:
            step = math.ceil(waterfall_matrix.shape[0] / max_slices)
            reduced = []
            reduced_times = []
            for idx in range(0, waterfall_matrix.shape[0], step):
                reduced.append(np.mean(waterfall_matrix[idx : idx + step], axis=0))
                reduced_times.append(times[idx])
            waterfall_matrix = np.array(reduced)
            times = reduced_times
        return freqs_ref, avg_psd, sample_rate, center_freq, probe, np.array(times, dtype=np.float32), waterfall_matrix

    def _pump_events(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:  # pragma: no cover - window closed
            pass

    def _on_waterfall_pick(self, freq_hz: float) -> None:
        bandwidth = (
            self.selection.bandwidth
            if self.selection is not None
            else self.base_kwargs.get("bandwidth", 12_500.0)
        )
        if self.span_controller:
            self.span_controller.set_selection(freq_hz, bandwidth)
        self.selection = SelectionResult(freq_hz, bandwidth)
        self._on_span_change(self.selection)

    def _on_waterfall_closed(self) -> None:
        self.waterfall_window = None


class _SpanController:
    """Track matplotlib SpanSelector selections and update overlays."""

    def __init__(
        self,
        *,
        ax,
        canvas: FigureCanvasTkAgg,
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


class _WaterfallWindow:
    def __init__(self, root: tk.Tk, on_select, on_close) -> None:
        self.root = root
        self._on_select = on_select
        self._on_close = on_close
        self.window = tk.Toplevel(root)
        self.window.title("Waterfall (time vs frequency)")
        self.window.geometry("900x700")
        self.window.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.figure = Figure(figsize=(8.5, 5.5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.cid = self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.freqs_hz: Optional[np.ndarray] = None
        self.center_freq = 0.0
        self.sample_rate = 0.0
        self.image = None
        self.alive = True
        self._colorbar = None

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

    def _handle_close(self) -> None:
        self.alive = False
        try:
            if self.cid is not None:
                self.figure.canvas.mpl_disconnect(self.cid)
        except Exception:  # pragma: no cover - defensive
            pass
        self.window.destroy()
        if self._on_close:
            self._on_close()


class TkProgressSink(ProgressSink):
    """Thread-aware Tk progress renderer usable from worker threads."""

    def __init__(self, master: tk.Misc):
        if tk is None or ttk is None:
            raise RuntimeError(TK_DEPENDENCY_HINT)
        if master is None:
            raise RuntimeError("Tk root unavailable for progress dialog.")
        self.master = master
        self._queue: queue.Queue[tuple[str, tuple, dict]] = queue.Queue()
        self._after_id: Optional[str] = None
        self._window: Optional[tk.Toplevel] = None
        self._overall_total = 0.0
        self._overall_var: Optional[tk.DoubleVar] = None
        self._overall_completed = 0.0
        self._status_var: Optional[tk.StringVar] = None
        self._bars: Dict[str, Dict[str, object]] = {}
        self._cancel_callback: Optional[Callable[[], None]] = None
        self._stop_button: Optional[ttk.Button] = None
        self._cancelled = False
        self._closed = False

    # -- ProgressSink interface -----------------------------------------
    def start(self, phases, *, overall_total: float) -> None:
        copied = [PhaseState(**phase.__dict__) for phase in phases]
        self._enqueue("start", copied, overall_total)

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
        self._enqueue(
            "advance",
            phase.key,
            phase.completed,
            overall_completed,
            overall_total,
        )

    def status(self, message: str) -> None:
        self._enqueue("status", message)

    def close(self) -> None:
        self._enqueue("close")

    def cancel(self) -> None:
        self._enqueue("cancel")

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        self._enqueue("set_callback", callback)

    # -- Event routing ---------------------------------------------------
    def _enqueue(self, event: str, *args, **kwargs) -> None:
        if self._closed:
            return
        self._queue.put((event, args, kwargs))
        self._schedule()

    def _schedule(self) -> None:
        if self._after_id is not None:
            return
        try:
            self._after_id = self.master.after(0, self._drain_queue)
        except tk.TclError:
            self._after_id = None

    def _drain_queue(self) -> None:
        self._after_id = None
        while not self._queue.empty():
            event, args, kwargs = self._queue.get()
            handler = getattr(self, f"_handle_{event}", None)
            if handler is None:
                continue
            try:
                handler(*args, **kwargs)
            except tk.TclError:
                self._closed = True
                break

    # -- Handlers --------------------------------------------------------
    def _handle_start(self, phases, overall_total: float) -> None:
        if self._closed:
            return
        if self._window is None or not bool(self._window.winfo_exists()):
            self._window = tk.Toplevel(self.master)
            self._window.title("IQ to Audio — Processing")
            self._window.geometry("520x540")
            self._window.transient(self.master)
            self._window.grab_set()
            self._window.protocol("WM_DELETE_WINDOW", self._trigger_cancel)
            wrapper = ttk.Frame(self._window, padding=16)
            wrapper.pack(fill="both", expand=True)
            ttk.Label(
                wrapper,
                text="Processing status",
                font=("TkDefaultFont", 12, "bold"),
            ).pack(anchor="w")

            self._overall_total = max(overall_total, 1.0)
            self._overall_completed = 0.0
            self._overall_var = tk.DoubleVar(value=0.0)
            overall_bar = ttk.Progressbar(
                wrapper,
                mode="determinate",
                maximum=self._overall_total,
                variable=self._overall_var,
            )
            overall_bar.pack(fill="x", pady=(6, 12))

            self._bars = {}
            for phase in phases:
                frame = ttk.Frame(wrapper)
                frame.pack(fill="x", pady=4)
                ttk.Label(frame, text=phase.label).pack(anchor="w")
                maximum = max(phase.total, 1.0)
                var = tk.DoubleVar(value=0.0)
                bar = ttk.Progressbar(
                    frame,
                    mode="determinate",
                    maximum=maximum,
                    variable=var,
                )
                bar.pack(fill="x", padx=(12, 0))
                pct_var = tk.StringVar(value="0.0%")
                ttk.Label(
                    frame,
                    textvariable=pct_var,
                    foreground="SlateGray",
                    font=("TkDefaultFont", 9),
                ).pack(anchor="e")
                self._bars[phase.key] = {
                    "var": var,
                    "total": maximum,
                    "percent": pct_var,
                }

            self._status_var = tk.StringVar(value="Starting…")
            ttk.Label(
                wrapper,
                textvariable=self._status_var,
                foreground="SlateGray",
            ).pack(anchor="w", pady=(12, 0))
            self._stop_button = ttk.Button(
                wrapper,
                text="Stop processing",
                command=self._trigger_cancel,
            )
            state = "normal" if self._cancel_callback is not None else "disabled"
            self._stop_button.configure(state=state)
            self._stop_button.pack(anchor="e", pady=(16, 0))
        else:
            # Reset totals if a previous run reused the window.
            self._overall_total = max(overall_total, 1.0)
            self._overall_completed = 0.0
            if self._overall_var is not None:
                self._overall_var.set(0.0)
            for bar in self._bars.values():
                bar["var"].set(0.0)
                bar["percent"].set("0.0%")
        self._cancelled = False

    def _handle_advance(
        self,
        key: str,
        phase_completed: float,
        overall_completed: float,
        overall_total: float,
    ) -> None:
        if self._closed or self._overall_var is None:
            return
        if key in self._bars:
            bar = self._bars[key]
            total = max(bar["total"], 1.0)
            bar["var"].set(min(phase_completed, total))
            pct = 100.0 * min(phase_completed / total, 1.0)
            bar["percent"].set(f"{pct:5.1f}%")
        self._overall_completed = min(overall_completed, overall_total)
        self._overall_var.set(self._overall_completed)

    def _handle_status(self, message: str) -> None:
        if self._status_var is not None:
            self._status_var.set(message)

    def _handle_close(self) -> None:
        self._closed = True
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
            self._window = None
        self._bars.clear()
        self._overall_var = None
        self._status_var = None
        self._stop_button = None
        self._cancel_callback = None

    def _handle_cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        if self._status_var is not None:
            self._status_var.set("Cancelling…")
        if self._stop_button is not None:
            self._stop_button.configure(state="disabled")

    def _handle_set_callback(self, callback: Optional[Callable[[], None]]) -> None:
        self._cancel_callback = callback
        if self._stop_button is not None:
            state = "normal" if callback is not None else "disabled"
            self._stop_button.configure(state=state)

    # -- Helpers ---------------------------------------------------------
    def _trigger_cancel(self) -> None:
        if self._cancelled:
            return
        self._handle_cancel()
        if self._cancel_callback is not None:
            try:
                self._cancel_callback()
            except Exception as exc:  # pragma: no cover - keep UI responsive
                LOG.warning("Failed to invoke cancel callback: %s", exc)


def interactive_select(
    config: ProcessingConfig, seconds: float = 2.0
) -> InteractiveOutcome:
    """Compatibility wrapper returning the older InteractiveOutcome structure."""
    session = launch_interactive_session(
        input_path=config.in_path,
        base_kwargs={
            "target_freq": config.target_freq,
            "bandwidth": config.bandwidth,
            "center_freq": config.center_freq,
            "demod_mode": config.demod_mode,
            "fs_ch_target": config.fs_ch_target,
            "deemph_us": config.deemph_us,
            "squelch_dbfs": config.squelch_dbfs,
            "silence_trim": config.silence_trim,
            "squelch_enabled": config.squelch_enabled,
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
