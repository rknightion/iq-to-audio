from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..probe import SampleRateProbe
from ..visualize import SelectionResult
from .models import MAX_PREVIEW_SAMPLES, MAX_TARGET_FREQUENCIES, SnapshotData

COLOR_THEMES: dict[str, dict[str, str]] = {
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
    "paper": {
        "bg": "#f6f1e1",
        "face": "#f6f1e1",
        "line": "#b15d1c",
        "fg": "#2f2a1d",
        "grid": ":",
        "grid_color": "#d7c9a0",
    },
    "aurora": {
        "bg": "#061622",
        "face": "#061622",
        "line": "#6cf584",
        "fg": "#c3f1ff",
        "grid": "--",
        "grid_color": "#1b3646",
    },
}


@dataclass(slots=True)
class InteractiveState:
    base_kwargs: dict[str, Any]
    default_snapshot: float
    selected_path: Path | None = None
    sample_rate: float | None = None
    center_freq: float | None = None
    center_source: str = "unavailable"
    snapshot_seconds: float = 2.0
    full_snapshot: bool = False
    snapshot_data: SnapshotData | None = None
    selection: SelectionResult | None = None
    target_freqs: list[float] = field(default_factory=list)
    target_text: list[str] = field(default_factory=list)
    bandwidth_hz: float | None = None
    agc_enabled: bool = True
    preferred_agc: bool = True
    demod_mode: str = "nfm"
    input_format_choice: str = "auto"
    detected_format: str | None = None
    input_format_source: str = ""
    input_format_message: str = "Select a recording to detect input format."
    input_format_error: str = ""
    sample_rate_override: float | None = None
    sample_rate_source: str = ""
    output_dir: Path | None = None
    output_hint: str = "Select a recording to preview output location."
    nfft: int = 262_144
    smoothing: int = 3
    dynamic_range: int = 100
    theme: str = "contrast"
    waterfall_cmap: str = "magma"
    waterfall_slices: int = 400
    waterfall_floor: int = 110
    probe: SampleRateProbe | None = None
    center_detected_from: str = ""
    max_preview_samples: int = MAX_PREVIEW_SAMPLES
    max_target_freqs: int = MAX_TARGET_FREQUENCIES

    def __post_init__(self) -> None:
        self.snapshot_seconds = max(self.default_snapshot, 0.25)
        self._initialise_from_kwargs()

    def _initialise_from_kwargs(self) -> None:
        kwargs = self.base_kwargs
        if "center_freq" in kwargs and kwargs["center_freq"] is not None:
            self.center_freq = float(kwargs["center_freq"])
            self.center_source = kwargs.get("center_freq_source", "provided")
        target = kwargs.get("target_freq")
        targets = kwargs.get("target_freqs") or []
        if target and target not in targets:
            targets = [target] + [value for value in targets if value != target]
        self.target_freqs = list(targets)[: self.max_target_freqs]
        self.target_text = [
            f"{freq:.0f}" if isinstance(freq, (int, float)) and freq > 0 else "" for freq in self.target_freqs
        ]
        if len(self.target_text) < self.max_target_freqs:
            self.target_text.extend([""] * (self.max_target_freqs - len(self.target_text)))

        bandwidth = kwargs.get("bandwidth")
        if bandwidth:
            self.bandwidth_hz = float(bandwidth)

        output = kwargs.get("output_path")
        if output:
            output_path = Path(output)
            if output_path.is_file():
                output_path = output_path.parent
            self.output_dir = output_path
            self.output_hint = str(output_path)

        snapshot_seconds = kwargs.get("snapshot_seconds")
        if snapshot_seconds:
            self.snapshot_seconds = float(snapshot_seconds)

        self.full_snapshot = bool(kwargs.get("full_snapshot", False))
        self.nfft = int(kwargs.get("nfft", self.nfft))
        self.smoothing = int(kwargs.get("smoothing", self.smoothing))
        self.dynamic_range = int(kwargs.get("dynamic_range", self.dynamic_range))
        theme = kwargs.get("theme", self.theme)
        if theme in COLOR_THEMES:
            self.theme = theme
        cmap = kwargs.get("waterfall_cmap", self.waterfall_cmap)
        if cmap:
            self.waterfall_cmap = str(cmap)
        self.waterfall_slices = int(kwargs.get("waterfall_slices", self.waterfall_slices))
        self.waterfall_floor = int(kwargs.get("waterfall_floor", self.waterfall_floor))
        self.agc_enabled = bool(kwargs.get("agc_enabled", True))
        self.preferred_agc = self.agc_enabled
        demod_mode = kwargs.get("demod_mode")
        if isinstance(demod_mode, str) and demod_mode:
            self.demod_mode = demod_mode
        manual_container = kwargs.get("input_container")
        manual_format = kwargs.get("input_format")
        if manual_format:
            container = str(manual_container) if manual_container else "wav"
            self.input_format_choice = f"{container}:{manual_format}"
            self.input_format_source = kwargs.get("input_format_source", "manual")
        else:
            self.input_format_choice = "auto"
            self.input_format_source = kwargs.get("input_format_source", "")
        manual_rate = kwargs.get("input_sample_rate")
        if manual_rate:
            try:
                rate = float(manual_rate)
            except (TypeError, ValueError):
                rate = None
            if rate and rate > 0:
                self.sample_rate_override = rate
                self.sample_rate_source = "manual"
            else:
                self.sample_rate_override = None
                self.sample_rate_source = ""
        else:
            self.sample_rate_override = None
            self.sample_rate_source = ""

    def update_center(self, center: float, source: str) -> None:
        self.center_freq = center
        self.center_source = source
        self.base_kwargs["center_freq"] = center
        self.base_kwargs["center_freq_source"] = source

    def update_snapshot_duration(self, seconds: float) -> None:
        self.snapshot_seconds = max(seconds, 0.25)
        self.base_kwargs["snapshot_seconds"] = self.snapshot_seconds

    def set_target_frequencies(self, freqs: list[float]) -> None:
        unique: list[float] = []
        for freq in freqs:
            if freq <= 0:
                continue
            if any(abs(freq - other) < 0.5 for other in unique):
                continue
            unique.append(freq)
        self.target_freqs = unique[: self.max_target_freqs]
        self.base_kwargs["target_freqs"] = list(self.target_freqs)
        if self.target_freqs:
            self.base_kwargs["target_freq"] = self.target_freqs[0]
        else:
            self.base_kwargs["target_freq"] = 0.0

    def set_bandwidth(self, bandwidth: float) -> None:
        self.bandwidth_hz = max(bandwidth, 10.0)
        self.base_kwargs["bandwidth"] = self.bandwidth_hz

    def set_agc_enabled(self, enabled: bool) -> None:
        self.agc_enabled = enabled
        self.base_kwargs["agc_enabled"] = enabled
        self.preferred_agc = enabled

    def apply_agc_override(self, enabled: bool) -> None:
        self.agc_enabled = enabled
        self.base_kwargs["agc_enabled"] = enabled

    def set_demod_mode(self, mode: str) -> None:
        self.demod_mode = mode
        self.base_kwargs["demod_mode"] = mode

    def set_detected_format(
        self,
        format_key: str | None,
        *,
        source: str,
        message: str | None,
        error: str | None,
    ) -> None:
        self.detected_format = format_key
        self.input_format_source = source if format_key else ""
        if format_key:
            if self.input_format_choice == "auto":
                self.input_format_message = message or "Input format detected."
            self.input_format_error = ""
            if self.input_format_choice == "auto":
                self.base_kwargs.pop("input_format", None)
                self.base_kwargs.pop("input_container", None)
                self.base_kwargs["input_format_source"] = source
        else:
            if self.input_format_choice == "auto":
                self.input_format_message = message or "Unable to detect input format."
                self.input_format_error = error or ""
                self.base_kwargs.pop("input_format", None)
                self.base_kwargs.pop("input_container", None)
                self.base_kwargs.pop("input_format_source", None)
            else:
                self.input_format_error = ""

    def set_manual_format(self, container: str, codec: str) -> None:
        key = f"{container}:{codec}"
        self.input_format_choice = key
        self.base_kwargs["input_format"] = codec
        self.base_kwargs["input_container"] = container
        self.base_kwargs["input_format_source"] = "manual"
        self.input_format_error = ""
        self.input_format_message = f"Manual override: {container.upper()} / {codec}"

    def clear_manual_format(self) -> None:
        self.input_format_choice = "auto"
        self.base_kwargs.pop("input_format", None)
        self.base_kwargs.pop("input_container", None)
        if self.detected_format:
            self.base_kwargs["input_format_source"] = self.input_format_source
            if not self.input_format_message:
                self.input_format_message = "Input format detected."
        else:
            self.base_kwargs.pop("input_format_source", None)
            self.input_format_message = "Auto detection pending."
            self.input_format_error = ""

    def set_sample_rate_override(self, value: float | None) -> None:
        if value is None or value <= 0:
            self.sample_rate_override = None
            self.sample_rate_source = ""
            self.base_kwargs.pop("input_sample_rate", None)
            return
        self.sample_rate_override = float(value)
        self.sample_rate_source = "manual"
        self.base_kwargs["input_sample_rate"] = self.sample_rate_override

    def set_sample_rate_detected(self, value: float, source: str) -> None:
        self.sample_rate = value
        if not self.sample_rate_override:
            self.sample_rate_source = source

    def resolved_output_dir(self) -> Path | None:
        if self.output_dir is not None:
            return self.output_dir
        if self.selected_path is not None:
            return self.selected_path.parent
        return None
