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

    def update_center(self, center: float, source: str) -> None:
        self.center_freq = center
        self.center_source = source
        self.base_kwargs["center_freq"] = center
        self.base_kwargs["center_freq_source"] = source

    def update_snapshot_duration(self, seconds: float) -> None:
        self.snapshot_seconds = max(seconds, 0.25)
        self.base_kwargs["snapshot_seconds"] = self.snapshot_seconds

    def set_target_frequencies(self, freqs: list[float]) -> None:
        unique = []
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

    def resolved_output_dir(self) -> Path | None:
        if self.output_dir is not None:
            return self.output_dir
        if self.selected_path is not None:
            return self.selected_path.parent
        return None
