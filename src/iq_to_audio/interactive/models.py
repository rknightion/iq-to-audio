from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..probe import SampleRateProbe
from ..processing import ProcessingConfig
from ..progress import PhaseState, ProgressSink

MAX_PREVIEW_SAMPLES = 8_000_000  # Complex samples retained in memory for previews (~64 MB).
MAX_TARGET_FREQUENCIES = 5


@dataclass(slots=True)
class InteractiveOutcome:
    center_freq: float
    target_freq: float
    bandwidth: float
    probe: SampleRateProbe


@dataclass(slots=True)
class InteractiveSessionResult:
    configs: list[ProcessingConfig]
    progress_sink: ProgressSink | None

    @property
    def config(self) -> ProcessingConfig:
        return self.configs[0]


@dataclass(slots=True)
class SnapshotData:
    path: Path
    sample_rate: float
    center_freq: float
    probe: SampleRateProbe
    seconds: float
    mode: str
    freqs: np.ndarray
    psd_db: np.ndarray
    waterfall: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    samples: np.ndarray | None
    params: dict[str, Any]
    fft_frames: int


class StatusProgressSink(ProgressSink):
    """Simple progress sink that reflects pipeline status in the status bar."""

    def __init__(
        self,
        update: Callable[[str, bool], None],
        *,
        progress_update: Callable[[float], None] | None = None,
    ):
        self._update = update
        self._progress_update = progress_update
        self._status: str | None = None
        self._overall_total = 0.0
        self._overall_completed = 0.0
        self._cancel_callback: Callable[[], None] | None = None

    def start(self, phases: Iterable[PhaseState], *, overall_total: float) -> None:
        self._overall_total = max(overall_total, 0.0)
        self._overall_completed = 0.0
        self._status = "Processing…"
        if self._progress_update:
            self._progress_update(0.0)
        self._emit(highlight=True)

    def advance(
        self,
        phase: PhaseState,
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
        if self._progress_update:
            self._progress_update(1.0)

    def cancel(self) -> None:
        self._update("Cancelling…", True)
        if self._progress_update:
            self._progress_update(0.0)

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        self._cancel_callback = callback

    def trigger_cancel(self) -> None:
        if self._cancel_callback is not None:
            self._cancel_callback()

    def _emit(self, *, highlight: bool) -> None:
        message = self._status or "Processing…"
        ratio = 0.0
        if self._overall_total > 0 and self._overall_completed > 0:
            ratio = min(self._overall_completed / self._overall_total, 1.0)
            pct = 100.0 * ratio
            message = f"{message} — {pct:4.1f}%"
        if self._progress_update:
            self._progress_update(ratio)
        self._update(message, highlight)
