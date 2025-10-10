from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

try:  # pragma: no cover - tqdm is optional for programmatic use
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tests may run without tqdm installed
    tqdm = None  # type: ignore[assignment]


@dataclass
class PhaseState:
    """Track metadata and progress for a single processing phase."""

    key: str
    label: str
    total: float
    unit: str = "samples"
    completed: float = 0.0

    def remaining(self) -> float:
        return max(self.total - self.completed, 0.0)


class ProgressSink:
    """Interface for receiving progress events."""

    def start(self, phases: Iterable[PhaseState], *, overall_total: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def advance(
        self,
        phase: PhaseState,
        delta: float,
        *,
        overall_completed: float,
        overall_total: float,
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def status(self, message: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:  # pragma: no cover - optional hook
        return

    def cancel(self) -> None:  # pragma: no cover - optional hook
        raise NotImplementedError


class NullProgressSink(ProgressSink):
    """Sink that ignores all progress events."""

    def start(self, phases: Iterable[PhaseState], *, overall_total: float) -> None:
        return

    def advance(
        self,
        phase: PhaseState,
        delta: float,
        *,
        overall_completed: float,
        overall_total: float,
    ) -> None:
        return

    def status(self, message: str) -> None:
        return

    def close(self) -> None:
        return

    def cancel(self) -> None:
        return


class TqdmProgressSink(ProgressSink):
    """Render per-phase and aggregate progress using tqdm progress bars."""

    def __init__(self):
        if tqdm is None:
            raise RuntimeError(
                "tqdm is required for progress reporting but is not installed."
            )
        self._overall: Optional[tqdm] = None
        self._bars: Dict[str, tqdm] = {}
        self._status: Optional[str] = None
        self._cancel_callback: Optional[Callable[[], None]] = None

    def start(self, phases: Iterable[PhaseState], *, overall_total: float) -> None:
        phases_list = list(phases)
        total = overall_total if overall_total > 0 else None
        self._overall = tqdm(
            total=total,
            desc="Total",
            unit="samples",
            position=0,
            leave=True,
        )
        for idx, phase in enumerate(phases_list, start=1):
            phase_total = phase.total if phase.total > 0 else None
            bar = tqdm(
                total=phase_total,
                desc=phase.label,
                unit=phase.unit,
                position=idx,
                leave=True,
            )
            self._bars[phase.key] = bar

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        self._cancel_callback = callback

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
        if self._overall is not None:
            self._overall.update(delta)
            if overall_total > 0 and self._status:
                pct = 100.0 * min(overall_completed / overall_total, 1.0)
                self._overall.set_postfix_str(f"{self._status} ({pct:5.1f}%)")
        bar = self._bars.get(phase.key)
        if bar is not None:
            bar.update(delta)
            if phase.total > 0:
                pct = 100.0 * min(phase.completed / phase.total, 1.0)
                bar.set_postfix_str(f"{pct:5.1f}%")

    def status(self, message: str) -> None:
        self._status = message
        if self._overall is not None:
            self._overall.set_postfix_str(message)

    def close(self) -> None:
        if self._overall is not None:
            self._overall.close()
            self._overall = None
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()
        self._cancel_callback = None

    def cancel(self) -> None:
        if self._overall is not None:
            self._overall.set_postfix_str("Cancelled")


class ProgressTracker:
    """Coordinate progress phases and delegate rendering to a sink."""

    _MAX_STATUS_WIDTH = 48

    def __init__(self, sink: Optional[ProgressSink] = None):
        self._sink: ProgressSink = sink or NullProgressSink()
        self._phases: Dict[str, PhaseState] = {}
        self._overall_total = 0.0
        self._overall_completed = 0.0
        self._started = False
        self._cancelled = False
        self._cancel_notified = False

    def start(self, phases: Iterable[PhaseState]) -> None:
        if self._started:
            return
        self._phases = {phase.key: PhaseState(**phase.__dict__) for phase in phases}
        self._overall_total = sum(p.total for p in self._phases.values())
        self._overall_completed = 0.0
        self._sink.start(
            self._phases.values(),
            overall_total=self._overall_total,
        )
        self._started = True
        self._cancelled = False
        self._cancel_notified = False

    def advance(self, key: str, amount: float) -> None:
        if (
            not self._started
            or self._cancelled
            or key not in self._phases
            or amount <= 0
        ):
            return
        phase = self._phases[key]
        previous = phase.completed
        phase.completed = min(previous + amount, phase.total)
        delta = phase.completed - previous
        if delta <= 0:
            return
        self._overall_completed = min(
            self._overall_completed + delta, self._overall_total
        )
        self._sink.advance(
            phase,
            delta,
            overall_completed=self._overall_completed,
            overall_total=max(self._overall_total, 1e-9),
        )

    def status(self, message: str) -> None:
        if not self._started:
            return
        self._sink.status(self._normalize_status(message))

    def close(self) -> None:
        self._sink.close()
        self._started = False
        self._cancelled = False
        self._cancel_notified = False

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        if not self._cancel_notified and hasattr(self._sink, "cancel"):
            try:
                self._sink.cancel()
            except NotImplementedError:
                pass
            finally:
                self._cancel_notified = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _normalize_status(self, message: str) -> str:
        stripped = " ".join(str(message).split())
        if len(stripped) <= self._MAX_STATUS_WIDTH:
            return stripped
        return stripped[: self._MAX_STATUS_WIDTH - 1] + "…"


def clamp(amount: float, upper: float) -> float:
    return min(amount, upper)
