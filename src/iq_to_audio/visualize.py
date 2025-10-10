from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

try:  # Lazy import to avoid forcing matplotlib when unused.
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
except ImportError:  # pragma: no cover - handled at runtime
    plt = None  # type: ignore[assignment]
    Figure = Axes = object  # type: ignore[assignment]


def ensure_matplotlib() -> None:
    if plt is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "matplotlib is required for plotting. Please install it (pip install matplotlib)."
        )


def compute_psd(samples: np.ndarray, sample_rate: float, nfft: int = 1 << 18) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-sided PSD (dBFS) for complex samples."""
    if samples.size == 0:
        raise ValueError("Cannot plot PSD of empty signal.")
    use = samples
    if use.size > nfft:
        use = use[:nfft]
    window = np.hanning(use.size).astype(np.float64)
    win_power = np.sum(window**2) / use.size
    spectrum = np.fft.fftshift(np.fft.fft(use * window, n=nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    psd = spectrum * np.conj(spectrum) / (use.size * sample_rate * win_power + 1e-18)
    psd_db = 10.0 * np.log10(np.abs(psd) + 1e-18)
    return freqs.astype(np.float64), psd_db.astype(np.float64)


def plot_psd(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    *,
    title: str,
    xlabel: str = "Frequency offset (Hz)",
    ylabel: str = "Power (dBFS/Hz)",
    center_freq: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Render PSD on provided axes (creates figure if missing)."""
    ensure_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(freqs, psd_db, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":")
    if center_freq is not None:
        ticks = ax.get_xticks()
        tick_labels = [f"{(center_freq + x):.0f}" for x in ticks]
        ax2 = ax.secondary_xaxis("top")
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel("Absolute frequency (Hz)")
    return ax


def save_stage_psd(
    stage_samples: Dict[str, Tuple[np.ndarray, float]],
    output_path: Path,
    center_freq: float,
) -> None:
    """Persist PSD snapshots for named stages to a single PNG."""
    ensure_matplotlib()
    if not stage_samples:
        raise ValueError("No stage samples available for plotting.")
    stages = list(stage_samples.items())
    cols = min(2, len(stages))
    rows = int(np.ceil(len(stages) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.5 * rows), squeeze=False)
    for idx, (stage, (samples, rate)) in enumerate(stages):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        try:
            freqs, psd_db = compute_psd(samples, rate)
        except ValueError as exc:
            LOG.warning("Skipping PSD for %s: %s", stage, exc)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        plot_psd(
            freqs,
            psd_db,
            title=f"{stage} (fs={rate:.0f} Hz)",
            center_freq=center_freq if stage == "input" else 0.0,
            ax=ax,
        )
    for ax in axes.flatten()[len(stages) :]:
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@dataclass
class SelectionResult:
    center_freq: float
    bandwidth: float


def interactive_select_frequency(
    samples: np.ndarray,
    sample_rate: float,
    center_freq: float,
    *,
    initial_freq: float,
    initial_bw: float,
) -> SelectionResult:
    """Launch matplotlib UI to select center frequency and bandwidth."""
    ensure_matplotlib()

    freqs, psd_db = compute_psd(samples, sample_rate)
    abs_freqs = center_freq + freqs

    class _Controller:
        def __init__(self):
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax.set_title("Select frequency span (drag). Double-click to confirm.")
            self.ax.set_xlabel("Absolute frequency (Hz)")
            self.ax.set_ylabel("Power (dBFS/Hz)")
            self.ax.plot(abs_freqs, psd_db, lw=0.8)
            self.ax.grid(True, ls=":")
            self.span = None
            self.selected = SelectionResult(initial_freq, initial_bw)
            self._init_lines()
            self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            from matplotlib.widgets import SpanSelector

            self.selector = SpanSelector(
                self.ax,
                onselect=self._on_select,
                direction="horizontal",
                useblit=True,
                span_stays=True,
            )
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        def _init_lines(self):
            self._center_line = self.ax.axvline(
                self.selected.center_freq, color="C3", ls="--", lw=1.2, label="Center"
            )
            half_bw = self.selected.bandwidth / 2.0
            self._lo_line = self.ax.axvline(
                self.selected.center_freq - half_bw, color="C2", ls=":", lw=1.0
            )
            self._hi_line = self.ax.axvline(
                self.selected.center_freq + half_bw, color="C2", ls=":", lw=1.0
            )
            self.ax.legend(loc="best")

        def _update_lines(self):
            half_bw = max(self.selected.bandwidth / 2.0, 1.0)
            self._center_line.set_xdata([self.selected.center_freq, self.selected.center_freq])
            self._lo_line.set_xdata([self.selected.center_freq - half_bw] * 2)
            self._hi_line.set_xdata([self.selected.center_freq + half_bw] * 2)
            self.fig.canvas.draw_idle()

        def _on_select(self, xmin: float, xmax: float):
            if xmin == xmax:
                return
            lo, hi = sorted((xmin, xmax))
            center = 0.5 * (lo + hi)
            bw = max(hi - lo, 10.0)
            self.selected = SelectionResult(center, bw)
            LOG.info("Interactive selection: center=%.0f Hz, bw=%.0f Hz", center, bw)
            self._update_lines()

        def _on_click(self, event):
            if event.dblclick:
                LOG.info("Interactive selection confirmed by double-click.")
                plt.close(self.fig)

        def _on_key(self, event):
            if event.key == "enter":
                LOG.info("Interactive selection confirmed (Enter).")
                plt.close(self.fig)

    controller = _Controller()
    controller.fig.show()
    plt.show()
    return controller.selected
