from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.widgets import SpanSelector
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..visualize import SelectionResult


@dataclass(slots=True)
class SpanSelection:
    center_freq: float
    bandwidth: float

    @classmethod
    def from_selection(cls, selection: SelectionResult) -> SpanSelection:
        return cls(selection.center_freq, selection.bandwidth)

    def as_selection(self) -> SelectionResult:
        return SelectionResult(self.center_freq, self.bandwidth)


class WaterfallWindow(QtWidgets.QMainWindow):
    """Secondary window hosting the waterfall plot."""

    def __init__(
        self,
        parent: QtWidgets.QMainWindow,
        on_select: Callable[[float], None],
        on_close: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._on_select = on_select
        self._on_close = on_close

        self.setWindowTitle("Waterfall (time vs frequency)")
        self.resize(900, 700)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.figure = Figure(figsize=(8.5, 5.5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.draw()
        layout.addWidget(self.canvas)

        self.setCentralWidget(central)

        self.cid: int | None = None
        self.cid = self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.freqs_hz: np.ndarray | None = None
        self.center_freq = 0.0
        self.sample_rate = 0.0
        self.image: AxesImage | None = None
        self.alive = True
        self._colorbar: Colorbar | None = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._handle_close()
        event.accept()

    def _handle_close(self) -> None:
        self.alive = False
        with contextlib.suppress(Exception):
            if self.cid is not None:
                self.figure.canvas.mpl_disconnect(self.cid)
        self._on_close()

    def update_plot(
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
            with contextlib.suppress(Exception):
                self._colorbar.remove()
            self._colorbar = None
        self.ax.clear()
        peak = float(np.max(matrix[np.isfinite(matrix)])) if np.isfinite(matrix).any() else 0.0
        vmin = peak - max(20.0, floor_db)
        vmax = peak
        freq_mhz = self.freqs_hz / 1e6
        extent = (
            float(freq_mhz.min()),
            float(freq_mhz.max()),
            float(times.max()),
            float(times.min()),
        )
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
        if self.image is not None:
            self._colorbar = self.figure.colorbar(
                self.image, ax=self.ax, orientation="vertical", label="Power (dB)"
            )
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        freq_mhz = float(event.xdata)
        freq_hz = freq_mhz * 1e6
        self._on_select(freq_hz)


class SpanController:
    """Track matplotlib SpanSelector selections and update overlays."""

    def __init__(
        self,
        *,
        ax,
        canvas: FigureCanvasQTAgg,
        initial: SelectionResult,
        on_change: Callable[[SelectionResult], None],
    ):
        self.ax = ax
        self.canvas = canvas
        self.on_change = on_change
        self.selection = SpanSelection.from_selection(initial)

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
        self.selection = SpanSelection(center, bw)
        self._update_lines()
        self._emit()

    def _update_lines(self) -> None:
        half_bw = max(self.selection.bandwidth / 2.0, 1.0)
        self.center_line.set_xdata([self.selection.center_freq, self.selection.center_freq])
        self.low_line.set_xdata([self.selection.center_freq - half_bw] * 2)
        self.high_line.set_xdata([self.selection.center_freq + half_bw] * 2)
        self.canvas.draw_idle()

    def _emit(self) -> None:
        self.on_change(self.selection.as_selection())

    def set_selection(self, center: float, bandwidth: float) -> None:
        self.selection = SpanSelection(center, max(bandwidth, 10.0))
        self._update_lines()
        self._emit()


class PanelGroup(QtWidgets.QGroupBox):
    """Consistent framing for left-side control panels."""

    def __init__(self, title: str, *, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        )
        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                border: 1px solid palette(mid);
                border-radius: 8px;
                margin-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 6px;
                padding: 6px 10px;
            }
            """
        )

    def set_layout(self, layout: QtWidgets.QLayout) -> None:
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.setLayout(layout)


class LockedSplitter(QtWidgets.QSplitter):
    """QSplitter that prevents dragging particular handles."""

    def __init__(
        self,
        orientation: Qt.Orientation,
        *,
        locked_handles: set[int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(orientation, parent)
        self._locked_handles = locked_handles or set()

    def moveSplitter(self, pos: int, handle: int) -> None:  # noqa: N802
        if handle in self._locked_handles:
            return
        super().moveSplitter(pos, handle)
