from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import fft as sp_fft

LOG = logging.getLogger(__name__)

_NUMPY_EPS = 1e-18


def compute_psd(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = 1 << 18,
    *,
    fft_workers: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a single-sided PSD (dBFS) for complex samples.

    The implementation mirrors the legacy helper in visualize.py but optionally
    leverages SciPy's multi-threaded FFTs when available.
    """
    if samples.size == 0:
        raise ValueError("Cannot compute PSD for an empty signal.")
    use = samples
    if use.size > nfft:
        use = use[:nfft]
    window = np.hanning(use.size).astype(np.float64)
    win_power = np.sum(window ** 2) / use.size
    windowed = np.asarray(use, dtype=np.complex128) * window
    spectrum = _fft_dispatch(windowed, nfft, fft_workers)
    spectrum = np.asarray(sp_fft.fftshift(spectrum))
    freqs = np.asarray(
        sp_fft.fftshift(sp_fft.fftfreq(nfft, d=1.0 / sample_rate)),
        dtype=np.float64,
    )
    scale = (use.size * sample_rate * win_power) + _NUMPY_EPS
    psd = spectrum * np.conj(spectrum) / scale
    psd_db = 10.0 * np.log10(np.abs(psd) + _NUMPY_EPS)
    return freqs.astype(np.float64), psd_db.astype(np.float64)


@dataclass
class WaterfallResult:
    freqs: np.ndarray
    times: np.ndarray
    matrix: np.ndarray


def streaming_waterfall(
    chunks: Iterable[np.ndarray | None],
    sample_rate: float,
    *,
    nfft: int,
    hop: Optional[int] = None,
    max_slices: int = 400,
    fft_workers: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, WaterfallResult, int]:
    """Generate averaged PSD and waterfall slices from a stream of blocks.

    Returns the averaged PSD for the entire capture, along with a capped
    waterfall result (at most ``max_slices`` time slices) and the number of
    FFT frames that were accumulated.
    """
    hop = max(1, hop or nfft // 4)
    fft_plan = _SlidingFFT(sample_rate=sample_rate, nfft=nfft, fft_workers=fft_workers)
    aggregator = _WaterfallAggregator(max_slices=max_slices)
    psd_sum: Optional[np.ndarray] = None
    frames = 0

    for start_index, window in _sliding_windows(chunks, nfft=nfft, hop=hop):
        psd = fft_plan.psd(window)
        if psd_sum is None:
            psd_sum = psd.astype(np.float64, copy=True)
        else:
            psd_sum += psd
        aggregator.add(psd, start_index / sample_rate)
        frames += 1
        if frames % 200 == 0:
            LOG.debug("Accumulated %d FFT frames for waterfall preview.", frames)

    if frames == 0 or psd_sum is None:
        raise ValueError("Input did not contain enough samples for one FFT frame.")

    avg_psd = psd_sum / frames
    times, matrix = aggregator.finalize()
    waterfall = WaterfallResult(freqs=fft_plan.freqs.copy(), times=times, matrix=matrix)
    return fft_plan.freqs.copy(), avg_psd.astype(np.float64), waterfall, frames


def _sliding_windows(
    chunks: Iterable[np.ndarray | None],
    *,
    nfft: int,
    hop: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield overlapping windows of size ``nfft`` from incoming blocks."""
    pending = np.empty(0, dtype=np.complex64)
    offset = 0
    for chunk in chunks:
        if chunk is None:
            continue
        block = np.asarray(chunk, dtype=np.complex64)
        if block.size == 0:
            continue
        if pending.size:
            block = np.concatenate((pending, block))
            offset -= pending.size
        start = 0
        total = block.size
        if total < nfft:
            pending = block
            offset += total
            continue
        while start + nfft <= total:
            window = block[start : start + nfft]
            yield offset + start, window
            start += hop
        pending = block[start:]
        offset += total - pending.size
        if pending.size > nfft:
            pending = pending[-nfft:]
    # No trailing window when fewer than nfft samples remain.


def _fft_dispatch(
    samples: np.ndarray,
    nfft: int,
    fft_workers: Optional[int],
) -> np.ndarray:
    if fft_workers and fft_workers > 1:
        try:
            return np.asarray(sp_fft.fft(samples, n=nfft, workers=fft_workers))
        except TypeError:
            return np.asarray(sp_fft.fft(samples, n=nfft))
    return np.asarray(sp_fft.fft(samples, n=nfft))


class _SlidingFFT:
    def __init__(
        self,
        *,
        sample_rate: float,
        nfft: int,
        fft_workers: Optional[int],
    ):
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.fft_workers = fft_workers if fft_workers and fft_workers > 1 else None
        self.window = np.hanning(self.nfft).astype(np.float64)
        self.win_power = np.sum(self.window ** 2) / self.nfft
        self.freqs = sp_fft.fftshift(
            sp_fft.fftfreq(self.nfft, d=1.0 / self.sample_rate)
        ).astype(np.float64)

    def psd(self, samples: np.ndarray) -> np.ndarray:
        if samples.size != self.nfft:
            raise ValueError(
                f"Sliding FFT expected {self.nfft} samples, received {samples.size}."
            )
        windowed = np.asarray(samples, dtype=np.complex128) * self.window
        spectrum = _fft_dispatch(windowed, self.nfft, self.fft_workers)
        spectrum = np.asarray(sp_fft.fftshift(spectrum))
        scale = (self.nfft * self.sample_rate * self.win_power) + _NUMPY_EPS
        psd = spectrum * np.conj(spectrum) / scale
        result = 10.0 * np.log10(np.abs(psd) + _NUMPY_EPS)
        return np.asarray(result, dtype=np.float64)


class _WaterfallAggregator:
    """Bounded-memory accumulator for waterfall intensity slices."""

    def __init__(self, *, max_slices: int):
        self.max_slices = max(1, int(max_slices))
        self._slices: list[np.ndarray] = []
        self._times: list[float] = []

    def add(self, psd: np.ndarray, time_seconds: float) -> None:
        self._slices.append(np.asarray(psd, dtype=np.float32))
        self._times.append(float(time_seconds))
        self._maybe_reduce()

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._slices:
            return np.empty(0, dtype=np.float32), np.empty((0, 0), dtype=np.float32)
        matrix = np.stack(self._slices, axis=0).astype(np.float32, copy=False)
        times = np.asarray(self._times, dtype=np.float32)
        return times, matrix

    def _maybe_reduce(self) -> None:
        while len(self._slices) > self.max_slices:
            new_slices: list[np.ndarray] = []
            new_times: list[float] = []
            it = iter(range(0, len(self._slices), 2))
            for idx in it:
                first = self._slices[idx]
                if idx + 1 < len(self._slices):
                    second = self._slices[idx + 1]
                    avg = (first.astype(np.float64) + second.astype(np.float64)) / 2.0
                    new_slices.append(avg.astype(np.float32))
                    new_times.append(self._times[idx])
                else:
                    new_slices.append(first)
                    new_times.append(self._times[idx])
            self._slices = new_slices
            self._times = new_times
