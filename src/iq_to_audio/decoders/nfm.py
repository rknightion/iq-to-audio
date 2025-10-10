from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy import signal

from .base import Decoder, DecoderStats
from .common import SquelchGate


class QuadratureDemod:
    """Classic arctangent quadrature FM demodulator."""

    def __init__(self):
        self.prev = np.complex64(1 + 0j)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        prevs = np.concatenate(([self.prev], samples[:-1]))
        prod = samples * np.conj(prevs)
        demod = np.angle(prod).astype(np.float32)
        self.prev = samples[-1]
        return demod


class DeemphasisFilter:
    """Single-pole de-emphasis filter expressed in discrete time."""

    def __init__(self, tau_us: float, sample_rate: Optional[float] = None):
        self.tau_us = tau_us
        self.alpha = 0.0
        self.beta = 0.0
        self.state = 0.0
        self._b = np.array([1.0], dtype=np.float64)
        self._a = np.array([1.0], dtype=np.float64)
        if sample_rate is not None:
            self.configure(sample_rate)

    def configure(self, sample_rate: float) -> None:
        tau_sec = max(self.tau_us * 1e-6, 1e-6)
        alpha = math.exp(-1.0 / (sample_rate * tau_sec))
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.state = 0.0
        self._b = np.array([self.beta], dtype=np.float64)
        self._a = np.array([1.0, -self.alpha], dtype=np.float64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples
        zi = np.array([self.state], dtype=np.float64)
        audio, zf = signal.lfilter(
            self._b,
            self._a,
            samples.astype(np.float32, copy=False),
            zi=zi,
        )
        self.state = float(zf[0])
        return audio.astype(np.float32, copy=False)


class NarrowbandFMDecoder(Decoder):
    """Decoder implementing the NFM demodulation + deemphasis + squelch chain."""

    name = "narrowband_fm"

    def __init__(
        self,
        deemph_us: float,
        squelch_dbfs: Optional[float],
        silence_trim: bool,
        squelch_enabled: bool,
    ):
        self._deemph_us = deemph_us
        self._squelch_dbfs = squelch_dbfs
        self._silence_trim = silence_trim
        self._squelch_enabled = squelch_enabled
        self._demod = QuadratureDemod()
        self._deemph = DeemphasisFilter(deemph_us)
        self._squelch: Optional[SquelchGate] = None
        self._last_stats: Optional[DecoderStats] = None
        self._intermediates: Dict[str, tuple[np.ndarray, float]] = {}
        self._sample_rate = 0.0

    def setup(self, sample_rate: float) -> None:
        self._deemph.configure(sample_rate)
        if self._squelch_enabled:
            self._squelch = SquelchGate(
                sample_rate=sample_rate,
                threshold_dbfs=self._squelch_dbfs,
                silence_trim=self._silence_trim,
            )
        else:
            self._squelch = None
        self._sample_rate = sample_rate

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, Optional[DecoderStats]]:
        if self._squelch is None and self._sample_rate == 0.0:
            raise RuntimeError("Decoder.setup(sample_rate) must be called before processing data.")
        demod = self._demod.process(samples)
        deemph = self._deemph.process(demod)
        if self._squelch is not None:
            gated, threshold, dbfs, dropped = self._squelch.process(deemph)
        else:
            gated = deemph
            rms = math.sqrt(float(np.mean(deemph.astype(np.float64) ** 2)) + 1e-18)
            dbfs = 20.0 * math.log10(rms + 1e-12)
            threshold = self._squelch_dbfs if self._squelch_dbfs is not None else dbfs
            dropped = False
        stats = DecoderStats(rms_dbfs=dbfs, gate_threshold_dbfs=threshold, dropped=dropped)
        self._last_stats = stats
        if samples.size:
            self._intermediates = {
                "demod": (demod.copy(), self._sample_rate),
                "deemph": (deemph.copy(), self._sample_rate),
                "audio": (gated.copy(), self._sample_rate),
            }
        return gated, stats

    def finalize(self) -> None:
        # Nothing to flush for this decoder.
        return

    @property
    def last_stats(self) -> Optional[DecoderStats]:
        return self._last_stats

    def intermediates(self) -> Dict[str, tuple[np.ndarray, float]]:
        return dict(self._intermediates)


__all__ = [
    "DecoderStats",
    "DeemphasisFilter",
    "NarrowbandFMDecoder",
    "QuadratureDemod",
    "SquelchGate",
]
