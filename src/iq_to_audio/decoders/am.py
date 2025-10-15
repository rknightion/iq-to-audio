from __future__ import annotations

import math

import numpy as np

from .base import Decoder, DecoderStats
from .common import DCBlocker


class AMDecoder(Decoder):
    """Envelope detector with DC blocking."""

    name = "am"

    def __init__(self, dc_radius: float = 0.995):
        self._dc_blocker = DCBlocker(radius=dc_radius)
        self._last_stats: DecoderStats | None = None
        self._intermediates: dict[str, tuple[np.ndarray, float]] = {}
        self._sample_rate = 0.0

    def setup(self, sample_rate: float) -> None:
        self._sample_rate = sample_rate

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, DecoderStats | None]:
        if self._sample_rate == 0.0:
            raise RuntimeError("Decoder.setup(sample_rate) must be called before processing data.")
        envelope = np.abs(samples).astype(np.float32, copy=False)
        ac_coupled = self._dc_blocker.process(envelope)
        rms = math.sqrt(float(np.mean(ac_coupled.astype(np.float64) ** 2)) + 1e-18)
        dbfs = 20.0 * math.log10(rms + 1e-12)
        stats = DecoderStats(rms_dbfs=dbfs)
        self._last_stats = stats
        if samples.size:
            self._intermediates = {
                "envelope": (envelope.copy(), self._sample_rate),
                "dc_block": (ac_coupled.copy(), self._sample_rate),
                "audio": (ac_coupled.copy(), self._sample_rate),
            }
        return ac_coupled, stats

    def finalize(self) -> None:
        return

    def intermediates(self) -> dict[str, tuple[np.ndarray, float]]:
        return dict(self._intermediates)

    @property
    def last_stats(self) -> DecoderStats | None:
        return self._last_stats
