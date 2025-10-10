from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .base import Decoder, DecoderStats
from .common import DCBlocker, SquelchGate


class AMDecoder(Decoder):
    """Envelope detector with DC blocking and optional squelch."""

    name = "am"

    def __init__(
        self,
        squelch_dbfs: Optional[float],
        silence_trim: bool,
        squelch_enabled: bool,
        dc_radius: float = 0.995,
    ):
        self._squelch_dbfs = squelch_dbfs
        self._silence_trim = silence_trim
        self._squelch_enabled = squelch_enabled
        self._dc_blocker = DCBlocker(radius=dc_radius)
        self._squelch: Optional[SquelchGate] = None
        self._last_stats: Optional[DecoderStats] = None
        self._intermediates: Dict[str, tuple[np.ndarray, float]] = {}
        self._sample_rate = 0.0

    def setup(self, sample_rate: float) -> None:
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
        envelope = np.abs(samples).astype(np.float32, copy=False)
        ac_coupled = self._dc_blocker.process(envelope)
        if self._squelch is not None:
            gated, threshold, dbfs, dropped = self._squelch.process(ac_coupled)
        else:
            gated = ac_coupled
            rms = math.sqrt(float(np.mean(ac_coupled.astype(np.float64) ** 2)) + 1e-18)
            dbfs = 20.0 * math.log10(rms + 1e-12)
            threshold = self._squelch_dbfs if self._squelch_dbfs is not None else dbfs
            dropped = False
        stats = DecoderStats(rms_dbfs=dbfs, gate_threshold_dbfs=threshold, dropped=dropped)
        self._last_stats = stats
        if samples.size:
            self._intermediates = {
                "envelope": (envelope.copy(), self._sample_rate),
                "dc_block": (ac_coupled.copy(), self._sample_rate),
                "audio": (gated.copy(), self._sample_rate),
            }
        return gated, stats

    def finalize(self) -> None:
        return

    def intermediates(self) -> Dict[str, tuple[np.ndarray, float]]:
        return dict(self._intermediates)

    @property
    def last_stats(self) -> Optional[DecoderStats]:
        return self._last_stats
