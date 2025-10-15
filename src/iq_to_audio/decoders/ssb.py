from __future__ import annotations

import math

import numpy as np

from .base import Decoder, DecoderStats
from .common import DCBlocker


class SSBDecoder(Decoder):
    """Single-sideband speech demodulator supporting USB and LSB."""

    name = "ssb"

    def __init__(
        self,
        sideband: str,
        agc_enabled: bool,
        dc_radius: float = 0.995,
        agc_target_dbfs: float = -12.0,
        agc_decay: float = 0.001,
    ):
        sideband = sideband.lower()
        if sideband not in {"usb", "lsb"}:
            raise ValueError("sideband must be 'usb' or 'lsb'")
        self._sideband = sideband
        self._agc_enabled = agc_enabled
        self._dc_blocker = DCBlocker(radius=dc_radius)
        self._last_stats: DecoderStats | None = None
        self._intermediates: dict[str, tuple[np.ndarray, float]] = {}
        self._sample_rate = 0.0
        self._agc_level = 10.0 ** (agc_target_dbfs / 20.0)
        self._agc_decay = agc_decay

    def setup(self, sample_rate: float) -> None:
        self._sample_rate = sample_rate

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, DecoderStats | None]:
        if self._sample_rate == 0.0:
            raise RuntimeError("Decoder.setup(sample_rate) must be called before processing data.")
        analytic = np.conj(samples) if self._sideband == "lsb" else samples
        baseband = analytic.real.astype(np.float32, copy=False)
        dc_audio = self._dc_blocker.process(baseband)
        processed = self._apply_agc(dc_audio) if self._agc_enabled else dc_audio
        rms = math.sqrt(float(np.mean(processed.astype(np.float64) ** 2)) + 1e-18)
        dbfs = 20.0 * math.log10(rms + 1e-12)
        stats = DecoderStats(rms_dbfs=dbfs)
        self._last_stats = stats
        if samples.size:
            intermediates: dict[str, tuple[np.ndarray, float]] = {
                "analytic": (analytic.copy(), self._sample_rate),
                "dc_block": (dc_audio.copy(), self._sample_rate),
            }
            if self._agc_enabled:
                intermediates["agc"] = (processed.copy(), self._sample_rate)
            intermediates["audio"] = (processed.copy(), self._sample_rate)
            self._intermediates = intermediates
        return processed, stats

    def finalize(self) -> None:
        return

    def intermediates(self) -> dict[str, tuple[np.ndarray, float]]:
        return dict(self._intermediates)

    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        target = self._agc_level
        decay = self._agc_decay
        gain = 1.0
        out = np.empty_like(audio, dtype=np.float32)
        for idx, sample in enumerate(audio):
            magnitude = abs(sample)
            if magnitude > 1e-6:
                desired_gain = target / magnitude
                gain += decay * (desired_gain - gain)
            out[idx] = sample * gain
        return out

    @property
    def last_stats(self) -> DecoderStats | None:
        return self._last_stats
