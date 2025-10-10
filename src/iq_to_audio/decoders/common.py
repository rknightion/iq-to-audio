from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


class DCBlocker:
    """Simple one-pole DC blocker suitable for audio signals."""

    def __init__(self, radius: float = 0.995):
        if not 0.0 < radius < 1.0:
            raise ValueError("radius must be between 0 and 1")
        self.radius = radius
        self._x_prev = 0.0
        self._y_prev = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples
        out = np.empty_like(samples, dtype=np.float32)
        x_prev = self._x_prev
        y_prev = self._y_prev
        r = self.radius
        for idx, sample in enumerate(samples.astype(np.float32, copy=False)):
            y = sample - x_prev + r * y_prev
            out[idx] = y
            x_prev = sample
            y_prev = y
        self._x_prev = float(x_prev)
        self._y_prev = float(y_prev)
        return out


class SquelchGate:
    """Adaptive squelch gate with optional hard threshold and silence trimming."""

    def __init__(
        self,
        sample_rate: float,
        threshold_dbfs: Optional[float],
        silence_trim: bool,
        hold_ms: float = 120.0,
    ):
        self.sample_rate = sample_rate
        self.manual_threshold = threshold_dbfs
        self.silence_trim = silence_trim
        self.hold_samples = int(sample_rate * hold_ms / 1000.0)
        self.hold_counter = 0
        self.noise_dbfs: Optional[float] = None
        self.open = False

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
        if samples.size == 0:
            return samples, self.manual_threshold or -120.0, -120.0, False

        rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)) + 1e-18)
        dbfs = 20.0 * math.log10(rms + 1e-12)

        if self.manual_threshold is None:
            if self.noise_dbfs is None:
                self.noise_dbfs = dbfs
            elif not self.open:
                self.noise_dbfs = 0.95 * self.noise_dbfs + 0.05 * dbfs
            threshold = (self.noise_dbfs or dbfs) + 6.0
        else:
            threshold = self.manual_threshold

        dropped = False
        if dbfs >= threshold:
            self.open = True
            self.hold_counter = self.hold_samples
        else:
            if self.hold_counter > 0:
                self.hold_counter = max(0, self.hold_counter - samples.size)
            else:
                self.open = False

        if self.open:
            audio = samples
        else:
            if self.silence_trim:
                audio = np.empty(0, dtype=samples.dtype)
                dropped = True
            else:
                audio = np.zeros_like(samples)

        return audio, threshold, dbfs, dropped
