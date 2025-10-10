from __future__ import annotations

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
