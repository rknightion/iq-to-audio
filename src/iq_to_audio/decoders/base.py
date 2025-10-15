from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DecoderStats:
    """Runtime statistics from a decoder stage."""

    rms_dbfs: float


class Decoder(ABC):
    """Abstract interface for demodulation/decoder implementations."""

    name: str = "decoder"

    @abstractmethod
    def setup(self, sample_rate: float) -> None:
        """Prepare decoder state for the given input sample rate."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Allow decoder to flush any pending state."""
        pass

    @abstractmethod
    def process(self, samples: np.ndarray) -> tuple[np.ndarray, DecoderStats | None]:
        """Consume baseband samples and return audio plus optional stats."""

    def intermediates(self) -> dict[str, tuple[np.ndarray, float]]:
        """Return diagnostic intermediate buffers keyed by stage name."""
        return {}
