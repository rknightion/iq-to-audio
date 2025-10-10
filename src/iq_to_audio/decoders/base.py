from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DecoderStats:
    """Runtime statistics from a decoder stage."""

    rms_dbfs: float
    gate_threshold_dbfs: float
    dropped: bool


class Decoder(ABC):
    """Abstract interface for demodulation/decoder implementations."""

    name: str = "decoder"

    def setup(self, sample_rate: float) -> None:
        """Prepare decoder state for the given input sample rate."""
        pass

    def finalize(self) -> None:
        """Allow decoder to flush any pending state."""
        pass

    @abstractmethod
    def process(self, samples: np.ndarray) -> Tuple[np.ndarray, Optional[DecoderStats]]:
        """Consume baseband samples and return audio plus optional stats."""

    def intermediates(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Return diagnostic intermediate buffers keyed by stage name."""
        return {}
