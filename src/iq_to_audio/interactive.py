from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .processing import IQReader, ProcessingConfig
from .probe import SampleRateProbe, probe_sample_rate
from .utils import parse_center_frequency
from .visualize import SelectionResult, interactive_select_frequency

LOG = logging.getLogger(__name__)


@dataclass
class InteractiveOutcome:
    center_freq: float
    target_freq: float
    bandwidth: float
    probe: SampleRateProbe


def gather_snapshot(config: ProcessingConfig, seconds: float = 2.0) -> tuple[np.ndarray, float, float, SampleRateProbe]:
    """Read a short segment of IQ data for spectrum display."""
    probe = probe_sample_rate(config.in_path)
    sample_rate = probe.value

    center_freq = config.center_freq
    if center_freq is None:
        parsed = parse_center_frequency(config.in_path)
        if parsed is None:
            raise ValueError(
                "Center frequency not provided and filename parsing failed. "
                "Supply --fc when using interactive mode."
            )
        center_freq = parsed

    block_samples = int(sample_rate * seconds)
    LOG.info("Gathering interactive snapshot: %.2f s (~%d complex samples).", seconds, block_samples)

    with IQReader(config.in_path, block_samples, config.iq_order) as reader:
        block = reader.read_block()

    if block is None or block.size == 0:
        raise RuntimeError("Failed to read samples for interactive snapshot.")

    return block, sample_rate, center_freq, probe


def interactive_select(config: ProcessingConfig, seconds: float = 2.0) -> InteractiveOutcome:
    """Launch interactive UI to choose frequency/bandwidth, returning updated params."""
    block, sample_rate, center_freq, probe = gather_snapshot(config, seconds=seconds)
    initial_freq = config.target_freq if config.target_freq else center_freq
    initial_bw = config.bandwidth
    selection: SelectionResult = interactive_select_frequency(
        block,
        sample_rate,
        center_freq=center_freq,
        initial_freq=initial_freq,
        initial_bw=initial_bw,
    )
    return InteractiveOutcome(
        center_freq=center_freq,
        target_freq=selection.center_freq,
        bandwidth=selection.bandwidth,
        probe=probe,
    )
