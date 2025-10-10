from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Tuple

from .processing import ProcessingConfig, ProcessingPipeline, ProcessingResult
from .utils import detect_center_frequency
from .progress import ProgressSink

LOG = logging.getLogger(__name__)


def _preview_output_path(config: ProcessingConfig) -> Path:
    if config.output_path:
        base = config.output_path
    else:
        ft = int(config.target_freq) if config.target_freq > 0 else 0
        base = config.in_path.with_name(f"audio_{ft}_48k.wav")
    return base.with_name(f"{base.stem}_preview{base.suffix}")


def run_preview(
    config: ProcessingConfig,
    seconds: float,
    *,
    progress_sink: Optional[ProgressSink] = None,
    on_pipeline: Optional[Callable[[ProcessingPipeline], None]] = None,
) -> Tuple[ProcessingResult, Path]:
    if seconds <= 0:
        raise ValueError("Preview seconds must be positive.")
    preview_output = _preview_output_path(config)
    preview_output.parent.mkdir(parents=True, exist_ok=True)
    center_freq = config.center_freq
    center_source = config.center_freq_source
    if center_freq is None:
        detection = detect_center_frequency(config.in_path)
        if detection.value is None:
            raise ValueError(
                "Center frequency not supplied and could not be determined from metadata or filename. "
                "Use --fc to provide it explicitly."
            )
        center_freq = detection.value
        center_source = detection.source
        LOG.info(
            "Center frequency detected via %s for preview run.",
            detection.source if detection.source else "metadata/filename",
        )
    preview_config = replace(
        config,
        output_path=preview_output,
        center_freq=center_freq,
        center_freq_source=center_source,
        max_input_seconds=seconds,
    )
    pipeline = ProcessingPipeline(preview_config)
    if on_pipeline is not None:
        try:
            on_pipeline(pipeline)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to initialize preview pipeline: {exc}") from exc
    result = pipeline.run(progress_sink=progress_sink)
    LOG.info("Preview DSP complete (%s)", preview_output)
    return result, preview_output
