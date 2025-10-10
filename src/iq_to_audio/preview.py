from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Tuple

from .processing import FFMPEG_HINT, ProcessingConfig, ProcessingPipeline, ProcessingResult
from .utils import detect_center_frequency
from .progress import ProgressSink

LOG = logging.getLogger(__name__)


def _trim_input_file(source: Path, seconds: float) -> Path:
    if seconds <= 0:
        raise ValueError("Preview seconds must be positive.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError(FFMPEG_HINT)

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(temp_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{seconds}",
        "-c",
        "copy",
        str(tmp),
    ]
    LOG.info("Creating %.2f s preview snippet from %s", seconds, source)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to create preview snippet: {exc}") from exc
    return tmp


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
    preview_input = _trim_input_file(config.in_path, seconds)
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
            "Center frequency detected via %s for preview input.",
            detection.source if detection.source else "metadata/filename",
        )
    preview_config = replace(
        config,
        in_path=preview_input,
        output_path=preview_output,
        center_freq=center_freq,
        center_freq_source=center_source,
    )
    pipeline = ProcessingPipeline(preview_config)
    if on_pipeline is not None:
        try:
            on_pipeline(pipeline)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to initialize preview pipeline: {exc}") from exc
    try:
        result = pipeline.run(progress_sink=progress_sink)
    finally:
        try:
            os.remove(preview_input)
        except OSError:
            pass
    LOG.info("Preview DSP complete (%s)", preview_output)
    return result, preview_output
