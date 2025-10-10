from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Tuple

from .processing import FFMPEG_HINT, ProcessingConfig, ProcessingPipeline, ProcessingResult
from .progress import ProgressSink


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
    preview_config = replace(config, in_path=preview_input, output_path=preview_output)
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
    return result, preview_output
