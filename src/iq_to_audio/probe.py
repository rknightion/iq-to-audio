from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf


@dataclass
class SampleRateProbe:
    ffprobe: Optional[float]
    header: Optional[float]
    wave: Optional[float] = None

    @property
    def value(self) -> float:
        """Return the best available sample rate measurement."""
        if self.ffprobe:
            return self.ffprobe
        if self.header:
            return self.header
        if self.wave:
            return self.wave
        raise RuntimeError("Unable to determine sample rate from ffprobe or header.")


def probe_sample_rate(path: Path) -> SampleRateProbe:
    """Probe the WAV sample rate via ffprobe with ignore_length fallback."""
    ffprobe_rate = _ffprobe_sample_rate(path)
    header_rate = _header_sample_rate(path)
    wave_rate = _wave_sample_rate(path)
    return SampleRateProbe(ffprobe=ffprobe_rate, header=header_rate, wave=wave_rate)


def _ffprobe_sample_rate(path: Path) -> Optional[float]:
    global _FFPROBE_WARNED
    if not shutil.which("ffprobe"):
        if not _FFPROBE_WARNED:
            LOG.warning(FFPROBE_HINT)
            _FFPROBE_WARNED = True
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "json",
        "-ignore_length",
        "1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as exc:
        LOG.warning("ffprobe failed for %s: %s", path, exc)
        return None
    except FileNotFoundError:
        return None

    try:
        parsed = json.loads(result.stdout.decode("utf-8"))
        streams = parsed.get("streams", [])
        if not streams:
            return None
        rate_val = streams[0].get("sample_rate")
        if rate_val is None:
            return None
        return float(rate_val)
    except (ValueError, KeyError, TypeError):
        return None


def _header_sample_rate(path: Path) -> Optional[float]:
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return None
    return float(info.samplerate) if info.samplerate else None


def _wave_sample_rate(path: Path) -> Optional[float]:
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            return float(wf.getframerate())
    except wave.Error:
        return None
    except FileNotFoundError:
        return None

LOG = logging.getLogger(__name__)

FFPROBE_HINT = (
    "ffprobe executable not found. Install FFmpeg (e.g., `sudo apt install ffmpeg` (linux), "
    "`brew install ffmpeg` (macOS), `winget install ffmpeg` (Windows) "
    "or download from https://ffmpeg.org/download.html) and ensure it is on PATH."
)

_FFPROBE_WARNED = False
