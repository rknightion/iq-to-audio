from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .utils import resolve_ffprobe_executable


@dataclass
class SampleRateProbe:
    ffprobe: float | None
    header: float | None
    wave: float | None = None

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


def _ffprobe_sample_rate(path: Path) -> float | None:
    global _FFPROBE_WARNED
    ffprobe_path = resolve_ffprobe_executable()
    if ffprobe_path is None:
        if not _FFPROBE_WARNED:
            LOG.warning(FFPROBE_HINT)
            _FFPROBE_WARNED = True
        return None

    cmd = [
        str(ffprobe_path),
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
        result = subprocess.run(cmd, capture_output=True, check=True)
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


def _header_sample_rate(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return None
    return float(info.samplerate) if info.samplerate else None


def _wave_sample_rate(path: Path) -> float | None:
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
