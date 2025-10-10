from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import soundfile as sf

LOG = logging.getLogger(__name__)

_FREQ_PATTERN = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?)(?P<unit>\s*[kKmMgG]?)\s*(?:[Hh][Zz])?"
)
_FILENAME_FREQ_PATTERN = re.compile(
    r"(?i)(\d+(?:\.\d+)?)([kmg]?)(?:hz)"
)
_METADATA_KEYS = [
    "center_frequency",
    "centerfrequency",
    "frequency",
    "tuner_frequency",
    "tunerfrequency",
    "carrier_frequency",
    "rx_frequency",
    "hz",
]


@dataclass
class CenterFrequencyResult:
    value: Optional[float]
    source: str = "unavailable"


def detect_center_frequency(path: Path) -> CenterFrequencyResult:
    """Detect center frequency (Hz) via metadata first, then filename heuristics."""
    metadata = _center_frequency_from_metadata(path)
    if metadata is not None:
        return metadata
    filename = _center_frequency_from_filename(path)
    if filename is not None:
        return filename
    return CenterFrequencyResult(value=None, source="unavailable")


def parse_center_frequency(path: Path) -> Optional[float]:
    """Backwards-compatible shim returning only the detected value."""
    return detect_center_frequency(path).value


def _center_frequency_from_metadata(path: Path) -> Optional[CenterFrequencyResult]:
    tags: Dict[str, str] = {}
    tags.update(_soundfile_tags(path))
    ffprobe_tags = _ffprobe_tags(path)
    for key, value in ffprobe_tags.items():
        tags.setdefault(key, value)

    for key in _METADATA_KEYS:
        if key in tags:
            freq = _parse_frequency_text(tags[key])
            if freq:
                return CenterFrequencyResult(freq, f"metadata:{key}")

    for key, raw in tags.items():
        freq = _parse_frequency_text(raw)
        if freq:
            return CenterFrequencyResult(freq, f"metadata:{key}")
    return None


def _center_frequency_from_filename(path: Path) -> Optional[CenterFrequencyResult]:
    name = path.name
    matches = []
    for match in _FILENAME_FREQ_PATTERN.finditer(name):
        value = _apply_unit(match.group(1), match.group(2))
        if value and value >= 1_000.0:
            matches.append((value, match.start()))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    value, _ = matches[0]

    stem = path.stem.lower()
    if stem.startswith("baseband_"):
        source = "filename:sdrpp"
    elif re.match(r"\d{2}-\d{2}-\d{2}_", stem):
        source = "filename:sdrsharp"
    else:
        source = "filename"
    return CenterFrequencyResult(value, source)


def _soundfile_tags(path: Path) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return tags
    extra = getattr(info, "extra_info", None)
    if not extra:
        return tags
    for line in extra.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        tags[key.strip().lower()] = value.strip()
    return tags


def _ffprobe_tags(path: Path) -> Dict[str, str]:
    if not shutil.which("ffprobe"):
        return {}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags:stream_tags",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        LOG.debug("ffprobe metadata probe failed for %s: %s", path, exc)
        return {}
    try:
        parsed = json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        LOG.debug("ffprobe metadata parse failed for %s: %s", path, exc)
        return {}
    tags: Dict[str, str] = {}
    format_tags = (
        parsed.get("format", {}).get("tags", {})
        if isinstance(parsed, dict)
        else {}
    )
    for key, value in format_tags.items():
        tags[key.lower()] = str(value)
    for stream in parsed.get("streams", []) or []:
        stream_tags = stream.get("tags", {})
        if not isinstance(stream_tags, dict):
            continue
        for key, value in stream_tags.items():
            tags.setdefault(key.lower(), str(value))
    return tags


def _parse_frequency_text(text: str) -> Optional[float]:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    cleaned = stripped.replace(",", "").replace("_", "")
    try:
        value = float(cleaned)
        if value > 0:
            return value
    except ValueError:
        pass
    match = _FREQ_PATTERN.search(cleaned)
    if not match:
        return None
    magnitude = float(match.group("value"))
    unit = match.group("unit").strip().lower()
    multiplier = {
        "": 1.0,
        "k": 1e3,
        "m": 1e6,
        "g": 1e9,
    }.get(unit, 1.0)
    value = magnitude * multiplier
    return value if value > 0 else None


def _apply_unit(raw_value: str, unit: str) -> Optional[float]:
    try:
        magnitude = float(raw_value)
    except ValueError:
        return None
    unit = unit.lower()
    multiplier = {
        "": 1.0,
        "k": 1e3,
        "m": 1e6,
        "g": 1e9,
    }.get(unit, 1.0)
    value = magnitude * multiplier
    return value if value > 0 else None


__all__ = [
    "CenterFrequencyResult",
    "detect_center_frequency",
    "parse_center_frequency",
]
