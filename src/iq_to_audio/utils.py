from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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
    value: float | None
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


def parse_center_frequency(path: Path) -> float | None:
    """Backwards-compatible shim returning only the detected value."""
    return detect_center_frequency(path).value


def _candidate_names(base: str) -> tuple[str, ...]:
    """Return candidate executable names (adds .exe on Windows if needed)."""
    is_windows = sys.platform.startswith("win")

    if not is_windows:
        return (base,)

    # Windows: check if already has .exe extension
    base_lower = base.lower()
    if base_lower.endswith(".exe"):
        return (base,)
    return (base, f"{base}.exe")


def _env_override_path(env_var: str, *, base: str) -> Path | None:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        for name in _candidate_names(base):
            option = candidate / name
            if option.exists():
                return option
        return None
    if candidate.exists():
        return candidate
    # Honour missing extension on Windows env overrides.
    if sys.platform.startswith("win") and candidate.suffix.lower() != ".exe":
        exe_candidate = candidate.with_suffix(candidate.suffix + ".exe" if candidate.suffix else ".exe")
        if exe_candidate.exists():
            return exe_candidate
    return None


def _bundled_executable(*, base: str, subdir: str) -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    search_roots: list[Path] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        search_roots.append(Path(meipass))

    try:
        executable_path = Path(sys.executable).resolve()
    except (AttributeError, RuntimeError):
        executable_path = None
    if executable_path is not None:
        search_roots.append(executable_path.parent)
        if sys.platform == "darwin":
            # PyInstaller places executables inside Contents/MacOS; resources live next door.
            search_roots.append(executable_path.parents[1] / "Resources")

    for root in search_roots:
        candidate_dir = root / subdir
        for name in _candidate_names(base):
            candidate = candidate_dir / name
            if candidate.exists():
                return candidate
    return None


def _path_executable(*, base: str) -> Path | None:
    resolved = shutil.which(base)
    return Path(resolved).resolve() if resolved else None


@functools.lru_cache(maxsize=1)
def resolve_ffmpeg_executable() -> Path | None:
    """Return the preferred ffmpeg binary path for the current environment."""
    for resolver in (
        lambda: _env_override_path("IQ_TO_AUDIO_FFMPEG", base="ffmpeg"),
        lambda: _bundled_executable(base="ffmpeg", subdir="ffmpeg"),
        lambda: _path_executable(base="ffmpeg"),
    ):
        resolved = resolver()
        if resolved:
            return resolved
    return None


@functools.lru_cache(maxsize=1)
def resolve_ffprobe_executable() -> Path | None:
    """Return the preferred ffprobe binary path for the current environment."""
    for resolver in (
        lambda: _env_override_path("IQ_TO_AUDIO_FFPROBE", base="ffprobe"),
        lambda: _bundled_executable(base="ffprobe", subdir="ffmpeg"),
        lambda: _path_executable(base="ffprobe"),
    ):
        resolved = resolver()
        if resolved:
            return resolved
    return None


def _center_frequency_from_metadata(path: Path) -> CenterFrequencyResult | None:
    tags: dict[str, str] = {}
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
        if key in _METADATA_KEYS:
            continue
        lowered = key.lower()
        if "freq" not in lowered and "hz" not in lowered:
            continue
        freq = _parse_frequency_text(raw)
        if freq:
            return CenterFrequencyResult(freq, f"metadata:{key}")
    return None


def _center_frequency_from_filename(path: Path) -> CenterFrequencyResult | None:
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


def _soundfile_tags(path: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
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


def _ffprobe_tags(path: Path) -> dict[str, str]:
    ffprobe_path = resolve_ffprobe_executable()
    if ffprobe_path is None:
        return {}
    cmd = [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format_tags:stream_tags",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        LOG.debug("ffprobe metadata probe failed for %s: %s", path, exc)
        return {}
    try:
        parsed = json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        LOG.debug("ffprobe metadata parse failed for %s: %s", path, exc)
        return {}
    tags: dict[str, str] = {}
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


def _parse_frequency_text(text: str | None) -> float | None:
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


def _apply_unit(raw_value: str, unit: str) -> float | None:
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
