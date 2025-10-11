from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

LOG = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class InputFormatSpec:
    """Descriptor for supported IQ input encodings."""

    container: str  # "wav" or "raw"
    codec: str  # ffmpeg codec/subtype identifier (pcm_u8, pcm_s16le, pcm_f32le)
    label: str
    bytes_per_frame: int  # bytes per complex sample on disk (I+Q)
    ffmpeg_input_format: str | None  # raw reader hint ("u8", "s16le", "f32le")
    requires_sample_rate: bool

    @property
    def key(self) -> str:
        return f"{self.container}:{self.codec}"


@dataclass(slots=True)
class InputFormatDetection:
    spec: InputFormatSpec | None
    source: str
    message: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.spec is not None and self.error is None


_FORMAT_MAP: dict[tuple[str, str], InputFormatSpec] = {
    ("wav", "pcm_u8"): InputFormatSpec(
        container="wav",
        codec="pcm_u8",
        label="WAV PCM unsigned 8-bit",
        bytes_per_frame=2,
        ffmpeg_input_format=None,
        requires_sample_rate=False,
    ),
    ("wav", "pcm_s16le"): InputFormatSpec(
        container="wav",
        codec="pcm_s16le",
        label="WAV PCM signed 16-bit",
        bytes_per_frame=4,
        ffmpeg_input_format=None,
        requires_sample_rate=False,
    ),
    ("wav", "pcm_f32le"): InputFormatSpec(
        container="wav",
        codec="pcm_f32le",
        label="WAV float32",
        bytes_per_frame=8,
        ffmpeg_input_format=None,
        requires_sample_rate=False,
    ),
    ("raw", "pcm_u8"): InputFormatSpec(
        container="raw",
        codec="pcm_u8",
        label="RAW complex u8 (.cu8)",
        bytes_per_frame=2,
        ffmpeg_input_format="u8",
        requires_sample_rate=True,
    ),
    ("raw", "pcm_s16le"): InputFormatSpec(
        container="raw",
        codec="pcm_s16le",
        label="RAW complex s16 (.cs16)",
        bytes_per_frame=4,
        ffmpeg_input_format="s16le",
        requires_sample_rate=True,
    ),
    ("raw", "pcm_f32le"): InputFormatSpec(
        container="raw",
        codec="pcm_f32le",
        label="RAW complex f32 (.cf32)",
        bytes_per_frame=8,
        ffmpeg_input_format="f32le",
        requires_sample_rate=True,
    ),
}

_RAW_SUFFIX_MAP: dict[str, tuple[str, str]] = {
    ".cu8": ("raw", "pcm_u8"),
    ".cs16": ("raw", "pcm_s16le"),
    ".cf32": ("raw", "pcm_f32le"),
    ".iq": ("raw", "pcm_s16le"),
}

_WAV_SUBTYPE_MAP: dict[str, str] = {
    "PCM_U8": "pcm_u8",
    "PCM_S8": "pcm_u8",
    "PCM_16": "pcm_s16le",
    "PCM_S16LE": "pcm_s16le",
    "PCM_F32LE": "pcm_f32le",
    "FLOAT": "pcm_f32le",
}

_FFPROBE_HINT = (
    "ffprobe executable not found. Install FFmpeg (e.g., `sudo apt install ffmpeg`, "
    "`brew install ffmpeg`, or download from https://ffmpeg.org/download.html) and ensure it is on PATH."
)


def list_supported_formats(container: str | None = None) -> Iterable[InputFormatSpec]:
    """Iterate supported formats, optionally filtering by container."""
    for spec in _FORMAT_MAP.values():
        if container is None or spec.container == container:
            yield spec


def get_format(container: str, codec: str) -> InputFormatSpec:
    try:
        return _FORMAT_MAP[(container, codec)]
    except KeyError as exc:
        raise ValueError(f"Unsupported input format: {container}:{codec}") from exc


def parse_user_format(value: str, *, default_container: str | None = None) -> tuple[str, str]:
    """Parse CLI/GUI overrides like 'raw:cu8', 'wav-s16', or 'f32'."""
    normalized = value.strip().lower()
    if not normalized or normalized == "auto":
        raise ValueError("parse_user_format() expects a non-auto value.")

    separators = (":", "-")
    container = None
    codec_token = normalized
    for sep in separators:
        if sep in normalized:
            parts = [part for part in normalized.split(sep) if part]
            if len(parts) == 2:
                container, codec_token = parts
                break
    container = container or default_container

    alias_map = {
        "u8": "pcm_u8",
        "cu8": "pcm_u8",
        "s8": "pcm_u8",
        "s16": "pcm_s16le",
        "cs16": "pcm_s16le",
        "pcm16": "pcm_s16le",
        "pcm_s16": "pcm_s16le",
        "f32": "pcm_f32le",
        "float32": "pcm_f32le",
        "cf32": "pcm_f32le",
    }
    codec = alias_map.get(codec_token, codec_token.replace(".", ""))
    if codec not in {"pcm_u8", "pcm_s16le", "pcm_f32le"}:
        raise ValueError(f"Unsupported input codec override: {value}")

    if container is None:
        # Default to WAV unless filename suggests raw (cu8/cs16/cf32 tokens)
        container = "raw" if codec_token.startswith("c") else "wav"

    if container not in {"wav", "raw"}:
        raise ValueError(f"Unknown input container override: {container}")
    return container, codec


def detect_input_format(path: Path) -> InputFormatDetection:
    """Detect supported IQ encodings from WAV headers or RAW filename hints."""
    suffix = path.suffix.lower()
    if suffix in _RAW_SUFFIX_MAP:
        container, codec = _RAW_SUFFIX_MAP[suffix]
        spec = get_format(container, codec)
        return InputFormatDetection(
            spec=spec,
            source=f"extension:{suffix}",
            message=f"Detected {spec.label} via extension.",
        )

    if suffix == ".raw":
        return InputFormatDetection(
            spec=None,
            source="extension:.raw",
            error="Raw '.raw' files need a manual format selection (cu8/cs16/cf32).",
        )

    if suffix not in {".wav", ".wave", ".wv", ".rf64"}:
        return InputFormatDetection(
            spec=None,
            source=f"extension:{suffix or 'none'}",
            error="Unsupported input type. Provide a WAV/RAW IQ recording.",
        )

    try:
        info = sf.info(str(path))
    except RuntimeError as exc:
        LOG.debug("soundfile header read failed for %s: %s", path, exc)
        codec = _ffprobe_codec(path)
        if codec:
            return _codec_to_detection(codec, source="ffprobe")
        return InputFormatDetection(
            spec=None,
            source="soundfile",
            error="Unable to read WAV header; specify format manually.",
        )

    subtype = (info.subtype or "").upper()
    codec = _WAV_SUBTYPE_MAP.get(subtype)
    if codec:
        spec = get_format("wav", codec)
        return InputFormatDetection(
            spec=spec,
            source=f"wav:{subtype.lower()}",
            message=f"WAV subtype {subtype} detected.",
        )
    if subtype in {"PCM_24", "PCM_S24LE", "PCM_32", "PCM_S32LE"}:
        return InputFormatDetection(
            spec=None,
            source=f"wav:{subtype.lower()}",
            error="32-bit/24-bit PCM WAV inputs are not supported. Export as 16-bit or float32.",
        )
    if subtype:
        return InputFormatDetection(
            spec=None,
            source=f"wav:{subtype.lower()}",
            error=f"Unsupported WAV subtype {subtype}. Export as PCM 16-bit or float32.",
        )
    codec = _ffprobe_codec(path)
    if codec:
        return _codec_to_detection(codec, source="ffprobe")
    return InputFormatDetection(
        spec=None,
        source="wav",
        error="Unable to determine WAV subtype; specify format manually.",
    )


def _codec_to_detection(codec: str, *, source: str) -> InputFormatDetection:
    codec_lower = codec.lower()
    if codec_lower in {"pcm_u8", "pcm_s8"}:
        spec = get_format("wav", "pcm_u8")
    elif codec_lower in {"pcm_s16le", "pcm_s16be", "pcm_16"}:
        spec = get_format("wav", "pcm_s16le")
    elif codec_lower in {"pcm_f32le", "pcm_float"}:
        spec = get_format("wav", "pcm_f32le")
    elif codec_lower in {"pcm_s32le", "pcm_32"}:
        return InputFormatDetection(
            spec=None,
            source=f"{source}:{codec_lower}",
            error="32-bit PCM input detected; please convert to 16-bit or float32.",
        )
    else:
        return InputFormatDetection(
            spec=None,
            source=f"{source}:{codec_lower}",
            error=f"Unsupported WAV codec '{codec}'. Export as PCM 16-bit or float32.",
        )
    return InputFormatDetection(
        spec=spec,
        source=f"{source}:{codec_lower}",
        message=f"Detected {spec.label} via {source}.",
    )


def _ffprobe_codec(path: Path) -> str | None:
    if not shutil.which("ffprobe"):
        LOG.debug("ffprobe unavailable for codec detection: %s", path)
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        LOG.warning(_FFPROBE_HINT)
        return None
    except subprocess.CalledProcessError as exc:
        LOG.debug("ffprobe codec detection failed for %s: %s", path, exc)
        return None
    try:
        parsed = json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    streams = parsed.get("streams") if isinstance(parsed, dict) else None
    if not streams:
        return None
    codec = streams[0].get("codec_name")
    return str(codec) if codec else None


def deduce_container(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".wav", ".wave", ".wv", ".rf64"}:
        return "wav"
    if suffix in _RAW_SUFFIX_MAP:
        return "raw"
    return "wav"


def resolve_input_format(
    path: Path,
    *,
    requested: str | None,
    container_hint: str | None = None,
) -> tuple[InputFormatSpec, str]:
    """Resolve the effective input format, using overrides or detection."""
    container = container_hint or deduce_container(path)
    if requested and requested.strip().lower() != "auto":
        manual_container, codec = parse_user_format(requested, default_container=container)
        spec = get_format(manual_container, codec)
        return spec, "manual"

    detection = detect_input_format(path)
    if detection.spec is not None:
        return detection.spec, detection.source
    raise ValueError(detection.error or "Unable to determine input format.")
