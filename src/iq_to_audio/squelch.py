from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy import signal

SquelchMethod = Literal["adaptive", "static", "transient"]

LOG = logging.getLogger(__name__)

_MIN_DBFS = -160.0
_EPS = 1e-10


def _ensure_2d(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples[:, np.newaxis]
    if samples.ndim != 2:
        raise ValueError(f"Expected mono/stereo audio, received shape {samples.shape!r}.")
    return samples


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    window = max(int(window), 1)
    if window == 1:
        return values_arr
    kernel = np.ones(window, dtype=np.float32) / float(window)
    averaged = np.convolve(values_arr, kernel, mode="same")
    return np.asarray(averaged, dtype=np.float32)


def _envelope(samples: np.ndarray, window: int) -> np.ndarray:
    magnitude = np.mean(np.abs(samples), axis=1, dtype=np.float64)
    magnitude_arr = np.asarray(magnitude, dtype=np.float32)
    return _moving_average(magnitude_arr, window)


def _dbfs(values: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    safe = np.maximum(values_arr, _EPS)
    db = np.maximum(_MIN_DBFS, 20.0 * np.log10(safe))
    return np.asarray(db, dtype=np.float32)


def _estimate_noise_floor(
    envelope_db: np.ndarray,
    percentile: float,
) -> float:
    pct = float(np.clip(percentile, 0.0, 1.0)) * 100.0
    return float(np.percentile(envelope_db, pct))


def _percentile_difference(
    values: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    low = np.percentile(values_arr, low_pct * 100.0)
    high = np.percentile(values_arr, high_pct * 100.0)
    span = max(high - low, 1e-6)
    return np.asarray((values_arr - low) / span, dtype=np.float32)


def _dilate_mask(mask: np.ndarray, head: int, tail: int) -> np.ndarray:
    expanded = mask.copy()
    if tail > 0:
        kernel = np.ones(tail + 1, dtype=np.int8)
        tail_conv = np.convolve(mask.astype(np.int8, copy=False), kernel, mode="full")
        expanded |= tail_conv[: mask.size] > 0
    if head > 0:
        kernel = np.ones(head + 1, dtype=np.int8)
        head_conv = np.convolve(mask[::-1].astype(np.int8, copy=False), kernel, mode="full")
        expanded |= head_conv[: mask.size][::-1] > 0
    return expanded


def _smooth_gain(mask: np.ndarray, fade: int) -> np.ndarray:
    if fade <= 0:
        return mask.astype(np.float32, copy=False)
    ramp = np.linspace(0.0, 1.0, fade + 1, dtype=np.float32)
    fade_kernel = np.concatenate((ramp[:-1], np.ones(1, dtype=np.float32), ramp[1:][::-1]))
    fade_kernel /= float(fade_kernel.max(initial=1.0))
    mask_float = mask.astype(np.float32, copy=False)
    pad = fade_kernel.size // 2
    padded = np.pad(mask_float, pad, mode="edge")
    smoothed_full = np.convolve(padded, fade_kernel, mode="same")
    smoothed = smoothed_full[pad:-pad] if pad > 0 else smoothed_full
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32, copy=False)


def _apply_trim(
    samples: np.ndarray,
    gain: np.ndarray,
    sample_rate: float,
    lead_seconds: float,
    trail_seconds: float,
) -> np.ndarray:
    active_indices = np.flatnonzero(gain > 1e-3)
    if active_indices.size == 0:
        return samples[:0].copy()
    lead_samples = int(max(0, round(sample_rate * lead_seconds)))
    trail_samples = int(max(0, round(sample_rate * trail_seconds)))
    start = max(0, active_indices[0] - lead_samples)
    stop = min(samples.shape[0], active_indices[-1] + trail_samples + 1)
    return samples[start:stop].copy()


@dataclass(slots=True)
class SquelchConfig:
    method: SquelchMethod = "adaptive"
    auto_noise_floor: bool = True
    manual_noise_floor_db: float | None = None
    noise_floor_percentile: float = 0.2
    threshold_margin_db: float = 6.0
    window_seconds: float = 0.04
    transient_window_seconds: float = 0.012
    transient_margin_db: float = 8.0
    hold_seconds: float = 0.12
    fade_seconds: float = 0.01
    trim_silence: bool = True
    trim_lead_seconds: float = 0.15
    trim_trail_seconds: float = 0.35

    def resolve_noise_floor(self, envelope_db: np.ndarray) -> float:
        if self.auto_noise_floor:
            return _estimate_noise_floor(envelope_db, self.noise_floor_percentile)
        if self.manual_noise_floor_db is None:
            raise ValueError("manual_noise_floor_db must be provided when auto_noise_floor=False.")
        return float(self.manual_noise_floor_db)


@dataclass(slots=True)
class AudioPostOptions:
    config: SquelchConfig
    overwrite: bool = False
    cleaned_suffix: str = "-cleaned"
    allowed_suffixes: Sequence[str] = (".wav", ".flac", ".ogg", ".mp3")


@dataclass(slots=True)
class SquelchFileResult:
    input_path: Path
    output_path: Path
    samples_in: int
    samples_out: int
    duration_in: float
    duration_out: float
    bytes_in: int
    bytes_out: int
    noise_floor_db: float
    threshold_db: float
    method: SquelchMethod
    retained_ratio: float


@dataclass(slots=True)
class SquelchSummary:
    results: list[SquelchFileResult]
    errors: list[tuple[Path, Exception]]

    @property
    def processed(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return len(self.errors)

    @property
    def total(self) -> int:
        return self.processed + self.failed

    def aggregate_duration_delta(self) -> float:
        return float(sum(item.duration_out - item.duration_in for item in self.results))

    def aggregate_size_delta(self) -> int:
        return int(sum(item.bytes_out - item.bytes_in for item in self.results))


def _transient_mask(
    samples: np.ndarray,
    sample_rate: float,
    config: SquelchConfig,
) -> np.ndarray:
    short_win = max(1, int(round(config.transient_window_seconds * sample_rate)))
    long_win = max(short_win * 4, int(round(config.window_seconds * sample_rate)))
    short_env = _envelope(samples, short_win)
    long_env = _envelope(samples, long_win)
    diff_db = _dbfs(short_env) - _dbfs(long_env + _EPS)
    return np.asarray(diff_db >= config.transient_margin_db, dtype=bool)


def _adaptive_mask(
    envelope_db: np.ndarray,
    threshold_db: float,
) -> np.ndarray:
    above = envelope_db >= threshold_db
    if not np.any(above):
        return above
    baseline = np.minimum.accumulate(envelope_db)
    relative = envelope_db - baseline
    score = _percentile_difference(relative, 0.05, 0.95)
    adaptive_threshold = np.clip(threshold_db + 6.0 * (1.0 - score), threshold_db - 6.0, threshold_db + 6.0)
    return envelope_db >= adaptive_threshold


def _static_mask(envelope_db: np.ndarray, threshold_db: float) -> np.ndarray:
    return envelope_db >= threshold_db


def apply_squelch(
    audio: np.ndarray,
    sample_rate: float,
    config: SquelchConfig,
) -> tuple[np.ndarray, float, float]:
    samples = _ensure_2d(np.asarray(audio, dtype=np.float32))
    window = max(1, int(round(config.window_seconds * sample_rate)))
    envelope = _envelope(samples, window)
    envelope_db = _dbfs(envelope)
    noise_floor_db = config.resolve_noise_floor(envelope_db)
    threshold_db = noise_floor_db + config.threshold_margin_db

    if config.method == "transient":
        mask = _transient_mask(samples, sample_rate, config)
    elif config.method == "adaptive":
        mask = _adaptive_mask(envelope_db, threshold_db)
    elif config.method == "static":
        mask = _static_mask(envelope_db, threshold_db)
    else:
        raise ValueError(f"Unsupported squelch method: {config.method}")

    head = int(round(sample_rate * config.hold_seconds))
    expanded_mask = _dilate_mask(mask, head=head, tail=head)
    fade = int(round(sample_rate * config.fade_seconds))
    gain = _smooth_gain(expanded_mask, fade)
    cleaned = samples * gain[:, np.newaxis]

    if config.trim_silence:
        trimmed = _apply_trim(
            cleaned,
            gain,
            sample_rate,
            config.trim_lead_seconds,
            config.trim_trail_seconds,
        )
    else:
        trimmed = cleaned.copy()

    if trimmed.size == 0:
        trimmed = np.zeros((0, cleaned.shape[1]), dtype=np.float32)

    return trimmed.astype(np.float32, copy=False), noise_floor_db, threshold_db


def _derive_output_path(path: Path, options: AudioPostOptions) -> Path:
    if options.overwrite:
        return path
    suffix = options.cleaned_suffix
    if not suffix:
        suffix = "-cleaned"
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _load_audio(path: Path) -> tuple[np.ndarray, int, str, int, int]:
    with sf.SoundFile(path) as handle:
        data = handle.read(always_2d=True, dtype="float32")
        sample_rate = int(handle.samplerate)
        subtype = handle.subtype or "PCM_16"
        channels = int(handle.channels)
        frames = int(handle.frames)
    return data, sample_rate, subtype, channels, frames


def _write_audio(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str,
) -> None:
    sf.write(path.as_posix(), samples, sample_rate, subtype=subtype)


def _eligible_inputs(paths: Iterable[Path], allowed: Sequence[str]) -> list[Path]:
    choices: list[Path] = []
    suffixes = tuple(s.lower() for s in allowed)
    for path in paths:
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        choices.append(path)
    return choices


def gather_audio_targets(path: Path, options: AudioPostOptions) -> list[Path]:
    if path.is_file():
        return _eligible_inputs([path], options.allowed_suffixes)
    if path.is_dir():
        return _eligible_inputs(sorted(path.iterdir()), options.allowed_suffixes)
    raise FileNotFoundError(f"No such file or directory: {path}")


def process_audio_file(
    path: Path,
    options: AudioPostOptions,
) -> SquelchFileResult:
    data, sample_rate, subtype, _channels, _frames = _load_audio(path)
    cleaned, noise_floor_db, threshold_db = apply_squelch(data, float(sample_rate), options.config)
    output_path = _derive_output_path(path, options)
    _write_audio(output_path, cleaned, sample_rate, subtype=subtype)
    samples_in = int(data.shape[0])
    samples_out = int(cleaned.shape[0])
    duration_in = samples_in / float(sample_rate)
    duration_out = samples_out / float(sample_rate)
    bytes_in = path.stat().st_size
    bytes_out = output_path.stat().st_size
    retained_ratio = samples_out / samples_in if samples_in else 0.0
    return SquelchFileResult(
        input_path=path,
        output_path=output_path,
        samples_in=samples_in,
        samples_out=samples_out,
        duration_in=duration_in,
        duration_out=duration_out,
        bytes_in=bytes_in,
        bytes_out=bytes_out,
        noise_floor_db=noise_floor_db,
        threshold_db=threshold_db,
        method=options.config.method,
        retained_ratio=retained_ratio,
    )


def process_audio_batch(
    targets: Sequence[Path],
    options: AudioPostOptions,
    *,
    progress_cb: Callable[[int, int, Path], None] | None = None,
) -> SquelchSummary:
    results: list[SquelchFileResult] = []
    errors: list[tuple[Path, Exception]] = []
    total = len(targets)
    for index, path in enumerate(targets, start=1):
        if progress_cb:
            with np.errstate(all="ignore"):
                progress_cb(index - 1, total, path)
        try:
            result = process_audio_file(path, options)
        except Exception as exc:  # pragma: no cover - surfaced to UI/CLI
            LOG.error("Audio post-processing failed for %s: %s", path, exc)
            errors.append((path, exc))
            continue
        results.append(result)
        if progress_cb:
            progress_cb(index, total, path)
    return SquelchSummary(results=results, errors=errors)
