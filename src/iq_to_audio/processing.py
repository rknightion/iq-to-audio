from __future__ import annotations

import contextlib
import io
import itertools
import logging
import math
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import fft, ifft
from scipy.signal import firwin, kaiser_beta

from .decoders import create_decoder
from .input_formats import InputFormatSpec, resolve_input_format
from .probe import SampleRateProbe, probe_sample_rate
from .progress import PhaseState, ProgressSink, ProgressTracker
from .utils import detect_center_frequency, resolve_ffmpeg_executable
from .visualize import save_stage_psd

LOG = logging.getLogger(__name__)

FFMPEG_HINT = (
    "ffmpeg executable not found. Install FFmpeg (e.g., `sudo apt install ffmpeg` (linux), "
    "`brew install ffmpeg` (macOS), `winget install ffmpeg` (Windows) "
    "or download from https://ffmpeg.org/download.html) and ensure it is on PATH."
)


@dataclass
class ProcessingConfig:
    in_path: Path
    target_freq: float = 0.0
    bandwidth: float = 12_500.0
    center_freq: float | None = None
    center_freq_source: str | None = None
    demod_mode: str = "nfm"
    fs_ch_target: float = 96_000.0
    deemph_us: float = 300.0
    agc_enabled: bool = True
    output_path: Path | None = None
    dump_iq_path: Path | None = None
    chunk_size: int = 1_048_576  # complex samples per block (~0.1 s @ 10 MS/s)
    filter_block: int = 65_536
    iq_order: str = "iq"
    probe_only: bool = False
    mix_sign_override: int | None = None
    plot_stages_path: Path | None = None
    fft_workers: int | None = None
    max_input_seconds: float | None = None
    input_container: str | None = None
    input_format: str | None = None
    input_format_source: str | None = None
    input_sample_rate: float | None = None


def tune_chunk_size(sample_rate: float, requested: int) -> int:
    """Heuristic to choose a performant chunk size without exhausting memory."""
    base = max(1, requested)
    if sample_rate <= 0:
        return base
    target_seconds = 0.25  # default to ~250 ms per block
    if sample_rate >= 2_000_000.0:
        target_seconds = 0.40
    if sample_rate >= 5_000_000.0:
        target_seconds = 0.50
    desired = int(round(sample_rate * target_seconds))
    if desired <= base:
        return base
    max_chunk = 4_194_304  # 4M complex samples (~32 MB)
    desired = min(max_chunk, max(base, desired))
    power = 1 << math.ceil(math.log2(desired))
    return int(min(max(power, base), max_chunk))


class IQReader:
    """Stream IQ samples from SDR baseband recordings using ffmpeg."""

    def __init__(
        self,
        path: Path,
        chunk_size: int,
        iq_order: str,
        input_format: InputFormatSpec,
        *,
        sample_rate: float | None = None,
    ):
        self.path = path
        self.chunk_size = chunk_size
        self.iq_order = iq_order
        self.input_format = input_format
        self.sample_rate = sample_rate
        self.proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_buffer: list[str] = []  # Store for error reporting
        self.frame_bytes = 8  # float32 stereo output
        self.input_bytes_per_frame = input_format.bytes_per_frame

    def __enter__(self) -> IQReader:
        ffmpeg_path = resolve_ffmpeg_executable()
        if ffmpeg_path is None:
            raise RuntimeError(FFMPEG_HINT)
        ffmpeg_executable = str(ffmpeg_path)
        cmd: list[str]
        if self.input_format.container == "raw":
            if self.sample_rate is None or self.sample_rate <= 0:
                raise ValueError(
                    "Raw IQ inputs require a sample rate override. Provide --input-sample-rate or set it in the GUI."
                )
            sample_rate = int(round(self.sample_rate))
            fmt = self.input_format.ffmpeg_input_format
            if fmt is None:
                raise ValueError(f"Unsupported raw input format: {self.input_format.label}")
            cmd = [
                ffmpeg_executable,
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "error",
                "-f",
                fmt,
                "-ac",
                "2",
                "-ar",
                str(sample_rate),
                "-i",
                str(self.path),
                "-f",
                "f32le",
                "-ac",
                "2",
                "-",
            ]
        else:
            cmd = [
                ffmpeg_executable,
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "error",
                "-ignore_length",
                "1",
                "-i",
                str(self.path),
                "-f",
                "f32le",
                "-ac",
                "2",
                "-",
            ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:  # pragma: no cover - dependency missing at runtime
            raise RuntimeError(FFMPEG_HINT) from exc
        except OSError as exc:  # pragma: no cover - runtime failure launching ffmpeg
            raise RuntimeError(f"Failed to launch ffmpeg: {exc}") from exc

        # Start stderr consumer thread to prevent deadlock on large IQ files.
        # In frozen builds, pipe buffers can fill up if stderr isn't consumed,
        # blocking ffmpeg while we're busy reading stdout.
        def _consume_stderr():
            if not self.proc or not self.proc.stderr:
                return
            try:
                for line in self.proc.stderr:
                    try:
                        decoded = line.decode("utf-8", errors="replace").strip()
                        if decoded:
                            self._stderr_buffer.append(decoded)
                            # Keep only last 50 lines to prevent unbounded growth
                            if len(self._stderr_buffer) > 50:
                                self._stderr_buffer.pop(0)
                    except Exception:
                        pass
            except Exception:
                pass

        self._stderr_thread = threading.Thread(
            target=_consume_stderr,
            name="IQReader-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.proc:
            # Close pipes first to signal subprocess
            if self.proc.stdout:
                with contextlib.suppress(Exception):
                    self.proc.stdout.close()
            if self.proc.stderr:
                with contextlib.suppress(Exception):
                    self.proc.stderr.close()

            # Gentle shutdown first
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if terminate didn't work (critical for frozen builds)
                LOG.warning("IQReader ffmpeg process did not terminate gracefully, forcing kill")
                self.proc.kill()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    LOG.error("Failed to kill IQReader ffmpeg subprocess (PID %d)", self.proc.pid)

            # Report any stderr messages if there were errors
            if exc_type is not None and self._stderr_buffer:
                LOG.debug("ffmpeg stderr: %s", " | ".join(self._stderr_buffer[-5:]))

            self.proc = None

        # Ensure stderr thread completes
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            block = self.read_block()
            if block is None or block.size == 0:
                break
            yield block

    def read_block(self) -> np.ndarray | None:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("IQReader has not been entered.")

        target_bytes = self.chunk_size * self.frame_bytes
        buffer = bytearray()
        while len(buffer) < target_bytes:
            chunk = self.proc.stdout.read(target_bytes - len(buffer))
            if not chunk:
                break
            buffer.extend(chunk)

        if not buffer:
            return None

        # Drop trailing incomplete frame if any.
        remainder = len(buffer) % self.frame_bytes
        if remainder:
            buffer = buffer[:-remainder]

        if not buffer:
            return None

        raw = np.frombuffer(buffer, dtype="<f4")
        i_samples, q_samples = self._extract_iq(raw)
        iq = i_samples.astype(np.float32, copy=False) + 1j * q_samples.astype(
            np.float32, copy=False
        )
        return iq.astype(np.complex64, copy=False)

    def _extract_iq(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.iq_order not in {"iq", "qi", "iq_inv", "qi_inv"}:
            raise ValueError(f"Unsupported iq_order '{self.iq_order}'")
        if self.iq_order.startswith("iq"):
            i = raw[0::2]
            q = raw[1::2]
        else:
            q = raw[0::2]
            i = raw[1::2]
        if self.iq_order.endswith("_inv"):
            q = -q
        return i, q


class ComplexOscillator:
    """Continuous complex exponential generator for frequency translation."""

    def __init__(self, freq_offset_hz: float, sample_rate: float):
        self.phase = 0.0
        self.increment = -2.0 * np.pi * freq_offset_hz / sample_rate

    def mix(self, samples: np.ndarray, sign: int) -> np.ndarray:
        if samples.size == 0:
            return samples
        n = np.arange(samples.size, dtype=np.float64)
        phases = self.phase + sign * self.increment * n
        osc = np.exp(1j * phases).astype(np.complex64)
        self.phase = (self.phase + sign * self.increment * samples.size) % (2.0 * np.pi)
        mixed = samples.astype(np.complex64, copy=False) * osc
        return np.asarray(mixed, dtype=np.complex64)


class OverlapSaveFIR:
    """FFT-based FIR filter suitable for long filters and streaming data."""

    def __init__(self, taps: np.ndarray, block_size: int, *, workers: int | None = None):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.taps = taps.astype(np.complex128)
        self.filter_len = len(taps)
        self.overlap = self.filter_len - 1
        self.block_size = block_size
        self.fft_size = 1 << math.ceil(math.log2(self.block_size + self.filter_len - 1))
        padded_taps = np.zeros(self.fft_size, dtype=np.complex128)
        padded_taps[: self.filter_len] = self.taps
        self.workers = workers if workers and workers > 1 else None
        self._fft_kwargs: dict[str, int] = {}
        if self.workers:
            try:
                fft(np.zeros(8, dtype=np.complex128), workers=self.workers)
            except TypeError:
                self.workers = None
            else:
                self._fft_kwargs = {"workers": self.workers}
        self.taps_fft = np.asarray(fft(padded_taps, **self._fft_kwargs))
        self.state = np.zeros(self.overlap, dtype=np.complex64)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples
        outputs: list[np.ndarray] = []
        cursor = 0
        arr = samples.astype(np.complex64)
        while cursor < arr.size:
            seg = arr[cursor : cursor + self.block_size]
            cursor += seg.size
            block = np.concatenate([self.state, seg]).astype(np.complex128)
            if block.size < self.fft_size:
                block = np.pad(block, (0, self.fft_size - block.size))
            spectrum = np.asarray(fft(block, **self._fft_kwargs))
            filtered = np.asarray(ifft(spectrum * self.taps_fft, **self._fft_kwargs))
            valid = filtered[self.overlap : self.overlap + seg.size]
            outputs.append(valid.astype(np.complex64))
            if self.overlap:
                if seg.size >= self.overlap:
                    self.state = seg[-self.overlap :].copy()
                else:
                    self.state = np.concatenate([self.state[seg.size :], seg]).astype(np.complex64)
        return np.concatenate(outputs)


class Decimator:
    def __init__(self, factor: int):
        self.factor = max(1, factor)
        self.offset = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self.factor == 1 or samples.size == 0:
            return samples
        start = (-self.offset) % self.factor
        decimated = samples[start :: self.factor]
        self.offset = (self.offset + samples.size) % self.factor
        return decimated


class IQDebugWriter:
    def __init__(self, path: Path | None, sample_rate: float):
        self.path = path
        self.sample_rate = sample_rate
        self.fd = path.open("wb") if path else None

    def write(self, samples: np.ndarray) -> None:
        if not self.fd or samples.size == 0:
            return
        data = samples.astype(np.complex64).view(np.float32)
        self.fd.write(data.tobytes())

    def close(self) -> None:
        if self.fd:
            self.fd.close()
            self.fd = None


class AudioWriter:
    """Pipe float32 audio to ffmpeg for final WAV encoding/resampling."""

    def __init__(self, output_path: Path, input_rate: float):
        ffmpeg_path = resolve_ffmpeg_executable()
        if ffmpeg_path is None:
            raise RuntimeError(FFMPEG_HINT)
        self.output_path = output_path
        self.input_rate = float(input_rate)
        # ffmpeg expects an integer sample rate; round to the nearest Hz.
        self.ffmpeg_rate = max(1, int(round(self.input_rate)))
        if abs(self.ffmpeg_rate - self.input_rate) > 0.5:
            LOG.debug(
                "Rounding writer input rate from %.6f to %d Hz for ffmpeg compatibility.",
                self.input_rate,
                self.ffmpeg_rate,
            )
        self.proc: subprocess.Popen[bytes] | None = None
        cmd = [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(self.ffmpeg_rate),
            "-i",
            "-",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "48000",
            str(output_path),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:  # pragma: no cover - dependency missing at runtime
            raise RuntimeError(FFMPEG_HINT) from exc
        except OSError as exc:  # pragma: no cover - runtime failure launching ffmpeg
            raise RuntimeError(f"Failed to launch ffmpeg: {exc}") from exc
        self.peak = 0.0
        self._queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._closed = False
        self._writer = threading.Thread(
            target=self._drain,
            name="AudioWriter",
            daemon=True,
        )
        self._writer.start()

    def write(self, samples: np.ndarray) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("AudioWriter is not ready.")
        if self._closed:
            raise RuntimeError("AudioWriter has already been closed.")
        if self._error:
            raise RuntimeError("ffmpeg writer failed") from self._error
        if samples.size == 0:
            return
        peak = float(np.max(np.abs(samples)))
        if peak > self.peak:
            self.peak = peak
        safe = np.clip(samples, -0.99, 0.99).astype(np.float32, copy=False)
        payload = safe.tobytes()
        self._queue.put(payload)
        if self._error:
            raise RuntimeError("ffmpeg writer failed") from self._error

    def _drain(self) -> None:
        stdin = self.proc.stdin if self.proc else None
        while True:
            payload = self._queue.get()
            if payload is None:
                break
            if self._error or not stdin:
                continue
            try:
                stdin.write(payload)
            except BrokenPipeError as exc:
                error = RuntimeError(
                    "ffmpeg exited unexpectedly while writing audio (Broken pipe). "
                    "Check that the preview/output path is writable."
                )
                error.__cause__ = exc
                self._error = error
            except BaseException as exc:  # pragma: no cover - defensive catch
                self._error = exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Signal writer thread to stop
        self._queue.put(None)
        if self._writer.is_alive():
            self._writer.join(timeout=10)

        if self.proc:
            # Close stdin to signal EOF to ffmpeg
            if self.proc.stdin:
                with contextlib.suppress(Exception):
                    self.proc.stdin.close()

            # Gentle then forceful shutdown
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if wait timed out (critical for frozen builds)
                LOG.warning("AudioWriter ffmpeg process did not exit gracefully, forcing kill")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    try:
                        self.proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        LOG.error(
                            "Failed to kill AudioWriter ffmpeg subprocess (PID %d)", self.proc.pid
                        )

            # Read any stderr messages
            if self.proc.stderr:
                try:
                    err = self.proc.stderr.read().decode("utf-8", errors="replace").strip()
                    if err:
                        LOG.debug("ffmpeg: %s", err)
                except Exception:
                    pass

            if self._error:
                raise RuntimeError("ffmpeg writer failed") from self._error

            self.proc = None


def _encode_iq_raw(samples: np.ndarray, codec: str) -> bytes:
    interleaved = np.empty(samples.size * 2, dtype=np.float32)
    interleaved[0::2] = samples.real
    interleaved[1::2] = samples.imag
    if codec == "pcm_f32le":
        return interleaved.astype("<f4", copy=False).tobytes()
    if codec == "pcm_s16le":
        scaled = np.clip(interleaved, -1.0, 0.999969) * 32767.0
        return scaled.astype("<i2", copy=False).tobytes()
    if codec == "pcm_u8":
        scaled = np.clip(interleaved, -1.0, 1.0)
        return np.round((scaled + 1.0) * 127.5).astype(np.uint8, copy=False).tobytes()
    raise ValueError(f"Unsupported raw codec {codec}")


class IQSliceWriter:
    """Write complex IQ slices while preserving container/codec."""

    _WAV_SUBTYPES = {
        "pcm_u8": "PCM_U8",
        "pcm_s16le": "PCM_16",
        "pcm_f32le": "FLOAT",
    }

    def __init__(self, output_path: Path, sample_rate: float, spec: InputFormatSpec):
        self.output_path = output_path
        self.sample_rate = float(sample_rate)
        self.spec = spec
        self.peak = 0.0
        self._file: sf.SoundFile | None = None
        self._fd: io.BufferedWriter | None = None
        if spec.container == "wav":
            subtype = self._WAV_SUBTYPES.get(spec.codec)
            if subtype is None:
                raise ValueError(f"Unsupported WAV codec for slices: {spec.codec}")
            self._file = sf.SoundFile(
                output_path,
                mode="w",
                samplerate=max(1, int(round(self.sample_rate))),
                channels=2,
                subtype=subtype,
                format="WAV",
            )
        else:
            self._fd = output_path.open("wb")

    def write(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        peak = float(np.max(np.abs(samples)))
        if peak > self.peak:
            self.peak = peak
        if self.spec.container == "wav":
            assert self._file is not None
            interleaved = np.column_stack((samples.real, samples.imag)).astype(
                np.float32, copy=False
            )
            self._file.write(interleaved)
        else:
            assert self._fd is not None
            payload = _encode_iq_raw(samples, self.spec.codec)
            self._fd.write(payload)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._fd is not None:
            self._fd.close()
            self._fd = None


def design_channel_filter(sample_rate: float, bandwidth: float, decimation: int) -> np.ndarray:
    guard = max(1_000.0, bandwidth * 0.5)
    cutoff = min(
        bandwidth * 0.5 * 1.05,
        (sample_rate / (2.0 * max(decimation, 1))) * 0.9,
    )
    if cutoff <= 0:
        raise ValueError("Invalid cutoff frequency for channel filter.")
    transition = guard
    width = transition / sample_rate
    ripple_db = 80.0
    num_taps = int(np.clip(4.0 / max(width, 1e-8), 1024, 32768))
    if num_taps % 2 == 0:
        num_taps += 1
    beta = kaiser_beta(ripple_db)
    taps = firwin(
        num_taps,
        cutoff=cutoff,
        window=("kaiser", beta),
        fs=sample_rate,
    )
    return np.asarray(taps, dtype=np.float64)


def choose_mix_sign(
    warmup: np.ndarray,
    sample_rate: float,
    freq_offset: float,
    taps: np.ndarray,
    decimation: int,
) -> int:
    if warmup.size == 0:
        return 1
    # Limit snippet length so mixer sign probing stays responsive on high-rate captures.
    max_len = max(int(sample_rate * 0.05), len(taps) * 4, 131_072)
    snippet_len = min(warmup.size, max_len)
    if snippet_len < len(taps):
        snippet_len = min(warmup.size, len(taps) * 2)
    snippet = warmup[:snippet_len].astype(np.complex64, copy=False)
    n = np.arange(snippet.size, dtype=np.float64)
    decim = max(decimation, 1)
    block_size = min(snippet.size, max(len(taps), 16_384))

    best_sign = 1
    best_power = -np.inf
    for sign in (1, -1):
        osc = np.exp(-1j * sign * 2.0 * np.pi * freq_offset * n / sample_rate).astype(
            np.complex64, copy=False
        )
        mixed = snippet * osc
        fir = OverlapSaveFIR(taps, block_size)
        filtered = fir.process(mixed)
        decimated = filtered[::decim]
        if decimated.size == 0:
            power = -np.inf
        else:
            discard = min(len(taps), decimated.size // 4)
            useful = decimated[discard:]
            if useful.size == 0:
                useful = decimated
            power = float(np.mean(np.abs(useful) ** 2))
        if power > best_power:
            best_power = power
            best_sign = sign
    return best_sign


@dataclass
class ProcessingResult:
    sample_rate_probe: SampleRateProbe
    center_freq: float
    target_freq: float
    freq_offset: float
    decimation: int
    fs_channel: float
    mix_sign: int
    audio_peak: float


class ProcessingCancelled(RuntimeError):  # noqa: N818
    """Raised when processing is aborted early by user request."""


class ProcessingPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._cancelled = False
        self._resolved_chunk_size: int | None = None
        self._resolved_fft_workers: int | None = None
        self._input_spec: InputFormatSpec | None = None

    def cancel(self) -> None:
        self._cancelled = True

    def _is_pass_through_mode(self) -> bool:
        mode = (self.config.demod_mode or "").lower()
        return mode in {"none", "pass", "iq"}

    def _resolve_fft_workers(self) -> int | None:
        if self._resolved_fft_workers is not None:
            return self._resolved_fft_workers

        configured = self.config.fft_workers
        if configured is not None:
            self._resolved_fft_workers = None if configured <= 1 else configured
            return self._resolved_fft_workers

        cpu_count = os.cpu_count() or 1

        # In frozen builds, be more conservative with worker allocation to prevent
        # thread contention with Qt event loop, subprocess I/O, and NumPy's own
        # thread pools (OpenMP/MKL/OpenBLAS).
        if getattr(sys, "frozen", False):
            # Leave significant headroom for other threads
            # Use 1 worker on low-core systems, half of cores otherwise
            max_workers = 1 if cpu_count <= 4 else max(2, cpu_count // 2)
            max_cap = 8  # Cap at 8 workers even on high-core systems
        else:
            # Development builds can be more aggressive
            max_workers = cpu_count - 1
            max_cap = 12

        if cpu_count <= 2:
            self._resolved_fft_workers = None
        else:
            self._resolved_fft_workers = min(max_cap, max(2, max_workers))

        LOG.debug(
            "FFT workers: auto=%d (cpu_count=%d, frozen=%s)",
            self._resolved_fft_workers or 0,
            cpu_count,
            getattr(sys, "frozen", False),
        )
        return self._resolved_fft_workers

    def _effective_chunk_size(self, sample_rate: float) -> int:
        if self._resolved_chunk_size is not None:
            return self._resolved_chunk_size
        chunk = tune_chunk_size(sample_rate, self.config.chunk_size)
        self._resolved_chunk_size = chunk
        return chunk

    def run(self, progress_sink: ProgressSink | None = None) -> ProcessingResult:
        tracker = ProgressTracker(progress_sink)
        if self._input_spec is None:
            spec, source = resolve_input_format(
                self.config.in_path,
                requested=self.config.input_format,
                container_hint=self.config.input_container,
            )
            self._input_spec = spec
            if not self.config.input_format_source:
                self.config.input_format_source = source
            if not self.config.input_container:
                self.config.input_container = spec.container
            if not self.config.input_format:
                self.config.input_format = spec.codec
        assert self._input_spec is not None
        input_spec = self._input_spec
        pass_through = self._is_pass_through_mode()

        header_bytes = 44 if input_spec.container == "wav" else 0
        frame_bytes = input_spec.bytes_per_frame
        output_path: Path | None = None
        cancel_logged = False
        last_status: str | None = None

        def _request_cancel() -> None:
            self._cancelled = True
            tracker.cancel()
            tracker.status("Cancelling…")

        def _check_cancel(stage: str = "") -> None:
            nonlocal cancel_logged
            if self._cancelled or tracker.cancelled:
                self._cancelled = True
                if not tracker.cancelled:
                    tracker.cancel()
                    tracker.status("Cancelling…")
                if not cancel_logged:
                    if stage:
                        LOG.info("Processing cancelled during %s.", stage)
                    else:
                        LOG.info("Processing cancelled by user.")
                    cancel_logged = True
                raise ProcessingCancelled("Processing cancelled by user.")

        stage_labels = {
            "design": "design filter",
            "init": "init dsp",
            "warmup": "warm-up",
            "channel": "channel",
            "dump": "dump IQ",
            "demod": f"demod {self.config.demod_mode.upper()}",
            "encode": "write audio",
            "finalize": "flush outputs",
            "complete": "Processing complete",
        }

        def _status_text(key: str, *, chunk: int | None = None) -> str:
            base = stage_labels.get(key, key)
            if chunk is None:
                return base
            return f"C{chunk} {base}"

        def report(message: str) -> None:
            nonlocal last_status
            tracker.status(message)
            if message != last_status:
                LOG.info(message)
                last_status = message

        if progress_sink is not None:
            with contextlib.suppress(AttributeError):
                progress_sink.set_cancel_callback(_request_cancel)

        manual_rate = self.config.input_sample_rate
        if manual_rate is not None and manual_rate <= 0:
            raise ValueError("Input sample rate override must be positive.")

        try:
            if input_spec.container == "raw":
                if manual_rate is None:
                    raise ValueError(
                        "Raw IQ inputs require --input-sample-rate (CLI) or a manual entry in the GUI."
                    )
                sample_rate = float(manual_rate)
                probe = SampleRateProbe(ffprobe=None, header=None, wave=sample_rate)
            else:
                probe = probe_sample_rate(self.config.in_path)
                if manual_rate is not None:
                    sample_rate = float(manual_rate)
                else:
                    try:
                        sample_rate = probe.value
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "Unable to determine input sample rate automatically. Provide --input-sample-rate or enter it manually in the GUI."
                        ) from exc

            preview_seconds = self.config.max_input_seconds
            if preview_seconds is not None and preview_seconds <= 0:
                preview_seconds = None
            max_input_samples: int | None = None
            if preview_seconds is not None and sample_rate > 0:
                max_input_samples = max(
                    1, int(math.floor(preview_seconds * sample_rate))
                )

            if self.config.target_freq <= 0 and not self.config.probe_only:
                raise ValueError("Target frequency must be positive. Provide --ft or use --interactive.")
            if self.config.bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")

            center_freq = self.config.center_freq
            center_source = (
                self.config.center_freq_source
                if self.config.center_freq_source
                else ("config" if center_freq is not None else "unavailable")
            )

            def _describe_source(source: str) -> str:
                if ":" in source:
                    prefix, suffix = source.split(":", 1)
                    return f"{prefix} ({suffix})"
                return source

            if center_freq is None:
                detection = detect_center_frequency(self.config.in_path)
                if detection.value is None:
                    raise ValueError(
                        "Center frequency not supplied and could not be determined from metadata or filename. "
                        "Use --fc to provide it explicitly."
                    )
                center_freq = detection.value
                center_source = detection.source
                self.config.center_freq = center_freq
                self.config.center_freq_source = center_source
                LOG.info(
                    "Center frequency detected via %s.",
                    _describe_source(center_source),
                )

            target_freq = self.config.target_freq if self.config.target_freq > 0 else center_freq
            freq_offset = target_freq - center_freq

            decimation = max(1, int(round(sample_rate / self.config.fs_ch_target)))
            fs_channel = sample_rate / decimation
            if fs_channel > self.config.fs_ch_target * 1.5:
                decimation = int(math.floor(sample_rate / self.config.fs_ch_target))
                decimation = max(decimation, 1)
                fs_channel = sample_rate / decimation

            LOG.info(
                "Input sample rate %.2f Hz (ffprobe=%s, header=%s, wave=%s).",
                sample_rate,
                f"{probe.ffprobe:.2f}" if probe.ffprobe else "n/a",
                f"{probe.header:.2f}" if probe.header else "n/a",
                f"{probe.wave:.2f}" if probe.wave else "n/a",
            )
            LOG.info(
                "Center frequency %.0f Hz, target %.0f Hz, offset %.0f Hz.",
                center_freq,
                target_freq,
                freq_offset,
            )
            LOG.info(
                "Channel decimation factor %d -> %.2f Hz complex rate.",
                decimation,
                fs_channel,
            )
            LOG.info("Using %s demodulator.", self.config.demod_mode.upper())
            LOG.info(
                "AGC %s.",
                "enabled" if self.config.agc_enabled else "disabled",
            )

            try:
                file_size = self.config.in_path.stat().st_size
            except OSError:
                file_size = 0
            payload_bytes = max(file_size - header_bytes, 0)
            total_input_samples = max(payload_bytes / frame_bytes, 0.0)
            if max_input_samples is not None:
                if total_input_samples > 0:
                    total_input_samples = float(
                        min(total_input_samples, max_input_samples)
                    )
                else:
                    total_input_samples = float(max_input_samples)
            estimated_channel_samples = (
                total_input_samples / max(decimation, 1)
            )
            duration_seconds = total_input_samples / sample_rate if sample_rate > 0 else 0.0
            chunk_size = self._effective_chunk_size(sample_rate)
            estimated_chunks = (
                int(math.ceil(total_input_samples / chunk_size))
                if total_input_samples > 0
                else 0
            )
            if max_input_samples is not None and preview_seconds is not None:
                duration_seconds = min(duration_seconds, preview_seconds)
            estimated_audio_samples = max(duration_seconds * 48_000.0, 0.0)
            if max_input_samples is not None and preview_seconds is not None:
                limited_duration = (
                    duration_seconds if duration_seconds > 0 else preview_seconds
                )
                LOG.info(
                    "Preview constrained to %.2f s of IQ (~%.3f M complex samples).",
                    limited_duration,
                    total_input_samples / 1e6,
                )
            if estimated_chunks > 0:
                LOG.info(
                    "Expecting approximately %d processing chunks (chunk size %d samples, %.2f s of IQ).",
                    estimated_chunks,
                    chunk_size,
                    duration_seconds,
                )
            else:
                LOG.info(
                    "Processing chunk size %d samples; input duration could not be estimated from metadata.",
                    chunk_size,
                )
            if chunk_size != self.config.chunk_size:
                approx_duration = chunk_size / sample_rate if sample_rate > 0 else 0.0
                LOG.info(
                    "Adjusted chunk size from %d to %d samples (~%.3f s) for improved throughput.",
                    self.config.chunk_size,
                    chunk_size,
                    approx_duration,
                )
            fft_workers = self._resolve_fft_workers()
            if fft_workers:
                LOG.info("Using up to %d FFT workers where supported.", fft_workers)

            phases: list[PhaseState] = [
                PhaseState("ingest", "Ingest IQ", total_input_samples, unit="samples"),
                PhaseState("channel", "Channelize", estimated_channel_samples, unit="samples"),
                PhaseState("demod", "Demodulate", estimated_channel_samples, unit="samples"),
                PhaseState("encode", "Encode Audio", estimated_audio_samples, unit="samples"),
            ]
            if self.config.dump_iq_path:
                phases.insert(
                    3,
                    PhaseState(
                        "dump_iq",
                        "Write IQ Dump",
                        estimated_channel_samples,
                        unit="samples",
                    ),
                )
            tracker.start(phases)
            report(_status_text("design"))
            _check_cancel("initialization")

            taps = design_channel_filter(sample_rate, self.config.bandwidth, decimation)
            LOG.info("Designed FIR channel filter with %d taps.", len(taps))
            report(_status_text("init"))
            _check_cancel("initialization")

            oscillator = ComplexOscillator(freq_offset, sample_rate)
            channel_filter = OverlapSaveFIR(
                taps, self.config.filter_block, workers=fft_workers
            )
            decimator = Decimator(decimation)
            decoder = None
            if not pass_through:
                decoder = create_decoder(
                    self.config.demod_mode,
                    deemph_us=self.config.deemph_us,
                    agc_enabled=self.config.agc_enabled,
                )
                decoder.setup(fs_channel)
            iq_writer = IQDebugWriter(self.config.dump_iq_path, fs_channel)

            output_path = (
                self.config.output_path
                if self.config.output_path
                else self._default_output_path()
            )

            stage_snapshots: dict[str, tuple[np.ndarray, float]] = {}
            processed_samples = 0
            slice_writer: IQSliceWriter | None = None
            audio_writer: AudioWriter | None = None
            if not self.config.probe_only:
                if pass_through:
                    slice_writer = IQSliceWriter(output_path, fs_channel, input_spec)
                else:
                    audio_writer = AudioWriter(output_path, fs_channel)

            reader_sample_rate = sample_rate if input_spec.container == "raw" else None
            with IQReader(
                self.config.in_path,
                chunk_size,
                self.config.iq_order,
                input_spec,
                sample_rate=reader_sample_rate,
            ) as reader:
                iterator = iter(reader)
                warmup = next(iterator, None)
                if warmup is None:
                    raise RuntimeError("Input stream produced no samples.")
                _check_cancel("warm-up")

                limit_exhausted = False
                if max_input_samples is not None and warmup.size > max_input_samples:
                    warmup = warmup[:max_input_samples]
                    limit_exhausted = True

                mix_sign = (
                    self.config.mix_sign_override
                    if self.config.mix_sign_override in (1, -1)
                    else choose_mix_sign(warmup, sample_rate, freq_offset, taps, decimation)
                )
                LOG.info("Selected mixer sign %d based on warm-up snippet.", mix_sign)
                report(_status_text("warmup"))
                _check_cancel("warm-up")

                if self.config.probe_only:
                    _check_cancel("probe-only")
                    if self.config.plot_stages_path:
                        stage_snapshots["input"] = (warmup.copy(), sample_rate)
                    tracker.advance("ingest", warmup.size)
                    report("Probe-only inspection complete")
                    if decoder is not None:
                        decoder.finalize()
                    iq_writer.close()
                    return ProcessingResult(
                        sample_rate_probe=probe,
                        center_freq=center_freq,
                        target_freq=target_freq,
                        freq_offset=freq_offset,
                        decimation=decimation,
                        fs_channel=fs_channel,
                        mix_sign=mix_sign,
                        audio_peak=0.0,
                    )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    processed_samples = 0
                    for idx, raw_block in enumerate(itertools.chain((warmup,), iterator)):
                        block = raw_block
                        if max_input_samples is not None:
                            remaining = max_input_samples - processed_samples
                            if remaining <= 0:
                                limit_exhausted = True
                                break
                            if block.size > remaining:
                                block = block[:remaining]
                                limit_exhausted = True
                        if block.size == 0:
                            continue
                        _check_cancel(f"chunk {idx + 1}")
                        tracker.advance("ingest", block.size)
                        processed_samples += block.size
                        if self.config.plot_stages_path and idx == 0 and not self.config.probe_only:
                            stage_snapshots["input"] = (block.copy(), sample_rate)
                        report(_status_text("channel", chunk=idx + 1))
                        mixed = oscillator.mix(block, mix_sign)
                        if self.config.plot_stages_path and idx == 0:
                            stage_snapshots["mixed"] = (mixed.copy(), sample_rate)

                        filtered = channel_filter.process(mixed)
                        if self.config.plot_stages_path and idx == 0:
                            stage_snapshots["filtered"] = (filtered.copy(), sample_rate)

                        decimated = decimator.process(filtered)
                        tracker.advance("channel", float(decimated.size))
                        if self.config.dump_iq_path:
                            report(_status_text("dump", chunk=idx + 1))
                            iq_writer.write(decimated)
                            tracker.advance("dump_iq", float(decimated.size))
                        if self.config.plot_stages_path and idx == 0:
                            stage_snapshots["decimated"] = (decimated.copy(), fs_channel)

                        baseband_power = np.mean(np.abs(decimated) ** 2) if decimated.size else 0.0
                        LOG.debug(
                            "Block %d: mixed=%d samples, decimated=%d, power=%.3e",
                            idx,
                            block.size,
                            decimated.size,
                            baseband_power,
                        )

                        if pass_through:
                            report(_status_text("demod", chunk=idx + 1))
                            if slice_writer is None:
                                raise RuntimeError("IQ slice writer missing during pass-through mode.")
                            slice_writer.write(decimated)
                            tracker.advance("demod", float(decimated.size))
                        else:
                            report(_status_text("demod", chunk=idx + 1))
                            if decoder is None or audio_writer is None:
                                raise RuntimeError("Decoder or audio writer unavailable during demodulation.")
                            audio, stats = decoder.process(decimated)
                            intermediates = decoder.intermediates()
                            if intermediates:
                                first_stage = next(iter(intermediates.values()))
                                demod_len = float(first_stage[0].size)
                            else:
                                demod_len = float(decimated.size)
                            tracker.advance("demod", demod_len)
                            if self.config.plot_stages_path and idx == 0 and intermediates:
                                for name, (buf, rate) in intermediates.items():
                                    stage_snapshots[name] = (buf.copy(), rate)
                            LOG.debug(
                                "Demod chunk %d: %d samples, rms=%.2f dBFS",
                                idx,
                                int(demod_len),
                                stats.rms_dbfs if stats else float("nan"),
                            )

                            report(_status_text("encode", chunk=idx + 1))
                            audio_writer.write(audio)
                            _check_cancel(f"chunk {idx + 1} encode")
                            if audio.size:
                                duration = audio.size / max(fs_channel, 1e-9)
                                tracker.advance("encode", duration * 48_000.0)
                        if (
                            max_input_samples is not None
                            and processed_samples >= max_input_samples
                        ):
                            limit_exhausted = True
                            break
                finally:
                    report(_status_text("finalize"))
                    if decoder is not None:
                        decoder.finalize()
                    iq_writer.close()
                    if audio_writer is not None:
                        audio_writer.close()
                    if slice_writer is not None:
                        slice_writer.close()

            if limit_exhausted and preview_seconds is not None:
                processed_duration = (
                    processed_samples / sample_rate if sample_rate > 0 else 0.0
                )
                LOG.info(
                    "Stopped after %.2f s due to preview limit (processed %.3f M complex samples).",
                    processed_duration if processed_duration > 0 else preview_seconds,
                    processed_samples / 1e6,
                )

            if self.config.plot_stages_path and not self.config.probe_only:
                try:
                    save_stage_psd(stage_snapshots, self.config.plot_stages_path, center_freq)
                    LOG.info("Saved stage PSD plots to %s", self.config.plot_stages_path)
                except Exception as exc:  # pragma: no cover - plotting errors logged
                    LOG.warning("Failed to save stage plots: %s", exc)

            peak_source = 0.0
            if pass_through and slice_writer is not None:
                peak_source = slice_writer.peak
                LOG.info(
                    "IQ slice peak magnitude %.2f dBFS (complex).",
                    20.0 * math.log10(max(slice_writer.peak, 1e-6)),
                )
            elif not pass_through and audio_writer is not None:
                peak_source = audio_writer.peak
                LOG.info(
                    "Audio peak level %.2f dBFS.",
                    20.0 * math.log10(max(audio_writer.peak, 1e-6)),
                )
            report(_status_text("complete"))

            return ProcessingResult(
                sample_rate_probe=probe,
                center_freq=center_freq,
                target_freq=target_freq,
                freq_offset=freq_offset,
                decimation=decimation,
                fs_channel=fs_channel,
                mix_sign=mix_sign,
                audio_peak=peak_source,
            )
        except ProcessingCancelled:
            if not self.config.probe_only and output_path:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    LOG.debug("Failed to remove cancelled output %s", output_path)
            raise
        finally:
            tracker.close()

    def _default_output_path(self) -> Path:
        ft = int(self.config.target_freq)
        if self._is_pass_through_mode():
            spec = self._input_spec
            in_suffix = self.config.in_path.suffix
            wav_suffixes = {".wav", ".wave", ".wv", ".rf64"}
            if spec and spec.container == "wav":
                ext = in_suffix if in_suffix.lower() in wav_suffixes else ".wav"
            elif spec and spec.container == "raw":
                codec_ext = {
                    "pcm_u8": ".cu8",
                    "pcm_s16le": ".cs16",
                    "pcm_f32le": ".cf32",
                }.get(spec.codec, ".raw")
                ext = in_suffix or codec_ext
            else:
                ext = in_suffix or ".wav"
            return self.config.in_path.with_name(f"slice_{ft}{ext}")
        return self.config.in_path.with_name(f"audio_{ft}_48k.wav")
