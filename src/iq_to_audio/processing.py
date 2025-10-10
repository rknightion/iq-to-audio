from __future__ import annotations

import itertools
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy import signal

from .decoders import create_decoder
from .probe import SampleRateProbe, probe_sample_rate
from .progress import PhaseState, ProgressSink, ProgressTracker
from .utils import detect_center_frequency
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
    center_freq: Optional[float] = None
    demod_mode: str = "nfm"
    fs_ch_target: float = 96_000.0
    deemph_us: float = 300.0
    squelch_dbfs: Optional[float] = None
    silence_trim: bool = False
    squelch_enabled: bool = True
    agc_enabled: bool = True
    output_path: Optional[Path] = None
    dump_iq_path: Optional[Path] = None
    chunk_size: int = 1_048_576  # complex samples per block (~0.1 s @ 10 MS/s)
    filter_block: int = 65_536
    iq_order: str = "iq"
    probe_only: bool = False
    mix_sign_override: Optional[int] = None
    plot_stages_path: Optional[Path] = None


class IQReader:
    """Stream IQ samples from an SDR++ baseband WAV using ffmpeg."""

    def __init__(self, path: Path, chunk_size: int, iq_order: str):
        self.path = path
        self.chunk_size = chunk_size
        self.proc: Optional[subprocess.Popen] = None
        self.iq_order = iq_order
        self.frame_bytes = 4  # 2 channels * int16

    def __enter__(self) -> "IQReader":
        if not shutil.which("ffmpeg"):
            raise RuntimeError(FFMPEG_HINT)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-ignore_length",
            "1",
            "-i",
            str(self.path),
            "-f",
            "s16le",
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.proc:
            if self.proc.stdout:
                self.proc.stdout.close()
            if self.proc.stderr:
                self.proc.stderr.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        self.proc = None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            block = self.read_block()
            if block is None or block.size == 0:
                break
            yield block

    def read_block(self) -> Optional[np.ndarray]:
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

        raw = np.frombuffer(buffer, dtype="<i2")
        i_samples, q_samples = self._extract_iq(raw)
        iq = i_samples.astype(np.float32) / 32768.0 + 1j * (
            q_samples.astype(np.float32) / 32768.0
        )
        return iq

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
        self.phase = (self.phase + sign * self.increment * samples.size) % (
            2.0 * np.pi
        )
        return samples * osc


class OverlapSaveFIR:
    """FFT-based FIR filter suitable for long filters and streaming data."""

    def __init__(self, taps: np.ndarray, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.taps = taps.astype(np.complex128)
        self.filter_len = len(taps)
        self.overlap = self.filter_len - 1
        self.block_size = block_size
        self.fft_size = 1 << math.ceil(
            math.log2(self.block_size + self.filter_len - 1)
        )
        padded_taps = np.zeros(self.fft_size, dtype=np.complex128)
        padded_taps[: self.filter_len] = self.taps
        self.taps_fft = np.fft.fft(padded_taps)
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
            spectrum = np.fft.fft(block)
            filtered = np.fft.ifft(spectrum * self.taps_fft)
            valid = filtered[self.overlap : self.overlap + seg.size]
            outputs.append(valid.astype(np.complex64))
            if self.overlap:
                if seg.size >= self.overlap:
                    self.state = seg[-self.overlap :].copy()
                else:
                    self.state = np.concatenate(
                        [self.state[seg.size :], seg]
                    ).astype(np.complex64)
        return np.concatenate(outputs)


class Decimator:
    def __init__(self, factor: int):
        self.factor = max(1, factor)
        self.offset = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self.factor == 1 or samples.size == 0:
            return samples
        start = (-self.offset) % self.factor
        decimated = samples[start:: self.factor]
        self.offset = (self.offset + samples.size) % self.factor
        return decimated


class IQDebugWriter:
    def __init__(self, path: Optional[Path], sample_rate: float):
        self.path = path
        self.sample_rate = sample_rate
        self.fd = open(path, "wb") if path else None

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
        if not shutil.which("ffmpeg"):
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
        cmd = [
            "ffmpeg",
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

    def write(self, samples: np.ndarray) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("AudioWriter is not ready.")
        if samples.size == 0:
            return
        peak = float(np.max(np.abs(samples)))
        if peak > self.peak:
            self.peak = peak
        safe = np.clip(samples, -0.99, 0.99).astype(np.float32, copy=False)
        try:
            self.proc.stdin.write(safe.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(
                "ffmpeg exited unexpectedly while writing audio (Broken pipe). "
                "Check that the preview/output path is writable."
            ) from exc

    def close(self) -> None:
        if not self.proc:
            return
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)
        if self.proc.stderr:
            err = self.proc.stderr.read().decode("utf-8").strip()
            if err:
                LOG.debug("ffmpeg: %s", err)
        self.proc = None


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
    beta = signal.kaiser_beta(ripple_db)
    taps = signal.firwin(
        num_taps,
        cutoff=cutoff,
        window=("kaiser", beta),
        fs=sample_rate,
    )
    return taps.astype(np.float64)


def choose_mix_sign(
    warmup: np.ndarray,
    sample_rate: float,
    freq_offset: float,
    taps: np.ndarray,
    decimation: int,
) -> int:
    snippet_len = min(warmup.size, int(sample_rate * 0.5))
    if snippet_len < taps.size:
        snippet_len = warmup.size
    snippet = warmup[:snippet_len]
    n = np.arange(snippet.size, dtype=np.float64)
    best_sign = 1
    best_power = -np.inf
    for sign in (1, -1):
        osc = np.exp(-1j * sign * 2.0 * np.pi * freq_offset * n / sample_rate)
        mixed = snippet * osc
        filtered = signal.lfilter(taps, [1.0], mixed)
        decimated = filtered[:: max(decimation, 1)]
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


class ProcessingCancelled(RuntimeError):
    """Raised when processing is aborted early by user request."""


class ProcessingPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self, progress_sink: Optional[ProgressSink] = None) -> ProcessingResult:
        tracker = ProgressTracker(progress_sink)
        header_bytes = 44  # baseline WAV header size (approximate)
        frame_bytes = 4  # 2 channels * int16
        output_path: Optional[Path] = None
        cancel_logged = False
        last_status: Optional[str] = None

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

        def report(message: str) -> None:
            nonlocal last_status
            tracker.status(message)
            if message != last_status:
                LOG.info(message)
                last_status = message

        if progress_sink is not None:
            try:
                progress_sink.set_cancel_callback(_request_cancel)
            except AttributeError:
                pass

        try:
            probe = probe_sample_rate(self.config.in_path)
            try:
                sample_rate = probe.value
            except RuntimeError as exc:
                raise RuntimeError(
                    "Unable to determine input sample rate. Ensure ffprobe/ffmpeg is installed or the WAV header is valid."
                ) from exc

            if self.config.target_freq <= 0 and not self.config.probe_only:
                raise ValueError("Target frequency must be positive. Provide --ft or use --interactive.")
            if self.config.bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")

            center_freq = self.config.center_freq
            center_source = "config"

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
                "Squelch %s, silence trim %s, AGC %s.",
                "enabled" if self.config.squelch_enabled else "disabled",
                "enabled" if self.config.silence_trim else "disabled",
                "enabled" if self.config.agc_enabled else "disabled",
            )
            if not self.config.squelch_enabled and self.config.silence_trim:
                LOG.warning("silence_trim is ignored because squelch is disabled.")

            try:
                file_size = self.config.in_path.stat().st_size
            except OSError:
                file_size = 0
            payload_bytes = max(file_size - header_bytes, 0)
            total_input_samples = max(payload_bytes / frame_bytes, 0.0)
            estimated_channel_samples = (
                total_input_samples / max(decimation, 1)
            )
            duration_seconds = (
                total_input_samples / sample_rate if sample_rate > 0 else 0.0
            )
            estimated_audio_samples = max(duration_seconds * 48_000.0, 0.0)

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
            report("Designing channel filter")
            _check_cancel("initialization")

            taps = design_channel_filter(sample_rate, self.config.bandwidth, decimation)
            LOG.info("Designed FIR channel filter with %d taps.", len(taps))
            report("Initializing DSP stages")
            _check_cancel("initialization")

            oscillator = ComplexOscillator(freq_offset, sample_rate)
            channel_filter = OverlapSaveFIR(taps, self.config.filter_block)
            decimator = Decimator(decimation)
            decoder = create_decoder(
                self.config.demod_mode,
                deemph_us=self.config.deemph_us,
                squelch_dbfs=self.config.squelch_dbfs,
                silence_trim=self.config.silence_trim,
                squelch_enabled=self.config.squelch_enabled,
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

            with IQReader(self.config.in_path, self.config.chunk_size, self.config.iq_order) as reader:
                iterator = iter(reader)
                warmup = next(iterator, None)
                if warmup is None:
                    raise RuntimeError("Input stream produced no samples.")
                _check_cancel("warm-up")

                if self.config.plot_stages_path and not self.config.probe_only:
                    stage_snapshots["input"] = (warmup.copy(), sample_rate)

                mix_sign = (
                    self.config.mix_sign_override
                    if self.config.mix_sign_override in (1, -1)
                    else choose_mix_sign(warmup, sample_rate, freq_offset, taps, decimation)
                )
                LOG.info("Selected mixer sign %d based on warm-up snippet.", mix_sign)
                report("Warm-up complete; processing stream")
                _check_cancel("warm-up")

                if self.config.probe_only:
                    _check_cancel("probe-only")
                    tracker.advance("ingest", warmup.size)
                    report("Probe-only inspection complete")
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
                audio_writer = AudioWriter(output_path, fs_channel)
                try:
                    for idx, block in enumerate(itertools.chain((warmup,), iterator)):
                        _check_cancel(f"chunk {idx + 1}")
                        tracker.advance("ingest", block.size)
                        report(f"Chunk {idx + 1}: channelizing")
                        mixed = oscillator.mix(block, mix_sign)
                        if self.config.plot_stages_path and idx == 0:
                            stage_snapshots["mixed"] = (mixed.copy(), sample_rate)

                        filtered = channel_filter.process(mixed)
                        if self.config.plot_stages_path and idx == 0:
                            stage_snapshots["filtered"] = (filtered.copy(), sample_rate)

                        decimated = decimator.process(filtered)
                        tracker.advance("channel", float(decimated.size))
                        if self.config.dump_iq_path:
                            report(f"Chunk {idx + 1}: writing IQ dump")
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

                        report(
                            f"Chunk {idx + 1}: demodulating ({self.config.demod_mode.upper()})"
                        )
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
                            "Demod chunk %d: %d samples, rms=%.2f dBFS, gate %.2f dBFS, dropped=%s",
                            idx,
                            int(demod_len),
                            stats.rms_dbfs if stats else float("nan"),
                            stats.gate_threshold_dbfs if stats else float("nan"),
                            stats.dropped if stats else False,
                        )

                        report(f"Chunk {idx + 1}: encoding audio")
                        audio_writer.write(audio)
                        _check_cancel(f"chunk {idx + 1} encode")
                        if audio.size:
                            duration = audio.size / max(fs_channel, 1e-9)
                            tracker.advance("encode", duration * 48_000.0)
                finally:
                    report("Finalizing outputs")
                    decoder.finalize()
                    iq_writer.close()
                    audio_writer.close()

            if self.config.plot_stages_path and not self.config.probe_only:
                try:
                    save_stage_psd(stage_snapshots, self.config.plot_stages_path, center_freq)
                    LOG.info("Saved stage PSD plots to %s", self.config.plot_stages_path)
                except Exception as exc:  # pragma: no cover - plotting errors logged
                    LOG.warning("Failed to save stage plots: %s", exc)

            LOG.info(
                "Audio peak level %.2f dBFS.",
                20.0 * math.log10(max(audio_writer.peak, 1e-6)),
            )
            report("Processing complete")

            return ProcessingResult(
                sample_rate_probe=probe,
                center_freq=center_freq,
                target_freq=target_freq,
                freq_offset=freq_offset,
                decimation=decimation,
                fs_channel=fs_channel,
                mix_sign=mix_sign,
                audio_peak=audio_writer.peak,
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
        return self.config.in_path.with_name(f"audio_{ft}_48k.wav")
