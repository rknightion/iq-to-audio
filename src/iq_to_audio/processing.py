from __future__ import annotations

import logging
import math
import shutil
import subprocess
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from scipy import signal

from .probe import SampleRateProbe, probe_sample_rate
from .utils import parse_center_frequency
from .visualize import save_stage_psd

LOG = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    in_path: Path
    target_freq: float = 0.0
    bandwidth: float = 12_500.0
    center_freq: Optional[float] = None
    fs_ch_target: float = 96_000.0
    deemph_us: float = 300.0
    squelch_dbfs: Optional[float] = None
    silence_trim: bool = False
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
            raise RuntimeError("ffmpeg is required to read SDR++ baseband recordings.")
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
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
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


class QuadratureDemod:
    def __init__(self):
        self.prev = np.complex64(1 + 0j)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        prevs = np.concatenate(([self.prev], samples[:-1]))
        prod = samples * np.conj(prevs)
        demod = np.angle(prod).astype(np.float32)
        self.prev = samples[-1]
        return demod


class DeemphasisFilter:
    def __init__(self, tau_us: float, sample_rate: float):
        tau_sec = max(tau_us * 1e-6, 1e-6)
        alpha = math.exp(-1.0 / (sample_rate * tau_sec))
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.state = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples
        out = np.empty_like(samples, dtype=np.float32)
        state = self.state
        alpha = self.alpha
        beta = self.beta
        for idx, sample in enumerate(samples):
            state = alpha * state + beta * sample
            out[idx] = state
        self.state = state
        return out


class SquelchGate:
    def __init__(
        self,
        sample_rate: float,
        threshold_dbfs: Optional[float],
        silence_trim: bool,
        hold_ms: float = 120.0,
    ):
        self.sample_rate = sample_rate
        self.manual_threshold = threshold_dbfs
        self.silence_trim = silence_trim
        self.hold_samples = int(sample_rate * hold_ms / 1000.0)
        self.hold_counter = 0
        self.noise_dbfs: Optional[float] = None
        self.open = False

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
        if samples.size == 0:
            return samples, self.manual_threshold or -120.0, -120.0, False

        rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)) + 1e-18)
        dbfs = 20.0 * math.log10(rms + 1e-12)

        if self.manual_threshold is None:
            if self.noise_dbfs is None:
                self.noise_dbfs = dbfs
            elif not self.open:
                self.noise_dbfs = 0.95 * self.noise_dbfs + 0.05 * dbfs
            threshold = (self.noise_dbfs or dbfs) + 6.0
        else:
            threshold = self.manual_threshold

        dropped = False
        if dbfs >= threshold:
            self.open = True
            self.hold_counter = self.hold_samples
        else:
            if self.hold_counter > 0:
                self.hold_counter = max(0, self.hold_counter - samples.size)
            else:
                self.open = False

        if self.open:
            audio = samples
        else:
            if self.silence_trim:
                audio = np.empty(0, dtype=samples.dtype)
                dropped = True
            else:
                audio = np.zeros_like(samples)

        return audio, threshold, dbfs, dropped


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
            raise RuntimeError("ffmpeg is required to write WAV output.")
        self.output_path = output_path
        self.input_rate = input_rate
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            f"{input_rate}",
            "-i",
            "-",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "48000",
            str(output_path),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
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
        self.proc.stdin.write(safe.tobytes())

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


class ProcessingPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def run(self) -> ProcessingResult:
        probe = probe_sample_rate(self.config.in_path)
        sample_rate = probe.value

        if self.config.target_freq <= 0 and not self.config.probe_only:
            raise ValueError("Target frequency must be positive. Provide --ft or use --interactive.")
        if self.config.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")

        center_freq = self.config.center_freq
        if center_freq is None:
            inferred = parse_center_frequency(self.config.in_path)
            if inferred is None:
                raise ValueError(
                    "Center frequency not supplied and could not be parsed from filename. "
                    "Use --fc to provide it explicitly."
                )
            center_freq = inferred

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

        taps = design_channel_filter(sample_rate, self.config.bandwidth, decimation)
        LOG.info("Designed FIR channel filter with %d taps.", len(taps))

        oscillator = ComplexOscillator(freq_offset, sample_rate)
        channel_filter = OverlapSaveFIR(taps, self.config.filter_block)
        decimator = Decimator(decimation)
        demod = QuadratureDemod()
        deemph = DeemphasisFilter(self.config.deemph_us, fs_channel)
        squelch = SquelchGate(
            sample_rate=fs_channel,
            threshold_dbfs=self.config.squelch_dbfs,
            silence_trim=self.config.silence_trim,
        )
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

            if self.config.plot_stages_path and not self.config.probe_only:
                stage_snapshots["input"] = (warmup.copy(), sample_rate)

            mix_sign = (
                self.config.mix_sign_override
                if self.config.mix_sign_override in (1, -1)
                else choose_mix_sign(warmup, sample_rate, freq_offset, taps, decimation)
            )
            LOG.info("Selected mixer sign %d based on warm-up snippet.", mix_sign)

            if self.config.probe_only:
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

            audio_writer = AudioWriter(output_path, fs_channel)
            try:
                for idx, block in enumerate(itertools.chain((warmup,), iterator)):
                    mixed = oscillator.mix(block, mix_sign)
                    if self.config.plot_stages_path and idx == 0 and not self.config.probe_only:
                        stage_snapshots["mixed"] = (mixed.copy(), sample_rate)
                    filtered = channel_filter.process(mixed)
                    if self.config.plot_stages_path and idx == 0 and not self.config.probe_only:
                        stage_snapshots["filtered"] = (filtered.copy(), sample_rate)
                    decimated = decimator.process(filtered)
                    if self.config.dump_iq_path:
                        iq_writer.write(decimated)
                    if self.config.plot_stages_path and idx == 0 and not self.config.probe_only:
                        stage_snapshots["decimated"] = (decimated.copy(), fs_channel)
                    baseband_power = np.mean(np.abs(decimated) ** 2) if decimated.size else 0.0
                    LOG.debug(
                        "Block: mixed=%d samples, decimated=%d, power=%.3e",
                        block.size,
                        decimated.size,
                        baseband_power,
                    )
                    audio = demod.process(decimated)
                    if self.config.plot_stages_path and idx == 0 and not self.config.probe_only:
                        stage_snapshots["demod"] = (audio.copy(), fs_channel)
                    audio = deemph.process(audio)
                    if self.config.plot_stages_path and idx == 0 and not self.config.probe_only:
                        stage_snapshots["deemph"] = (audio.copy(), fs_channel)
                    gated, threshold, dbfs, dropped = squelch.process(audio)
                    if self.config.plot_stages_path and idx == 0 and not self.config.probe_only:
                        stage_snapshots["squelch"] = (gated.copy(), fs_channel)
                    LOG.debug(
                        "Demod chunk: %d samples, rms=%.2f dBFS, gate %.2f dBFS, dropped=%s",
                        audio.size,
                        dbfs,
                        threshold,
                        dropped,
                    )
                    audio_writer.write(gated)
            finally:
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

    def _default_output_path(self) -> Path:
        ft = int(self.config.target_freq)
        return self.config.in_path.with_name(f"audio_{ft}_48k.wav")
