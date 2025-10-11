from __future__ import annotations

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import soundfile as sf

from .processing import ProcessingConfig, ProcessingPipeline

LOG = logging.getLogger(__name__)


def _generate_synthetic_iq(
    path: Path,
    sample_rate: float,
    seconds: float,
    freq_offset: float,
    *,
    amplitude: float = 0.7,
    noise_std: float = 0.02,
) -> None:
    total_samples = int(round(sample_rate * seconds))
    if total_samples <= 0:
        raise ValueError("Benchmark duration is too short to generate samples.")
    t = np.arange(total_samples, dtype=np.float64) / sample_rate
    tone = np.exp(1j * 2.0 * math.pi * freq_offset * t)
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=noise_std, size=(total_samples, 2))
    i = amplitude * tone.real + noise[:, 0]
    q = amplitude * tone.imag + noise[:, 1]
    iq = np.clip(np.column_stack((i, q)).astype(np.float32), -0.999, 0.999)
    sf.write(path, iq, int(sample_rate), format="WAV", subtype="PCM_16")


def run_benchmark(
    *,
    seconds: float,
    sample_rate: float,
    freq_offset: float,
    center_freq: Optional[float],
    target_freq: Optional[float],
    base_kwargs: Mapping[str, object] | None,
) -> int:
    if seconds <= 0:
        raise ValueError("Benchmark duration must be positive.")
    if sample_rate <= 0:
        raise ValueError("Benchmark sample rate must be positive.")
    half_band = sample_rate / 2.0
    if abs(freq_offset) >= half_band:
        raise ValueError("Benchmark offset must be within half the sample rate.")

    demod_value = (base_kwargs or {}).get("demod_mode")
    demod_mode = demod_value.lower() if isinstance(demod_value, str) else "nfm"

    if center_freq is not None and target_freq is not None:
        offset = target_freq - center_freq
    elif center_freq is not None:
        target_freq = center_freq + freq_offset
        offset = freq_offset
    elif target_freq is not None:
        center_freq = target_freq - freq_offset
        offset = freq_offset
    else:
        center_freq = 400_000_000.0
        target_freq = center_freq + freq_offset
        offset = freq_offset

    LOG.info(
        "Running benchmark: %.2f s at %.2f MS/s, demod=%s, offset %.1f kHz",
        seconds,
        sample_rate / 1e6,
        demod_mode.upper(),
        offset / 1e3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / f"benchmark_fc-{int(center_freq)}Hz.wav"
        _generate_synthetic_iq(
            input_path,
            sample_rate=sample_rate,
            seconds=seconds,
            freq_offset=offset,
        )

        kwargs: dict[str, Any] = dict(base_kwargs) if base_kwargs is not None else {}
        kwargs.update(
            {
                "target_freq": target_freq,
                "center_freq": center_freq,
                "center_freq_source": "benchmark",
                "demod_mode": demod_mode,
                "output_path": tmpdir_path / f"benchmark_audio_{demod_mode}.wav",
                "probe_only": False,
            }
        )

        config = ProcessingConfig(in_path=input_path, **kwargs)
        pipeline = ProcessingPipeline(config)

        start = time.perf_counter()
        result = pipeline.run(progress_sink=None)
        elapsed = time.perf_counter() - start

    iq_samples = sample_rate * seconds
    realtime = seconds / elapsed if elapsed > 0 else float("inf")
    peak_dbfs = 20.0 * math.log10(max(result.audio_peak, 1e-6))

    LOG.info(
        "Benchmark processed %.0f IQ samples in %.2f s (%.2f× realtime).",
        iq_samples,
        elapsed,
        realtime,
    )
    LOG.info(
        "Channel decimation %d -> %.1f Hz; audio peak %.2f dBFS.",
        result.decimation,
        result.fs_channel,
        peak_dbfs,
    )
    return 0


__all__ = ["run_benchmark"]
