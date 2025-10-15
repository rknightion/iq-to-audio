"""Performance benchmarks for iq-to-audio processing pipeline.

These benchmarks track performance over time for:
- Different demodulation modes (NFM, AM, USB, LSB)
- Various sample rates and chunk sizes
- Real vs synthetic signals
- AGC and filtering configurations

Run with: uv run pytest tests/test_benchmark.py --benchmark-json=results.json
"""

import logging
import tempfile
from pathlib import Path

import pytest

from iq_to_audio.benchmark import run_benchmark
from iq_to_audio.processing import ProcessingConfig, ProcessingPipeline

# ============================================================================
# Base Configuration Helpers
# ============================================================================


def _base_config(demod_mode: str = "nfm", **overrides) -> dict:
    """Create base configuration for benchmarks."""
    config = {
        "target_freq": 0.0,
        "bandwidth": 12_500.0,
        "center_freq": None,
        "center_freq_source": None,
        "demod_mode": demod_mode,
        "fs_ch_target": 96_000.0,
        "deemph_us": 300.0 if demod_mode == "nfm" else 0.0,
        "agc_enabled": True,
        "output_path": None,
        "dump_iq_path": None,
        "chunk_size": 131_072,
        "filter_block": 65_536,
        "iq_order": "iq",
        "probe_only": False,
        "mix_sign_override": None,
        "plot_stages_path": None,
        "fft_workers": 2,
    }
    config.update(overrides)
    return config


# ============================================================================
# Smoke Test (Not a Benchmark)
# ============================================================================


def test_benchmark_smoke(caplog):
    """Smoke test to verify benchmark module works."""
    with caplog.at_level(logging.INFO, logger="iq_to_audio.benchmark"):
        code = run_benchmark(
            seconds=0.2,
            sample_rate=200_000.0,
            freq_offset=5_000.0,
            center_freq=400_000_000.0,
            target_freq=400_005_000.0,
            base_kwargs=_base_config("nfm"),
        )
    assert code == 0
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Running benchmark" in messages
    assert "Benchmark processed" in messages


# ============================================================================
# Demodulation Mode Benchmarks (Synthetic Data)
# ============================================================================


@pytest.mark.benchmark(group="demod-synthetic")
def test_benchmark_nfm_synthetic(benchmark):
    """Benchmark NFM demodulation with synthetic signal."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=250_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", bandwidth=12_500.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="demod-synthetic")
def test_benchmark_am_synthetic(benchmark):
    """Benchmark AM demodulation with synthetic signal."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=250_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("am", bandwidth=10_000.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="demod-synthetic")
def test_benchmark_usb_synthetic(benchmark):
    """Benchmark USB (upper sideband) demodulation with synthetic signal."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=250_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("usb", bandwidth=2_800.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="demod-synthetic")
def test_benchmark_lsb_synthetic(benchmark):
    """Benchmark LSB (lower sideband) demodulation with synthetic signal."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=250_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("lsb", bandwidth=2_800.0),
    )
    assert result == 0


# ============================================================================
# Sample Rate Scaling Benchmarks
# ============================================================================


@pytest.mark.benchmark(group="sample-rate-scaling")
def test_benchmark_low_sample_rate(benchmark):
    """Benchmark at low sample rate (96 kHz) - typical for audio."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=96_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", bandwidth=12_500.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="sample-rate-scaling")
def test_benchmark_medium_sample_rate(benchmark):
    """Benchmark at medium sample rate (1 MHz) - typical for SDR."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", bandwidth=12_500.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="sample-rate-scaling")
def test_benchmark_high_sample_rate(benchmark):
    """Benchmark at high sample rate (2.5 MHz) - stress test."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=2_500_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", bandwidth=12_500.0),
    )
    assert result == 0


# ============================================================================
# Chunk Size Benchmarks (Memory vs Throughput Tradeoff)
# ============================================================================


@pytest.mark.benchmark(group="chunk-size")
def test_benchmark_small_chunks(benchmark):
    """Benchmark with small chunks (32k) - lower memory, more overhead."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", chunk_size=32_768, filter_block=16_384),
    )
    assert result == 0


@pytest.mark.benchmark(group="chunk-size")
def test_benchmark_medium_chunks(benchmark):
    """Benchmark with medium chunks (128k) - balanced."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", chunk_size=131_072, filter_block=65_536),
    )
    assert result == 0


@pytest.mark.benchmark(group="chunk-size")
def test_benchmark_large_chunks(benchmark):
    """Benchmark with large chunks (512k) - higher memory, less overhead."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", chunk_size=524_288, filter_block=262_144),
    )
    assert result == 0


# ============================================================================
# AGC Performance Benchmarks
# ============================================================================


@pytest.mark.benchmark(group="agc-impact")
def test_benchmark_nfm_with_agc(benchmark):
    """Benchmark NFM with AGC enabled."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", agc_enabled=True),
    )
    assert result == 0


@pytest.mark.benchmark(group="agc-impact")
def test_benchmark_nfm_without_agc(benchmark):
    """Benchmark NFM with AGC disabled."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", agc_enabled=False),
    )
    assert result == 0


# ============================================================================
# Bandwidth Benchmarks
# ============================================================================


@pytest.mark.benchmark(group="bandwidth-scaling")
def test_benchmark_narrow_bandwidth(benchmark):
    """Benchmark with narrow bandwidth (2.8 kHz) - voice."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("usb", bandwidth=2_800.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="bandwidth-scaling")
def test_benchmark_medium_bandwidth(benchmark):
    """Benchmark with medium bandwidth (12.5 kHz) - NFM."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=1_000_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", bandwidth=12_500.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="bandwidth-scaling")
def test_benchmark_wide_bandwidth(benchmark):
    """Benchmark with wide bandwidth (200 kHz) - wideband FM."""
    result = benchmark(
        run_benchmark,
        seconds=0.5,
        sample_rate=2_500_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", bandwidth=200_000.0, deemph_us=75.0),
    )
    assert result == 0


# ============================================================================
# Real Test File Benchmarks
# ============================================================================


@pytest.mark.benchmark(group="real-files")
def test_benchmark_real_am_file(benchmark, am_test_file):
    """Benchmark processing real AM test file."""

    def run_real_am():
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProcessingConfig(
                in_path=am_test_file,
                output_path=Path(tmpdir) / "output.wav",
                target_freq=132_300_000.0,
                bandwidth=10_000.0,
                demod_mode="am",
                fs_ch_target=48_000.0,
                agc_enabled=True,
                chunk_size=131_072,
                filter_block=65_536,
                iq_order="iq",
            )
            pipeline = ProcessingPipeline(config)
            result = pipeline.run(progress_sink=None)
            return result.decimation

    result = benchmark(run_real_am)
    assert result > 0


@pytest.mark.benchmark(group="real-files")
def test_benchmark_real_nfm_file(benchmark, nfm_test_file):
    """Benchmark processing real NFM test file."""

    def run_real_nfm():
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProcessingConfig(
                in_path=nfm_test_file,
                output_path=Path(tmpdir) / "output.wav",
                target_freq=456_872_500.0,
                bandwidth=12_500.0,
                demod_mode="nfm",
                fs_ch_target=48_000.0,
                deemph_us=300.0,
                agc_enabled=True,
                chunk_size=131_072,
                filter_block=65_536,
                iq_order="iq",
            )
            pipeline = ProcessingPipeline(config)
            result = pipeline.run(progress_sink=None)
            return result.decimation

    result = benchmark(run_real_nfm)
    assert result > 0


# ============================================================================
# Long Duration Benchmarks (Sustained Performance)
# ============================================================================


@pytest.mark.benchmark(group="sustained-performance")
def test_benchmark_sustained_nfm_1_second(benchmark):
    """Benchmark sustained NFM processing for 1 second of signal."""
    result = benchmark(
        run_benchmark,
        seconds=1.0,
        sample_rate=2_500_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("nfm", bandwidth=12_500.0),
    )
    assert result == 0


@pytest.mark.benchmark(group="sustained-performance")
def test_benchmark_sustained_am_1_second(benchmark):
    """Benchmark sustained AM processing for 1 second of signal."""
    result = benchmark(
        run_benchmark,
        seconds=1.0,
        sample_rate=2_500_000.0,
        freq_offset=5_000.0,
        center_freq=400_000_000.0,
        target_freq=400_005_000.0,
        base_kwargs=_base_config("am", bandwidth=10_000.0),
    )
    assert result == 0
