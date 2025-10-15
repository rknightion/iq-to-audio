#!/usr/bin/env python3
"""End-to-end test harness that exercises the CLI, DSP pipeline, and benchmarks.

The script runs pytest, demodulates bundled fixture captures, generates
visual diagnostics (PSD, waveform, waterfall), executes synthetic benchmarks,
and emits a rich HTML report under testreports/.
"""
from __future__ import annotations

import contextlib
import io
import logging
import math
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Force headless backend for CI environments.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402
from scipy.signal.windows import hann  # noqa: E402

from iq_to_audio.benchmark import run_benchmark  # noqa: E402
from iq_to_audio.preview import run_preview  # noqa: E402
from iq_to_audio.processing import ProcessingConfig, ProcessingPipeline  # noqa: E402
from iq_to_audio.progress import NullProgressSink  # noqa: E402
from iq_to_audio.utils import CenterFrequencyResult, detect_center_frequency  # noqa: E402

REPORT_DIR = Path("testreports")
ASSETS_DIR = REPORT_DIR / "assets"
ARTIFACTS_DIR = REPORT_DIR / "artifacts"
LOGS_DIR = REPORT_DIR / "logs"

TESTFILES_DIR = Path("testfiles")

LOG = logging.getLogger("run_comprehensive_tests")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class PytestSummary:
    passed: int
    failed: int
    skipped: int
    errors: int
    total: int
    time: float
    junit_path: Path
    log_path: Path


@dataclass
class AudioStats:
    samples: int
    sample_rate: float
    duration: float
    rms: float
    peak: float
    nonzero_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "rms": self.rms,
            "peak": self.peak,
            "nonzero_fraction": self.nonzero_fraction,
        }


@dataclass
class PipelineRunReport:
    name: str
    input_path: Path
    demod: str
    target_freq: float
    center_freq: float
    center_source: str
    output_audio: Path
    preview_audio: Path
    dump_iq_path: Path
    plot_stages_path: Path
    waveform_plot: Path
    waterfall_plot: Path
    histogram_plot: Path
    preview_waveform_plot: Path
    preview_waterfall_plot: Path
    audio_stats: AudioStats
    preview_stats: AudioStats
    processing_time: float
    preview_time: float
    log_path: Path
    preview_log_path: Path
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    demod: str
    seconds: float
    sample_rate: float
    offset: float
    exit_code: int
    elapsed: float
    realtime: float | None
    log_path: Path


def clear_report_directory() -> None:
    if REPORT_DIR.exists():
        LOG.info("Pruning existing report directory %s", REPORT_DIR)
        shutil.rmtree(REPORT_DIR)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run_pytest_suite() -> PytestSummary:
    junit_path = REPORT_DIR / "pytest-results.xml"
    log_path = LOGS_DIR / "pytest.log"
    LOG.info("Running pytest suite…")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        exit_code = pytest.main(
            [
                "-q",
                f"--junitxml={junit_path}",
                "--maxfail=1",
            ]
        )
    elapsed = time.perf_counter() - start
    log_text = stdout_buffer.getvalue() + "\n" + stderr_buffer.getvalue()
    log_path.write_text(log_text)
    if exit_code != 0:
        LOG.warning("Pytest exited with status %s", exit_code)

    tree = ET.parse(junit_path)
    root = tree.getroot()
    stats = {
        "tests": int(root.attrib.get("tests", 0)),
        "errors": int(root.attrib.get("errors", 0)),
        "failures": int(root.attrib.get("failures", 0)),
        "skipped": int(root.attrib.get("skipped", 0)),
        "time": float(root.attrib.get("time", elapsed)),
    }
    passed = stats["tests"] - stats["errors"] - stats["failures"] - stats["skipped"]
    return PytestSummary(
        passed=passed,
        failed=stats["failures"],
        skipped=stats["skipped"],
        errors=stats["errors"],
        total=stats["tests"],
        time=stats["time"],
        junit_path=junit_path,
        log_path=log_path,
    )


def compute_audio_stats(audio_path: Path) -> AudioStats:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    samples = int(audio.size)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio))) if samples else 0.0
    nonzero = float(np.count_nonzero(np.abs(audio) > 1e-4)) / float(samples or 1)
    duration = samples / float(sample_rate or 1.0)
    return AudioStats(
        samples=samples,
        sample_rate=float(sample_rate),
        duration=duration,
        rms=rms,
        peak=peak,
        nonzero_fraction=nonzero,
    )


def _configure_pipeline_logger(stream: io.StringIO) -> logging.Handler:
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logging.getLogger("iq_to_audio").addHandler(handler)
    return handler


def _teardown_pipeline_logger(handler: logging.Handler) -> None:
    logger = logging.getLogger("iq_to_audio")
    logger.removeHandler(handler)


def create_waveform_plot(audio: np.ndarray, sample_rate: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    total_samples = audio.size
    max_samples = min(total_samples, int(sample_rate * 5))
    segment = audio[:max_samples]
    times = np.arange(segment.size) / sample_rate
    ax.plot(times, segment, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, ls=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_histogram_plot(audio: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(audio, bins=120, color="#1f77b4", alpha=0.9, density=True)
    ax.set_title(title)
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")
    ax.grid(True, ls=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_waterfall_plot(
    audio: np.ndarray,
    sample_rate: float,
    path: Path,
    title: str,
) -> None:
    window = hann(2048, sym=False)
    freqs, times, sxx = spectrogram(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=2048,
        noverlap=1536,
        scaling="density",
        mode="magnitude",
    )
    sxx_db = 20.0 * np.log10(sxx + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.pcolormesh(times, freqs / 1e3, sxx_db, shading="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dBFS)")
    ax.set_ylim(0, np.min([sample_rate / 2000.0, 24]))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_cf32(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        return np.empty(0, dtype=np.complex64)
    return data.view(np.complex64)


def create_iq_waterfall_plot(
    iq_dump: Path,
    sample_rate: float,
    path: Path,
    title: str,
) -> None:
    iq = load_cf32(iq_dump)
    if iq.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No IQ data", ha="center", va="center")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    magnitude = np.abs(iq)
    create_waterfall_plot(magnitude, sample_rate, path, title)


def determine_center_frequency(path: Path) -> CenterFrequencyResult:
    detection = detect_center_frequency(path)
    if detection.value is None:
        raise RuntimeError(f"Unable to determine center frequency for {path}")
    return detection


def process_sample(
    *,
    name: str,
    input_path: Path,
    target_freq: float,
    demod: str,
    bandwidth: float,
    preview_seconds: float,
) -> PipelineRunReport:
    LOG.info("Processing %s (%s)", name, input_path.name)
    detection = determine_center_frequency(input_path)
    output_audio = ARTIFACTS_DIR / f"{name}_{demod}_audio.wav"
    preview_audio = ARTIFACTS_DIR / f"{name}_{demod}_audio_preview.wav"
    dump_iq_path = ARTIFACTS_DIR / f"{name}_{demod}_channel.cf32"
    plot_stages_path = ASSETS_DIR / f"{name}_{demod}_stages.png"

    config = ProcessingConfig(
        in_path=input_path,
        target_freq=target_freq,
        bandwidth=bandwidth,
        center_freq=detection.value,
        center_freq_source=detection.source,
        demod_mode=demod,
        fs_ch_target=96_000.0,
        deemph_us=300.0,
        agc_enabled=True,
        output_path=output_audio,
        dump_iq_path=dump_iq_path,
        chunk_size=262_144,
        filter_block=65_536,
        iq_order="iq",
        probe_only=False,
        mix_sign_override=None,
        plot_stages_path=plot_stages_path,
        fft_workers=4,
    )

    log_stream = io.StringIO()
    handler = _configure_pipeline_logger(log_stream)
    pipeline = ProcessingPipeline(config)
    start = time.perf_counter()
    result = pipeline.run(progress_sink=NullProgressSink())
    processing_time = time.perf_counter() - start
    _teardown_pipeline_logger(handler)
    log_path = LOGS_DIR / f"{name}_{demod}_pipeline.log"
    log_path.write_text(log_stream.getvalue())

    audio_stats = compute_audio_stats(output_audio)

    # Visualisations for full audio
    audio_full, sr_full = sf.read(output_audio, dtype="float32")
    waveform_plot = ASSETS_DIR / f"{name}_{demod}_waveform.png"
    histogram_plot = ASSETS_DIR / f"{name}_{demod}_histogram.png"
    waterfall_plot = ASSETS_DIR / f"{name}_{demod}_waterfall.png"
    create_waveform_plot(audio_full, sr_full, waveform_plot, f"{name} ({demod.upper()}) waveform")
    create_histogram_plot(audio_full, histogram_plot, f"{name} ({demod.upper()}) amplitude PDF")
    create_waterfall_plot(audio_full, sr_full, waterfall_plot, f"{name} ({demod.upper()}) waterfall")

    # Run preview for comparative stats
    preview_log_stream = io.StringIO()
    preview_handler = _configure_pipeline_logger(preview_log_stream)
    preview_start = time.perf_counter()
    _, preview_path = run_preview(
        config,
        preview_seconds,
        progress_sink=NullProgressSink(),
    )
    preview_time = time.perf_counter() - preview_start
    _teardown_pipeline_logger(preview_handler)
    preview_log_path = LOGS_DIR / f"{name}_{demod}_preview.log"
    preview_log_path.write_text(preview_log_stream.getvalue())

    if preview_path != preview_audio:
        # move into expected location for consistency
        shutil.move(preview_path, preview_audio)
    preview_stats = compute_audio_stats(preview_audio)
    preview_data, preview_sr = sf.read(preview_audio, dtype="float32")
    preview_waveform_plot = ASSETS_DIR / f"{name}_{demod}_preview_waveform.png"
    preview_waterfall_plot = ASSETS_DIR / f"{name}_{demod}_preview_waterfall.png"
    create_waveform_plot(
        preview_data,
        preview_sr,
        preview_waveform_plot,
        f"{name} preview ({demod.upper()}) waveform",
    )
    create_waterfall_plot(
        preview_data,
        preview_sr,
        preview_waterfall_plot,
        f"{name} preview ({demod.upper()}) waterfall",
    )

    # IQ waterfall derived from dump
    iq_waterfall_path = ASSETS_DIR / f"{name}_{demod}_iq_waterfall.png"
    create_iq_waterfall_plot(
        dump_iq_path,
        sample_rate=result.fs_channel,
        path=iq_waterfall_path,
        title=f"{name} channelized IQ magnitude ({demod.upper()})",
    )

    extras = {
        "fs_channel": result.fs_channel,
        "decimation": result.decimation,
        "mix_sign": result.mix_sign,
        "audio_peak_dbfs": 20.0 * math.log10(max(result.audio_peak, 1e-6)),
        "iq_waterfall_path": iq_waterfall_path,
    }

    return PipelineRunReport(
        name=name,
        input_path=input_path,
        demod=demod,
        target_freq=target_freq,
        center_freq=detection.value,
        center_source=detection.source,
        output_audio=output_audio,
        preview_audio=preview_audio,
        dump_iq_path=dump_iq_path,
        plot_stages_path=plot_stages_path,
        waveform_plot=waveform_plot,
        waterfall_plot=waterfall_plot,
        histogram_plot=histogram_plot,
        preview_waveform_plot=preview_waveform_plot,
        preview_waterfall_plot=preview_waterfall_plot,
        audio_stats=audio_stats,
        preview_stats=preview_stats,
        processing_time=processing_time,
        preview_time=preview_time,
        log_path=log_path,
        preview_log_path=preview_log_path,
        extra=extras,
    )


def run_benchmarks() -> list[BenchmarkReport]:
    LOG.info("Executing synthetic benchmarks…")
    specs = [
        {"demod": "nfm", "seconds": 1.0, "sample_rate": 2_500_000.0, "offset": 25_000.0},
        {"demod": "am", "seconds": 1.0, "sample_rate": 1_000_000.0, "offset": 10_000.0},
        {"demod": "usb", "seconds": 0.8, "sample_rate": 3_000_000.0, "offset": 12_500.0},
    ]
    reports: list[BenchmarkReport] = []
    for idx, spec in enumerate(specs, start=1):
        base_kwargs = {
            "target_freq": 0.0,
            "bandwidth": 12_500.0,
            "center_freq": None,
            "center_freq_source": None,
            "demod_mode": spec["demod"],
            "fs_ch_target": 96_000.0,
            "deemph_us": 300.0,
            "agc_enabled": True,
            "output_path": None,
            "dump_iq_path": None,
            "chunk_size": 131_072,
            "filter_block": 65_536,
            "iq_order": "iq",
            "probe_only": False,
            "mix_sign_override": None,
            "plot_stages_path": None,
            "fft_workers": 4,
        }
        log_stream = io.StringIO()
        handler = _configure_pipeline_logger(log_stream)
        start = time.perf_counter()
        exit_code = run_benchmark(
            seconds=spec["seconds"],
            sample_rate=spec["sample_rate"],
            freq_offset=spec["offset"],
            center_freq=None,
            target_freq=None,
            base_kwargs=base_kwargs,
        )
        elapsed = time.perf_counter() - start
        _teardown_pipeline_logger(handler)
        log_path = LOGS_DIR / f"benchmark_{idx}_{spec['demod']}.log"
        log_text = log_stream.getvalue()
        log_path.write_text(log_text)
        realtime = None
        match = re.search(r"\(([\d.]+)× realtime\)", log_text)
        if match:
            realtime = float(match.group(1))
        reports.append(
            BenchmarkReport(
                demod=spec["demod"],
                seconds=spec["seconds"],
                sample_rate=spec["sample_rate"],
                offset=spec["offset"],
                exit_code=exit_code,
                elapsed=elapsed,
                realtime=realtime,
                log_path=log_path,
            )
        )
    return reports


def format_dbfs(value: float) -> str:
    if value <= 0:
        return "-inf"
    return f"{20.0 * math.log10(value):.2f}"


def build_html_report(
    pytest_summary: PytestSummary,
    pipeline_reports: Sequence[PipelineRunReport],
    benchmark_reports: Sequence[BenchmarkReport],
) -> str:
    def format_audio_stats(stats: AudioStats) -> str:
        return (
            f"{stats.duration:.2f}s, RMS={stats.rms:.4f} ({format_dbfs(stats.rms)} dBFS), "
            f"Peak={stats.peak:.4f} ({format_dbfs(stats.peak)} dBFS), "
            f"Non-zero={stats.nonzero_fraction*100:.1f}%"
        )

    pipeline_rows = []
    for report in pipeline_reports:
        iq_link = report.extra.get("iq_waterfall_path")
        pipeline_rows.append(
            f"""
            <tr>
              <td>{report.name}</td>
              <td>{report.demod.upper()}</td>
              <td>{report.center_freq:.0f} Hz<br><small>{report.center_source}</small></td>
              <td>{report.target_freq:.0f} Hz</td>
              <td>{report.processing_time:.2f}s</td>
              <td>{format_audio_stats(report.audio_stats)}</td>
              <td>{format_audio_stats(report.preview_stats)}</td>
              <td>
                <a href="{report.output_audio.relative_to(REPORT_DIR)}">audio</a> |
                <a href="{report.preview_audio.relative_to(REPORT_DIR)}">preview</a><br>
                <a href="{report.log_path.relative_to(REPORT_DIR)}">log</a> |
                <a href="{report.preview_log_path.relative_to(REPORT_DIR)}">preview log</a>
              </td>
              <td>
                <a href="{report.plot_stages_path.relative_to(REPORT_DIR)}">stage PSD</a><br>
                <a href="{report.waveform_plot.relative_to(REPORT_DIR)}">waveform</a><br>
                <a href="{report.waterfall_plot.relative_to(REPORT_DIR)}">waterfall</a><br>
                <a href="{report.histogram_plot.relative_to(REPORT_DIR)}">histogram</a><br>
                <a href="{report.preview_waveform_plot.relative_to(REPORT_DIR)}">preview waveform</a><br>
                <a href="{report.preview_waterfall_plot.relative_to(REPORT_DIR)}">preview waterfall</a><br>
                <a href="{iq_link.relative_to(REPORT_DIR)}">IQ waterfall</a>
              </td>
            </tr>
            """
        )

    benchmark_rows = []
    for bench in benchmark_reports:
        realtime_str = f"{bench.realtime:.2f}×" if bench.realtime is not None else "n/a"
        benchmark_rows.append(
            f"""
            <tr>
              <td>{bench.demod.upper()}</td>
              <td>{bench.seconds:.2f}s</td>
              <td>{bench.sample_rate/1e6:.2f} MS/s</td>
              <td>{bench.offset/1e3:.1f} kHz</td>
              <td>{bench.elapsed:.2f}s</td>
              <td>{realtime_str}</td>
              <td>{'pass' if bench.exit_code == 0 else 'fail'} |
                  <a href="{bench.log_path.relative_to(REPORT_DIR)}">log</a></td>
            </tr>
            """
        )

    summary_color = "#c8f7c5" if pytest_summary.failed == 0 and pytest_summary.errors == 0 else "#f7c5c5"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IQ to Audio – Comprehensive Test Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      color: #222;
      background-color: #fafafa;
    }}
    header {{
      background-color: #1f3c88;
      color: white;
      padding: 20px 40px;
    }}
    header h1 {{
      margin: 0 0 10px 0;
      font-size: 28px;
    }}
    header p {{
      margin: 0;
      font-size: 16px;
    }}
    main {{
      padding: 30px 40px 60px 40px;
    }}
    section {{
      margin-bottom: 40px;
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.08);
      padding: 24px;
    }}
    h2 {{
      margin-top: 0;
      color: #1f3c88;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background-color: #f0f3ff;
    }}
    .summary {{
      background-color: {summary_color};
      padding: 16px;
      border-radius: 6px;
      font-size: 16px;
      line-height: 1.6;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }}
    figure {{
      margin: 0;
      background: #f7f7f7;
      border-radius: 6px;
      padding: 10px;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05);
    }}
    figure img {{
      width: 100%;
      height: auto;
      border-radius: 4px;
    }}
    figure figcaption {{
      margin-top: 8px;
      font-size: 13px;
      color: #444;
    }}
    details {{
      margin-top: 12px;
      background: #f7f7f7;
      border-radius: 6px;
      padding: 10px 14px;
    }}
    pre {{
      background: #1c1c1c;
      color: #eaeaea;
      padding: 12px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 12px;
    }}
    footer {{
      text-align: center;
      color: #666;
      font-size: 12px;
      margin-top: 40px;
    }}
    a {{
      color: #1f3c88;
    }}
  </style>
</head>
<body>
  <header>
    <h1>IQ to Audio – Comprehensive Test Report</h1>
    <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
  </header>
  <main>
    <section>
      <h2>Test Summary</h2>
      <div class="summary">
        <strong>Total tests:</strong> {pytest_summary.total} &nbsp;
        <strong>Passed:</strong> {pytest_summary.passed} &nbsp;
        <strong>Failed:</strong> {pytest_summary.failed} &nbsp;
        <strong>Errors:</strong> {pytest_summary.errors} &nbsp;
        <strong>Skipped:</strong> {pytest_summary.skipped} &nbsp;
        <strong>Runtime:</strong> {pytest_summary.time:.2f}s
        <br>
        JUnit: <a href="{pytest_summary.junit_path.relative_to(REPORT_DIR)}">{pytest_summary.junit_path.name}</a>,
        Logs: <a href="{pytest_summary.log_path.relative_to(REPORT_DIR)}">{pytest_summary.log_path.name}</a>
      </div>
    </section>

    <section>
      <h2>Fixture Demodulation Runs</h2>
      <p>Each bundled sample was processed end-to-end, generating 48 kHz audio, PSD snapshots, IQ waterfalls, and preview snippets.</p>
      <table>
        <thead>
          <tr>
            <th>Sample</th>
            <th>Demod</th>
            <th>Center<br>frequency</th>
            <th>Target</th>
            <th>Runtime</th>
            <th>Full audio stats</th>
            <th>Preview stats</th>
            <th>Artifacts</th>
            <th>Visuals</th>
          </tr>
        </thead>
        <tbody>
          {''.join(pipeline_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Selected Visual Diagnostics</h2>
      <div class="gallery">
"""

    # Collect a subset of figures for inline display (limit to avoid huge HTML)
    figures_to_embed = []
    for report in pipeline_reports:
        figures_to_embed.extend(
            [
                (report.waveform_plot, f"{report.name} ({report.demod.upper()}) waveform"),
                (report.waterfall_plot, f"{report.name} ({report.demod.upper()}) audio waterfall"),
                (report.preview_waterfall_plot, f"{report.name} preview waterfall"),
                (report.plot_stages_path, f"{report.name} stage PSD overview"),
            ]
        )
    for figure_path, caption in figures_to_embed:
        figures_html = f"""
        <figure>
          <img src="{figure_path.relative_to(REPORT_DIR)}" alt="{caption}">
          <figcaption>{caption}</figcaption>
        </figure>
        """
        html += figures_html

    html += """
      </div>
    </section>

    <section>
      <h2>Benchmark Results</h2>
      <p>Synthetic captures were generated to gauge throughput under different demodulators and sample rates.</p>
      <table>
        <thead>
          <tr>
            <th>Demod</th>
            <th>Duration</th>
            <th>Sample rate</th>
            <th>Offset</th>
            <th>Elapsed</th>
            <th>Realtime factor</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(benchmark_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Environment</h2>
      <p>Python {sys.version.split()[0]} | NumPy {np.__version__} | SciPy {SCIPY_VERSION} | Matplotlib {matplotlib.__version__}</p>
    </section>
  </main>
  <footer>
    IQ to Audio – comprehensive validation generated by run_comprehensive_tests.py
  </footer>
</body>
</html>
"""
    return html


def main() -> None:
    clear_report_directory()
    pytest_summary = run_pytest_suite()

    samples = [
        {
            "name": "nfm_handset",
            "input_path": TESTFILES_DIR / "fc-456834049Hz-ft-455837500-ft2-456872500-NFM.wav",
            "target_freq": 455_837_500.0,
            "demod": "nfm",
            "bandwidth": 12_500.0,
            "preview_seconds": 5.0,
        },
        {
            "name": "am_airband",
            "input_path": TESTFILES_DIR / "fc-132334577Hz-ft-132300000-AM.wav",
            "target_freq": 132_300_000.0,
            "demod": "am",
            "bandwidth": 12_500.0,
            "preview_seconds": 5.0,
        },
    ]

    pipeline_reports: list[PipelineRunReport] = []
    for spec in samples:
        pipeline_reports.append(process_sample(**spec))

    benchmark_reports = run_benchmarks()

    html = build_html_report(pytest_summary, pipeline_reports, benchmark_reports)
    html_path = REPORT_DIR / "index.html"
    html_path.write_text(html, encoding="utf-8")
    LOG.info("Report written to %s", html_path.resolve())

    if pytest_summary.failed > 0 or pytest_summary.errors > 0:
        LOG.error("Test failures detected; exiting with status 1.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - guard for CI visibility
        LOG.exception("Comprehensive test run failed: %s", exc)
        sys.exit(1)
