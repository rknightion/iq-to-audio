from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

DATA_DIR = Path(__file__).resolve().parent.parent / "testfiles"
NFM_FIXTURE = DATA_DIR / "fc-456834049Hz-ft-455837500-ft2-456872500-NFM.wav"
AM_FIXTURE = DATA_DIR / "fc-132334577Hz-ft-132300000-AM.wav"


def _run_cli(args: list[str]) -> str:
    cmd = [sys.executable, "-m", "iq_to_audio.cli", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"CLI failed with {result.returncode}: {result.stderr}")
    output = (result.stdout or "") + (result.stderr or "")
    return output.replace("\r", "\n")


def _audio_stats(path: Path) -> dict[str, float]:
    audio, sample_rate = sf.read(path, dtype="float32")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    nonzero = float(np.count_nonzero(np.abs(audio) > 1e-4)) / float(audio.size or 1)
    return {
        "samples": float(audio.size),
        "sample_rate": float(sample_rate),
        "rms": rms,
        "peak": peak,
        "nonzero_frac": nonzero,
    }


def test_cli_preview_nfm_generates_audio(tmp_path):
    output = tmp_path / "nfm.wav"
    logs = _run_cli(
        [
            "--cli",
            "--in",
            str(NFM_FIXTURE),
            "--ft",
            "455837500",
            "--demod",
            "nfm",
            "--preview",
            "5",
            "--out",
            str(output),
        ],
    )
    preview_path = output.with_name(f"{output.stem}_preview{output.suffix}")
    assert preview_path.exists()

    stats = _audio_stats(preview_path)
    assert stats["sample_rate"] == 48_000
    assert stats["samples"] > 200_000
    assert 0.03 < stats["rms"] < 0.04
    assert stats["nonzero_frac"] > 0.05
    assert stats["peak"] > 0.25

    assert "Expecting approximately" in logs
    assert "Chunk 1: channelizing" in logs
    assert "Processing complete" in logs
    assert "Center frequency 456834049 Hz" in logs


def test_cli_preview_am_runs_without_squelch(tmp_path):
    output = tmp_path / "am.wav"
    logs = _run_cli(
        [
            "--cli",
            "--in",
            str(AM_FIXTURE),
            "--ft",
            "132300000",
            "--demod",
            "am",
            "--preview",
            "5",
            "--no-squelch",
            "--out",
            str(output),
        ],
    )
    preview_path = output.with_name(f"{output.stem}_preview{output.suffix}")
    assert preview_path.exists()

    stats = _audio_stats(preview_path)
    assert stats["sample_rate"] == 48_000
    assert stats["samples"] > 200_000
    assert 4e-5 < stats["rms"] < 8e-5
    assert stats["nonzero_frac"] > 0.02
    assert stats["peak"] < 0.01

    assert "Using AM demodulator." in logs
    assert "Processing complete" in logs
    assert "Center frequency 132334577 Hz" in logs


@pytest.mark.parametrize("mode", ["usb", "lsb"])
def test_cli_preview_ssb_without_squelch(tmp_path, mode: str):
    output = tmp_path / f"{mode}.wav"
    logs = _run_cli(
        [
            "--cli",
            "--in",
            str(NFM_FIXTURE),
            "--ft",
            "455837500",
            "--demod",
            mode,
            "--preview",
            "5",
            "--no-squelch",
            "--out",
            str(output),
        ]
    )
    preview_path = output.with_name(f"{output.stem}_preview{output.suffix}")
    assert preview_path.exists()

    stats = _audio_stats(preview_path)
    assert stats["sample_rate"] == 48_000
    assert stats["samples"] > 200_000
    assert stats["nonzero_frac"] > 0.9
    assert stats["peak"] >= 0.5

    assert f"Using {mode.upper()} demodulator." in logs
    assert "Processing complete" in logs


def test_cli_dump_iq_and_plot(tmp_path):
    output = tmp_path / "nfm_full.wav"
    iq_dump = tmp_path / "channel.cf32"
    plot_path = tmp_path / "stages.png"
    logs = _run_cli(
        [
            "--cli",
            "--in",
            str(NFM_FIXTURE),
            "--ft",
            "455837500",
            "--demod",
            "nfm",
            "--preview",
            "3",
            "--no-squelch",
            "--out",
            str(output),
            "--dump-iq",
            str(iq_dump),
            "--plot-stages",
            str(plot_path),
            "--fft-workers",
            "2",
        ]
    )
    preview_path = output.with_name(f"{output.stem}_preview{output.suffix}")
    assert preview_path.exists()
    assert iq_dump.exists() and iq_dump.stat().st_size > 0
    assert plot_path.exists() and plot_path.stat().st_size > 0

    stats = _audio_stats(preview_path)
    assert stats["rms"] > 0.01

    assert ("writing IQ dump" in logs) or ("Dump IQ" in logs)
    assert "Saved stage PSD" in logs


def test_cli_benchmark_runs():
    logs = _run_cli(
        [
            "--benchmark",
            "--benchmark-seconds",
            "0.2",
            "--benchmark-sample-rate",
            "200000",
            "--benchmark-offset",
            "5000",
            "--demod",
            "am",
        ]
    )
    assert "Running benchmark" in logs
    assert "demod=AM" in logs
    assert "Benchmark processed" in logs
