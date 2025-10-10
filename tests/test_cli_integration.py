from pathlib import Path
import subprocess
import sys

import numpy as np
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

    audio, sample_rate = sf.read(preview_path, dtype="float32")
    assert sample_rate == 48_000
    assert audio.size > 0
    assert np.count_nonzero(np.abs(audio) > 1e-4) > 0

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

    audio, sample_rate = sf.read(preview_path, dtype="float32")
    assert sample_rate == 48_000
    assert audio.size > 0
    assert np.count_nonzero(np.abs(audio) > 1e-4) > 0

    assert "Using AM demodulator." in logs
    assert "Processing complete" in logs
    assert "Center frequency 132334577 Hz" in logs


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
