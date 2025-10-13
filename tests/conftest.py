# ruff: noqa: NPY002
"""
Shared pytest fixtures and configuration for iq-to-audio tests.

Provides synthetic IQ generation, Qt app fixtures, and DSP test utilities.
"""

import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from hypothesis import strategies as st

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_TESTFILES_DIR = Path(__file__).parent.parent / "testfiles"
_TESTFILES_ARCHIVE = _TESTFILES_DIR / "iq-to-audio-fixtures.tar.xz"

LOG = logging.getLogger(__name__)


def _download_fixtures_if_needed() -> bool:
    """
    Download test fixtures from Google Drive if not present.

    Uses the download_test_fixtures.py script with environment variables.
    Returns True if fixtures are available, False otherwise.
    """
    if _TESTFILES_ARCHIVE.exists():
        return True

    # Check if we have the necessary environment variables
    if not os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON") or not os.getenv("GDRIVE_FILE_ID"):
        LOG.warning(
            "Test fixtures not found and Google Drive credentials not configured. "
            "Tests requiring fixtures will be skipped."
        )
        return False

    # Try to run the download script
    script_path = Path(__file__).parent.parent / "scripts" / "download_test_fixtures.py"
    if not script_path.exists():
        LOG.warning("Download script not found at %s", script_path)
        return False

    try:
        LOG.info("Attempting to download test fixtures from Google Drive...")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        LOG.info("Download script output: %s", result.stdout)
        return _TESTFILES_ARCHIVE.exists()
    except subprocess.CalledProcessError as e:
        LOG.error("Failed to download test fixtures: %s", e.stderr)
        return False


@pytest.fixture(scope="session", autouse=True)
def download_test_fixtures():
    """
    Session-level fixture to download test fixtures before any tests run.

    This runs once per test session and attempts to download the fixtures
    from Google Drive if they're not already present.
    """
    _download_fixtures_if_needed()


def _extract_fixture(filename: str) -> Path:
    destination = _TESTFILES_DIR / filename
    if destination.exists() or not _TESTFILES_ARCHIVE.exists():
        return destination

    with tarfile.open(_TESTFILES_ARCHIVE, mode="r:xz") as archive:
        try:
            member = archive.getmember(filename)
        except KeyError:
            pytest.skip(f"{filename} missing from archive {_TESTFILES_ARCHIVE}")

        target_path = destination.resolve()
        base_path = _TESTFILES_DIR.resolve()
        member_path = (base_path / member.name).resolve()
        if not str(member_path).startswith(str(base_path)):
            raise ValueError(f"Unsafe path detected in archive: {member.name}")

        archive.extract(member, path=_TESTFILES_DIR, filter="data")

        if not target_path.exists():
            raise RuntimeError(f"Failed to extract {filename} from archive")

    return destination


def _ensure_fixture(filename: str) -> Path:
    _TESTFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _TESTFILES_DIR / filename
    if path.exists():
        return path
    _extract_fixture(filename)
    if path.exists():
        return path
    pytest.skip(
        f"Test fixture not available: {filename}. "
        "Ensure iq-to-audio-fixtures.tar.xz is present in testfiles/."
    )
    return path


# ============================================================================
# Qt Application Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def qapp():
    """
    Session-scoped Qt application fixture.
    Reuses existing app instance or creates new one for GUI tests.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Don't quit the app - let pytest handle cleanup


@pytest.fixture
def qtbot_timeout():
    """Default timeout for qtbot.waitSignal operations."""
    return 5000  # 5 seconds


# ============================================================================
# Synthetic IQ Data Generation
# ============================================================================


@pytest.fixture
def synthetic_iq():
    """Generate 100k samples of random complex IQ data."""
    return np.random.randn(100_000).astype(np.complex64)


def generate_tone_iq(
    freq_offset: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 0.5,
    noise_floor_db: float = -60.0,
) -> np.ndarray:
    """
    Generate a clean tone at freq_offset Hz with optional noise.

    Args:
        freq_offset: Frequency offset from center in Hz
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        amplitude: Signal amplitude (0-1)
        noise_floor_db: Noise floor in dB below signal

    Returns:
        Complex IQ samples as numpy array
    """
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    tone = amplitude * np.exp(1j * 2.0 * np.pi * freq_offset * t)

    # Add AWGN
    noise_amplitude = amplitude * 10 ** (noise_floor_db / 20)
    noise = noise_amplitude * (
        np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    )

    return (tone + noise).astype(np.complex64)


@pytest.fixture
def synthetic_nfm_iq(tmp_path):
    """
    Generate synthetic NFM modulated signal as WAV file.

    Returns path to generated WAV file with NFM signal.
    """
    carrier_offset = 5_000.0  # 5 kHz offset
    sample_rate = 2_500_000.0
    audio_freq = 1_000.0  # 1 kHz tone
    deviation = 5_000.0  # 5 kHz deviation
    duration = 1.0

    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Audio modulating signal
    audio = np.sin(2.0 * np.pi * audio_freq * t)

    # Frequency modulation
    phase = 2.0 * np.pi * carrier_offset * t + deviation * np.cumsum(audio) / sample_rate
    iq = np.exp(1j * phase).astype(np.complex64)

    # Add some noise
    noise = 0.01 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    iq = (iq + noise).astype(np.complex64)

    # Write to WAV
    path = tmp_path / "synthetic_nfm.wav"
    interleaved = np.stack([iq.real, iq.imag], axis=1)
    sf.write(path, interleaved, int(sample_rate), subtype="PCM_16")

    return path


@pytest.fixture
def synthetic_am_iq(tmp_path):
    """
    Generate synthetic AM signal as WAV file.

    Returns path to generated WAV file with AM signal.
    """
    carrier_offset = 5_000.0  # 5 kHz offset
    sample_rate = 2_500_000.0
    audio_freq = 1_000.0  # 1 kHz tone
    modulation_depth = 0.8
    duration = 1.0

    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # AM: carrier * (1 + m*audio)
    audio = np.sin(2.0 * np.pi * audio_freq * t)
    envelope = 1.0 + modulation_depth * audio
    carrier = np.exp(1j * 2.0 * np.pi * carrier_offset * t)
    iq = (envelope * carrier).astype(np.complex64)

    # Add some noise
    noise = 0.01 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    iq = (iq + noise).astype(np.complex64)

    # Write to WAV
    path = tmp_path / "synthetic_am.wav"
    interleaved = np.stack([iq.real, iq.imag], axis=1)
    sf.write(path, interleaved, int(sample_rate), subtype="PCM_16")

    return path


@pytest.fixture
def synthetic_usb_iq(tmp_path):
    """
    Generate synthetic USB (upper sideband) signal as WAV file.

    Returns path to generated WAV file with USB signal.
    """
    carrier_offset = 5_000.0
    sample_rate = 2_500_000.0
    audio_freq = 1_000.0
    duration = 1.0

    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Generate audio signal
    audio = 0.5 * np.sin(2.0 * np.pi * audio_freq * t)

    # Create analytic signal (USB)
    from scipy.signal import hilbert

    analytic = hilbert(audio)

    # Shift to carrier frequency
    carrier = np.exp(1j * 2.0 * np.pi * carrier_offset * t)
    iq = (analytic * carrier).astype(np.complex64)

    # Write to WAV
    path = tmp_path / "synthetic_usb.wav"
    interleaved = np.stack([iq.real, iq.imag], axis=1)
    sf.write(path, interleaved, int(sample_rate), subtype="PCM_16")

    return path


# ============================================================================
# Hypothesis Strategies for Property-Based Testing
# ============================================================================


@st.composite
def iq_samples(
    draw, min_size=100, max_size=10000, max_amplitude=10.0
):
    """
    Hypothesis strategy for generating complex IQ samples.

    Args:
        min_size: Minimum number of samples
        max_size: Maximum number of samples
        max_amplitude: Maximum amplitude for real/imag parts

    Returns:
        Complex numpy array of IQ samples
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    real = draw(
        st.lists(
            st.floats(
                min_value=-max_amplitude, max_value=max_amplitude, allow_nan=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    imag = draw(
        st.lists(
            st.floats(
                min_value=-max_amplitude, max_value=max_amplitude, allow_nan=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    return np.array(real, dtype=np.float32) + 1j * np.array(imag, dtype=np.float32)


@st.composite
def valid_decimation_factors(draw):
    """Generate valid decimation factors (2-100)."""
    return draw(st.integers(min_value=2, max_value=100))


@st.composite
def valid_sample_rates(draw):
    """Generate valid sample rates (8kHz - 10MHz)."""
    return draw(st.floats(min_value=8e3, max_value=10e6))


# ============================================================================
# Test Data Paths
# ============================================================================


@pytest.fixture
def testfiles_dir():
    """Path to testfiles directory."""
    _TESTFILES_DIR.mkdir(parents=True, exist_ok=True)
    return _TESTFILES_DIR


@pytest.fixture
def am_test_file(testfiles_dir):
    """Path to AM test file if it exists."""
    return _ensure_fixture("fc-132334577Hz-ft-132300000-AM.wav")


@pytest.fixture
def nfm_test_file(testfiles_dir):
    """Path to NFM test file if it exists."""
    return _ensure_fixture("fc-456834049Hz-ft-455837500-ft2-456872500-NFM.wav")


# ============================================================================
# Performance Benchmark Fixtures
# ============================================================================


@pytest.fixture
def large_iq_samples():
    """Generate 1M samples for performance testing."""
    return np.random.randn(1_000_000).astype(np.complex64)


@pytest.fixture
def benchmark_sizes():
    """Common sizes for benchmark parameterization."""
    return [1024, 4096, 16384, 65536, 262144]


# ============================================================================
# Temporary Directory Management
# ============================================================================


@pytest.fixture(autouse=True)
def change_test_dir(tmp_path, monkeypatch):
    """
    Automatically change to temp directory for each test.
    Helps prevent test pollution of project directory.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ============================================================================
# Mock FFmpeg/FFprobe for Tests
# ============================================================================


@pytest.fixture
def mock_ffmpeg(monkeypatch):
    """
    Mock ffmpeg and ffprobe commands for testing without actual binaries.
    Returns dict with mock functions that can be configured per test.
    """
    mocks = {
        "ffmpeg_called": False,
        "ffprobe_called": False,
        "ffmpeg_args": None,
        "ffprobe_args": None,
    }

    def mock_ffmpeg_run(cmd, *args, **kwargs):
        mocks["ffmpeg_called"] = True
        mocks["ffmpeg_args"] = cmd
        # Return mock CompletedProcess
        from subprocess import CompletedProcess

        return CompletedProcess(cmd, 0, b"", b"")

    def mock_ffprobe_run(cmd, *args, **kwargs):
        mocks["ffprobe_called"] = True
        mocks["ffprobe_args"] = cmd
        # Return mock JSON output
        import json
        from subprocess import CompletedProcess

        output = json.dumps(
            {
                "streams": [
                    {
                        "codec_name": "pcm_s16le",
                        "sample_rate": "2500000",
                        "channels": 2,
                    }
                ]
            }
        )
        return CompletedProcess(cmd, 0, output.encode(), b"")

    # Monkey patch subprocess.run based on command
    import subprocess

    original_run = subprocess.run

    def patched_run(cmd, *args, **kwargs):
        if "ffmpeg" in str(cmd[0]):
            return mock_ffmpeg_run(cmd, *args, **kwargs)
        elif "ffprobe" in str(cmd[0]):
            return mock_ffprobe_run(cmd, *args, **kwargs)
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", patched_run)
    return mocks


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may be slow)"
    )
    config.addinivalue_line(
        "markers", "requires_ffmpeg: mark test as requiring ffmpeg installation"
    )
    config.addinivalue_line(
        "markers", "requires_large_memory: mark test as requiring >1GB RAM"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add automatic markers based on test names/locations.
    """
    for item in items:
        # Auto-mark GUI tests
        if "test_interactive" in str(item.fspath) or "qt" in item.name.lower():
            item.add_marker(pytest.mark.gui)

        # Auto-mark benchmark tests
        if "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.benchmark)

        # Auto-mark slow tests
        if "integration" in str(item.fspath) or "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
