from pathlib import Path

import pytest

from iq_to_audio import utils
from iq_to_audio.utils import detect_center_frequency, parse_center_frequency


def test_parse_center_frequency_success():
    path = Path("baseband_456834049Hz_14-17-29_09-10-2025.wav")
    assert parse_center_frequency(path) == 456834049.0


def test_parse_center_frequency_missing():
    path = Path("capture.wav")
    assert parse_center_frequency(path) is None


def test_detect_center_frequency_filename_sdrsharp():
    path = Path("16-04-05_457026566Hz.wav")
    result = detect_center_frequency(path)
    assert result.value == pytest.approx(457026566.0)
    assert result.source.startswith("filename")


def test_detect_center_frequency_prefers_metadata(monkeypatch, tmp_path):
    target = tmp_path / "capture.wav"
    target.write_bytes(b"")

    monkeypatch.setattr(utils, "_soundfile_tags", lambda _path: {"center_frequency": "462.5 MHz"})
    monkeypatch.setattr(utils, "_ffprobe_tags", lambda _path: {})

    result = detect_center_frequency(target)
    assert result.value == pytest.approx(462_500_000.0)
    assert result.source == "metadata:center_frequency"
