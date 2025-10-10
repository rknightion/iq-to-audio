from pathlib import Path

from iq_to_audio.utils import parse_center_frequency


def test_parse_center_frequency_success():
    path = Path("baseband_456834049Hz_14-17-29_09-10-2025.wav")
    assert parse_center_frequency(path) == 456834049.0


def test_parse_center_frequency_missing():
    path = Path("capture.wav")
    assert parse_center_frequency(path) is None
