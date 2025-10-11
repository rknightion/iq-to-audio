from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from iq_to_audio.input_formats import detect_input_format, parse_user_format


def _write_wave(path: Path, subtype: str) -> None:
    samples = np.zeros((32, 2), dtype=np.float32)
    sf.write(path, samples, samplerate=48_000, subtype=subtype, format="WAV")


def test_detect_input_format_wav_pcm16(tmp_path):
    path = tmp_path / "pcm16.wav"
    _write_wave(path, "PCM_16")
    detection = detect_input_format(path)
    assert detection.ok
    assert detection.spec is not None
    assert detection.spec.codec == "pcm_s16le"
    assert detection.spec.container == "wav"


def test_detect_input_format_wav_float(tmp_path):
    path = tmp_path / "float.wav"
    _write_wave(path, "FLOAT")
    detection = detect_input_format(path)
    assert detection.ok
    assert detection.spec is not None
    assert detection.spec.codec == "pcm_f32le"


def test_detect_input_format_wav_int32_rejected(tmp_path):
    path = tmp_path / "int32.wav"
    _write_wave(path, "PCM_32")
    detection = detect_input_format(path)
    assert not detection.ok
    assert detection.error is not None
    assert "32-bit" in detection.error


def test_detect_input_format_raw_extension(tmp_path):
    path = tmp_path / "capture.cu8"
    path.write_bytes(b"")
    detection = detect_input_format(path)
    assert detection.ok
    assert detection.spec is not None
    assert detection.spec.container == "raw"
    assert detection.spec.codec == "pcm_u8"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("wav-s16", ("wav", "pcm_s16le")),
        ("wav:u8", ("wav", "pcm_u8")),
        ("raw-cu8", ("raw", "pcm_u8")),
        ("cf32", ("raw", "pcm_f32le")),
        ("pcm_s16le", ("wav", "pcm_s16le")),
    ],
)
def test_parse_user_format_variants(value: str, expected: tuple[str, str]):
    assert parse_user_format(value, default_container=None) == expected


def test_parse_user_format_invalid():
    with pytest.raises(ValueError):
        parse_user_format("unknown-format", default_container=None)
