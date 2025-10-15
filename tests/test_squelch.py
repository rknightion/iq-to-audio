from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from iq_to_audio.squelch import (
    AudioPostOptions,
    SquelchConfig,
    apply_squelch,
    gather_audio_targets,
    process_audio_batch,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def _synthetic_audio(
    rng: np.random.Generator,
    *,
    seconds: float,
    sample_rate: int = 48_000,
    tone_freq: float = 1_200.0,
) -> tuple[np.ndarray, int]:
    total = int(seconds * sample_rate)
    time = np.arange(total, dtype=np.float32) / sample_rate
    tone = 0.6 * np.sin(2.0 * np.pi * tone_freq * time, dtype=np.float32)
    tone_mask = (time > 0.4) & (time < seconds - 0.4)
    tone = tone * tone_mask.astype(np.float32)
    noise = 0.05 * rng.standard_normal(total, dtype=np.float32)
    return (tone + noise)[:, np.newaxis], sample_rate


def test_apply_squelch_adaptive_reduces_noise(rng: np.random.Generator) -> None:
    audio, sample_rate = _synthetic_audio(rng, seconds=2.5)
    config = SquelchConfig(method="adaptive")
    cleaned, noise_floor_db, threshold_db = apply_squelch(audio, float(sample_rate), config)
    assert cleaned.shape[0] <= audio.shape[0]
    assert noise_floor_db < threshold_db
    assert math.isclose(noise_floor_db, -26.0, abs_tol=8.0)
    assert threshold_db > noise_floor_db


def test_apply_squelch_transient_highlights_bursts(rng: np.random.Generator) -> None:
    audio, sample_rate = _synthetic_audio(rng, seconds=1.8)
    config = SquelchConfig(method="transient", trim_silence=False)
    cleaned, _, _ = apply_squelch(audio, float(sample_rate), config)
    active_ratio = float(np.count_nonzero(np.abs(cleaned) > 1e-4) / cleaned.size)
    assert active_ratio < 0.5


def test_process_audio_batch_creates_cleaned_files(tmp_path: Path, rng: np.random.Generator) -> None:
    audio, sample_rate = _synthetic_audio(rng, seconds=2.0)
    source = tmp_path / "sample.wav"
    sf.write(source.as_posix(), audio, sample_rate, subtype="PCM_16")
    options = AudioPostOptions(
        config=SquelchConfig(method="adaptive", trim_silence=True),
        overwrite=False,
        cleaned_suffix="-clean",
    )
    targets = gather_audio_targets(source, options)
    summary = process_audio_batch(targets, options)
    assert summary.processed == 1
    result = summary.results[0]
    assert result.output_path.exists()
    assert result.output_path.suffix == ".wav"
    assert result.bytes_out <= result.bytes_in
    with sf.SoundFile(result.output_path) as handle:
        assert handle.frames == result.samples_out
