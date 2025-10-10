import numpy as np
import pytest

from iq_to_audio.decoders.common import SquelchGate
from iq_to_audio.decoders.nfm import DeemphasisFilter, QuadratureDemod
from iq_to_audio.processing import (
    Decimator,
    choose_mix_sign,
    design_channel_filter,
)
from iq_to_audio.probe import SampleRateProbe
from iq_to_audio.visualize import compute_psd


def test_decimator_preserves_sequence():
    decimator = Decimator(3)
    first = decimator.process(np.arange(9, dtype=np.complex64))
    second = decimator.process(np.arange(9, 18, dtype=np.complex64))
    combined = np.concatenate([first, second])
    expected = np.arange(0, 18, 3, dtype=np.complex64)
    np.testing.assert_allclose(combined, expected)


def test_choose_mix_sign_prefers_positive_offset():
    sample_rate = 1_000_000.0
    freq_offset = 12_500.0
    decimation = 10
    taps = design_channel_filter(sample_rate, 12_500.0, decimation)
    n = np.arange(0, int(sample_rate * 0.1))
    # Simulate a complex tone located at +freq_offset relative to center.
    warmup = np.exp(1j * 2.0 * np.pi * freq_offset * n / sample_rate).astype(np.complex64)
    sign = choose_mix_sign(warmup, sample_rate, freq_offset, taps, decimation)
    assert sign == 1


def test_quadrature_demod_and_deemphasis():
    sample_rate = 96_000.0
    freq = 1000.0
    n = np.arange(0, int(sample_rate * 0.01))
    phase = np.cumsum(2.0 * np.pi * freq / sample_rate * np.ones_like(n))
    tone = np.exp(1j * phase)
    demod = QuadratureDemod()
    audio = demod.process(tone.astype(np.complex64))
    deemph = DeemphasisFilter(300.0, sample_rate)
    shaped = deemph.process(audio)
    assert audio.size == tone.size
    assert shaped.size == audio.size
    assert np.isfinite(shaped).all()


def test_sample_rate_probe_value_prefers_wave():
    probe = SampleRateProbe(ffprobe=None, header=None, wave=48_000.0)
    assert probe.value == 48_000.0


def test_compute_psd_output_shapes():
    sr = 1_000_000.0
    t = np.arange(0, 8192)
    samples = np.exp(1j * 2.0 * np.pi * 100_000 * t / sr).astype(np.complex64)
    freqs, psd = compute_psd(samples, sr, nfft=4096)
    assert freqs.shape == psd.shape
    assert np.isfinite(psd).all()


def test_squelch_gate_hold_prevents_voice_clipping():
    gate = SquelchGate(sample_rate=96_000.0, threshold_dbfs=None, silence_trim=False)
    noise_block = np.full(9600, 0.01, dtype=np.float32)
    _, noise_threshold, noise_dbfs, dropped = gate.process(noise_block)
    assert not gate.open
    assert not dropped
    assert noise_threshold - noise_dbfs == pytest.approx(4.0, abs=0.2)

    voice_block = np.full(9600, 0.2, dtype=np.float32)
    voice_audio, voice_threshold, voice_dbfs, dropped = gate.process(voice_block)
    assert gate.open
    assert not dropped
    np.testing.assert_allclose(voice_audio, voice_block, rtol=1e-6, atol=1e-6)
    assert voice_dbfs > noise_threshold
    assert voice_threshold == pytest.approx(noise_threshold, abs=0.5)

    quiet_block = np.zeros(9600, dtype=np.float32)
    for _ in range(4):
        quiet_audio, _, _, dropped = gate.process(quiet_block)
        assert gate.open  # sustain through hold interval
        assert not dropped
        np.testing.assert_allclose(quiet_audio, quiet_block)

    closing_audio, _, _, dropped = gate.process(quiet_block)
    assert not gate.open
    assert not dropped
    np.testing.assert_allclose(closing_audio, quiet_block)


def test_squelch_gate_trims_when_closed():
    gate = SquelchGate(sample_rate=96_000.0, threshold_dbfs=None, silence_trim=True)
    _ = gate.process(np.full(9600, 0.01, dtype=np.float32))
    _ = gate.process(np.full(9600, 0.2, dtype=np.float32))
    quiet_block = np.zeros(9600, dtype=np.float32)
    for _ in range(4):
        _ = gate.process(quiet_block)
    audio, _, _, dropped = gate.process(quiet_block)
    assert not gate.open
    assert dropped
    assert audio.size == 0
