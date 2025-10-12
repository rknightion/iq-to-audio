import numpy as np
import pytest
import soundfile as sf

from iq_to_audio.decoders.nfm import DeemphasisFilter, QuadratureDemod
from iq_to_audio.input_formats import get_format
from iq_to_audio.probe import SampleRateProbe
from iq_to_audio.processing import (
    Decimator,
    IQSliceWriter,
    ProcessingCancelled,
    ProcessingConfig,
    ProcessingPipeline,
    _encode_iq_raw,
    choose_mix_sign,
    design_channel_filter,
)
from iq_to_audio.progress import ProgressSink
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


def test_iq_slice_writer_raw_roundtrip(tmp_path):
    spec = get_format("raw", "pcm_s16le")
    path = tmp_path / "slice.cs16"
    writer = IQSliceWriter(path, 48_000.0, spec)
    samples = np.array([1.0 + 0.0j, -0.5 + 0.5j], dtype=np.complex64)
    writer.write(samples)
    writer.close()
    payload = path.read_bytes()
    expected = _encode_iq_raw(samples, "pcm_s16le")
    assert payload == expected


def test_iq_slice_writer_wav_roundtrip(tmp_path):
    spec = get_format("wav", "pcm_s16le")
    path = tmp_path / "slice.wav"
    writer = IQSliceWriter(path, 32_000.0, spec)
    samples = np.array([0.25 + 0.1j, -0.75 - 0.2j, 0.5 + 0j], dtype=np.complex64)
    writer.write(samples)
    writer.close()
    data, sr = sf.read(path, always_2d=True)
    assert sr == 32_000
    assert data.shape == (samples.size, 2)
    np.testing.assert_allclose(data[:, 0], samples.real, atol=1e-3)
    np.testing.assert_allclose(data[:, 1], samples.imag, atol=1e-3)


class _AutoCancelSink(ProgressSink):
    def __init__(self, pipeline: ProcessingPipeline):
        self.pipeline = pipeline
        self._triggered = False

    def start(self, phases, *, overall_total: float) -> None:
        return

    def advance(
        self,
        phase,
        delta: float,
        *,
        overall_completed: float,
        overall_total: float,
    ) -> None:
        if not self._triggered and delta > 0:
            self._triggered = True
            self.pipeline.cancel()

    def status(self, message: str) -> None:
        return

    def close(self) -> None:
        return


def test_processing_pipeline_cancellation_cleans_outputs(tmp_path, nfm_test_file):
    output_path = tmp_path / "cancel.wav"
    config = ProcessingConfig(
        in_path=nfm_test_file,
        target_freq=455_837_500.0,
        bandwidth=12_500.0,
        center_freq=None,
        center_freq_source=None,
        demod_mode="nfm",
        fs_ch_target=96_000.0,
        deemph_us=300.0,
        agc_enabled=True,
        output_path=output_path,
        dump_iq_path=None,
        chunk_size=262_144,
        filter_block=65_536,
        iq_order="iq",
        probe_only=False,
        mix_sign_override=None,
        plot_stages_path=None,
        fft_workers=None,
    )
    pipeline = ProcessingPipeline(config)
    sink = _AutoCancelSink(pipeline)
    with pytest.raises(ProcessingCancelled):
        pipeline.run(progress_sink=sink)
    assert not output_path.exists()
