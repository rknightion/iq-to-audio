import logging

from iq_to_audio.benchmark import run_benchmark


def test_benchmark_smoke(caplog):
    base_kwargs = {
        "target_freq": 0.0,
        "bandwidth": 12_500.0,
        "center_freq": None,
        "center_freq_source": None,
        "demod_mode": "nfm",
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
        "fft_workers": 2,
    }
    with caplog.at_level(logging.INFO, logger="iq_to_audio.benchmark"):
        code = run_benchmark(
            seconds=0.2,
            sample_rate=200_000.0,
            freq_offset=5_000.0,
            center_freq=400_000_000.0,
            target_freq=400_005_000.0,
            base_kwargs=base_kwargs,
        )
    assert code == 0
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Running benchmark" in messages
    assert "Benchmark processed" in messages
