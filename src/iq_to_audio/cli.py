from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from . import __version__
from .input_formats import parse_user_format
from .preview import run_preview
from .processing import (
    ProcessingCancelled,
    ProcessingConfig,
    ProcessingPipeline,
    ProcessingResult,
)
from .progress import TqdmProgressSink
from .squelch import AudioPostOptions, SquelchConfig, gather_audio_targets, process_audio_batch

LOG = logging.getLogger("iq_to_audio")


def positive_float(value: str) -> float:
    try:
        val = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if val <= 0:
        raise argparse.ArgumentTypeError("Expected a positive value.")
    return val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and demodulate a narrowband FM channel from SDR++ baseband WAV recordings.",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        required=False,
        type=Path,
        help="Input SDR++ baseband WAV file.",
    )
    parser.add_argument(
        "--ft",
        dest="target_freqs",
        type=positive_float,
        action="append",
        default=None,
        help="Target RF frequency in Hz. Supply up to five times to batch additional channels.",
    )
    parser.add_argument(
        "--bw",
        dest="bandwidth",
        type=positive_float,
        default=12_500.0,
        help="Channel bandwidth in Hz (default: 12500).",
    )
    parser.add_argument(
        "--fc",
        dest="center_freq",
        type=positive_float,
        help="Override center frequency in Hz if filename parsing fails.",
    )
    parser.add_argument(
        "--fs-ch",
        dest="fs_ch",
        type=positive_float,
        default=96_000.0,
        help="Desired complex channel sample rate prior to demod (default: 96 kHz).",
    )
    parser.add_argument(
        "--demod",
        dest="demod",
        choices=["nfm", "am", "usb", "lsb", "ssb", "none"],
        default="nfm",
        help="Demodulator to use (nfm, am, usb, lsb, ssb=alias for usb, none=no demodulation). Default: nfm.",
    )
    parser.add_argument(
        "--deemph",
        dest="deemph_us",
        type=positive_float,
        default=300.0,
        help="FM de-emphasis time constant in microseconds (default: 300).",
    )
    parser.add_argument(
        "--no-agc",
        dest="agc_enabled",
        action="store_false",
        help="Disable automatic gain control in supported demodulators.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        help="Output WAV path. Defaults to audio_<FT>_48k.wav alongside input.",
    )
    parser.add_argument(
        "--dump-iq",
        dest="dump_iq",
        type=Path,
        help="Optional path to write channelized complex float32 IQ (cf32).",
    )
    parser.add_argument(
        "--plot-stages",
        dest="plot_stages",
        type=Path,
        help="Save PSD plots for major pipeline stages to the given PNG path.",
    )
    parser.add_argument(
        "--chunk",
        dest="chunk_size",
        type=int,
        default=1_048_576,
        help="Complex samples per processing chunk (default: 1,048,576).",
    )
    parser.add_argument(
        "--fft-workers",
        dest="fft_workers",
        type=int,
        help="Number of worker threads for FFT-based stages (default: auto).",
    )
    parser.add_argument(
        "--filter-block",
        dest="filter_block",
        type=int,
        default=65_536,
        help="FFT block size for channel filter overlap-save (default: 65536).",
    )
    parser.add_argument(
        "--iq-order",
        dest="iq_order",
        choices=["iq", "qi", "iq_inv", "qi_inv"],
        default="iq",
        help="Interpretation of the stereo channels: iq (default), qi, iq_inv, qi_inv.",
    )
    parser.add_argument(
        "--input-format",
        dest="input_format",
        type=str,
        help="Override input encoding (wav-s16, wav-u8, wav-f32, raw-cu8, raw-cs16, raw-cf32).",
    )
    parser.add_argument(
        "--input-sample-rate",
        dest="input_sample_rate",
        type=positive_float,
        help="Manual input sample rate in Hz (used when headers are missing).",
    )
    parser.add_argument(
        "--mix-sign",
        dest="mix_sign",
        type=int,
        choices=[-1, 1],
        help="Override automatic mixer sign selection.",
    )
    parser.add_argument(
        "--probe-only",
        dest="probe_only",
        action="store_true",
        help="Probe metadata and exit without demodulating.",
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Launch interactive spectrum viewer to pick frequency/bandwidth.",
    )
    parser.add_argument(
        "--interactive-seconds",
        dest="interactive_seconds",
        type=positive_float,
        default=2.0,
        help="Duration of IQ snapshot (seconds) for interactive mode (default: 2.0).",
    )
    parser.add_argument(
        "--preview",
        dest="preview_seconds",
        type=positive_float,
        help="Preview only the first SECONDS of the recording and exit.",
    )
    parser.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="Run a synthetic throughput benchmark and exit.",
    )
    parser.add_argument(
        "--benchmark-seconds",
        dest="benchmark_seconds",
        type=positive_float,
        default=5.0,
        help="Duration of synthetic capture in seconds when benchmarking (default: 5).",
    )
    parser.add_argument(
        "--benchmark-sample-rate",
        dest="benchmark_sample_rate",
        type=positive_float,
        default=2_500_000.0,
        help="Sample rate in Hz for synthetic benchmark captures (default: 2.5e6).",
    )
    parser.add_argument(
        "--benchmark-offset",
        dest="benchmark_offset",
        type=float,
        default=25_000.0,
        help="Frequency offset (Hz) between center and target for benchmark tone (default: 25 kHz).",
    )
    parser.add_argument(
        "--cli",
        dest="cli",
        action="store_true",
        help="Run in CLI mode (default launches the interactive GUI).",
    )
    parser.add_argument(
        "--audio-post",
        dest="audio_post_path",
        type=Path,
        help="Apply audio post-processing (auto squelch) to the given file or directory.",
    )
    parser.add_argument(
        "--audio-post-mode",
        dest="audio_post_mode",
        choices=["adaptive", "static", "transient"],
        default="adaptive",
        help="Squelch algorithm to use when --audio-post is supplied (default: adaptive).",
    )
    parser.add_argument(
        "--audio-post-noise-floor",
        dest="audio_post_noise_floor",
        type=float,
        help="Manual noise floor in dBFS for --audio-post (auto-detected by default).",
    )
    parser.add_argument(
        "--audio-post-noise-percentile",
        dest="audio_post_percentile",
        type=float,
        default=0.2,
        help="Percentile used for auto noise floor estimation (default: 0.2 → 20th percentile).",
    )
    parser.add_argument(
        "--audio-post-threshold",
        dest="audio_post_threshold",
        type=float,
        default=6.0,
        help="Margin above noise floor in dBFS for the squelch threshold (default: 6).",
    )
    parser.add_argument(
        "--audio-post-lead",
        dest="audio_post_lead",
        type=float,
        default=0.15,
        help="Lead-in seconds retained when trimming silence (default: 0.15).",
    )
    parser.add_argument(
        "--audio-post-trail",
        dest="audio_post_trail",
        type=float,
        default=0.35,
        help="Trailing seconds retained when trimming silence (default: 0.35).",
    )
    parser.add_argument(
        "--audio-post-no-trim",
        dest="audio_post_trim",
        action="store_false",
        help="Disable silence trimming when performing --audio-post.",
    )
    parser.add_argument(
        "--audio-post-overwrite",
        dest="audio_post_overwrite",
        action="store_true",
        help="Overwrite original files when performing --audio-post (default writes -cleaned copies).",
    )
    parser.add_argument(
        "--audio-post-suffix",
        dest="audio_post_suffix",
        default="-cleaned",
        help="Suffix to append when writing cleaned copies (default: -cleaned).",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the iq-to-audio version and exit.",
    )
    parser.set_defaults(agc_enabled=True)
    parser.set_defaults(audio_post_trim=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cli and args.interactive:
        parser.error("--cli cannot be combined with --interactive.")
    if args.audio_post_path and args.interactive:
        parser.error("--audio-post cannot be combined with --interactive.")
    if args.audio_post_path and args.benchmark:
        parser.error("--audio-post cannot be combined with --benchmark.")
    if args.audio_post_path and not 0.0 <= args.audio_post_percentile <= 1.0:
        parser.error("--audio-post-noise-percentile must be between 0.0 and 1.0.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.audio_post_path:
        squelch_config = SquelchConfig(
            method=args.audio_post_mode,
            auto_noise_floor=args.audio_post_noise_floor is None,
            manual_noise_floor_db=args.audio_post_noise_floor,
            noise_floor_percentile=args.audio_post_percentile,
            threshold_margin_db=args.audio_post_threshold,
            trim_silence=args.audio_post_trim,
            trim_lead_seconds=args.audio_post_lead,
            trim_trail_seconds=args.audio_post_trail,
        )
        post_options = AudioPostOptions(
            config=squelch_config,
            overwrite=args.audio_post_overwrite,
            cleaned_suffix=args.audio_post_suffix,
        )
        try:
            post_targets = gather_audio_targets(args.audio_post_path, post_options)
        except Exception as exc:
            LOG.error("Unable to enumerate audio targets: %s", exc)
            return 1
        if not post_targets:
            LOG.error("No audio files found at %s.", args.audio_post_path)
            return 1
        LOG.info(
            "Audio post-processing %d file(s) via %s squelch (%s).",
            len(post_targets),
            squelch_config.method,
            "overwrite" if post_options.overwrite else f"suffix '{post_options.cleaned_suffix}'",
        )

        def _progress(completed: int, total: int, current: Path) -> None:
            if total <= 0:
                LOG.info("Processing %s", current)
                return
            completed = max(0, min(completed, total))
            pct = (completed / total) * 100.0
            LOG.info(" [%6.2f%%] %s", pct, current)

        summary = process_audio_batch(post_targets, post_options, progress_cb=_progress)
        for item in summary.results:
            retained_pct = item.retained_ratio * 100.0
            LOG.info(
                "%s -> %s | %.2fs → %.2fs | %.1f%% retained | floor %.1f dB | threshold %.1f dB",
                item.input_path,
                item.output_path,
                item.duration_in,
                item.duration_out,
                retained_pct,
                item.noise_floor_db,
                item.threshold_db,
            )
        if summary.errors:
            LOG.error("Audio post-processing failed on %d file(s).", summary.failed)
            for path, error in summary.errors:
                LOG.error(" - %s: %s", path, error)
            return 1
        LOG.info(
            "Audio post-processing complete: Δsize %+d bytes, Δduration %+0.2f s.",
            summary.aggregate_size_delta(),
            summary.aggregate_duration_delta(),
        )
        return 0

    frequencies: list[float] = list(args.target_freqs or [])

    input_format_value: str | None = None
    input_container: str | None = None
    input_format_source: str | None = None
    if args.input_format:
        try:
            container, codec = parse_user_format(args.input_format, default_container=None)
        except ValueError as exc:
            parser.error(f"--input-format: {exc}")
        input_format_value = codec
        input_container = container
        input_format_source = "cli"

    if len(frequencies) > 5:
        parser.error("At most five target frequencies are supported per run.")
    seen: list[float] = []
    for freq in frequencies:
        for prior in seen:
            if math.isclose(freq, prior, rel_tol=0.0, abs_tol=0.5):
                parser.error("Duplicate target frequencies are not allowed.")
        seen.append(freq)

    def annotate_path(base: Path | None, freq: float, total: int) -> Path | None:
        if base is None or total <= 1:
            return base
        freq_tag = int(round(freq))
        return base.with_name(f"{base.stem}_{freq_tag}{base.suffix}")

    shared_kwargs = {
        "bandwidth": args.bandwidth,
        "center_freq": args.center_freq,
        "center_freq_source": "cli" if args.center_freq is not None else None,
        "demod_mode": args.demod,
        "fs_ch_target": args.fs_ch,
        "deemph_us": args.deemph_us,
        "agc_enabled": args.agc_enabled,
        "chunk_size": args.chunk_size,
        "filter_block": args.filter_block,
        "iq_order": args.iq_order,
        "probe_only": args.probe_only,
        "mix_sign_override": args.mix_sign,
        "fft_workers": args.fft_workers,
        "input_format": input_format_value,
        "input_container": input_container,
        "input_format_source": input_format_source,
        "input_sample_rate": args.input_sample_rate,
    }
    base_kwargs = shared_kwargs.copy()
    base_kwargs.update(
        target_freq=frequencies[0] if frequencies else 0.0,
        target_freqs=list(frequencies),
        output_path=args.output_path,
        dump_iq_path=args.dump_iq,
        plot_stages_path=args.plot_stages,
    )

    if args.benchmark and args.interactive:
        parser.error("--benchmark cannot be combined with --interactive.")

    if args.benchmark:
        from .benchmark import run_benchmark

        benchmark_kwargs = dict(base_kwargs)
        benchmark_kwargs.pop("target_freqs", None)
        try:
            return run_benchmark(
                seconds=args.benchmark_seconds,
                sample_rate=args.benchmark_sample_rate,
                freq_offset=args.benchmark_offset,
                center_freq=args.center_freq,
                target_freq=frequencies[0] if frequencies else None,
                base_kwargs=benchmark_kwargs,
            )
        except Exception as exc:
            LOG.error("Benchmark failed: %s", exc)
            if args.verbose:
                LOG.exception("Benchmark error details")
            return 1

    progress_sink = None
    configured_configs: list[ProcessingConfig] = []

    launch_gui = args.interactive or (not args.cli and not args.benchmark)

    if launch_gui:
        try:
            from .interactive import launch_interactive_session
        except ImportError as exc:  # pragma: no cover - user feedback only
            LOG.error("Interactive mode unavailable: %s", exc)
            return 1
        try:
            session = launch_interactive_session(
                input_path=args.input_path,
                base_kwargs=base_kwargs,
                snapshot_seconds=args.interactive_seconds,
            )
            configured_configs = list(session.configs)
            progress_sink = session.progress_sink
        except KeyboardInterrupt:
            LOG.info("Interactive session cancelled.")
            return 0
        except Exception as exc:
            LOG.error("Interactive session failed: %s", exc)
            if args.verbose:
                LOG.exception("Interactive error details")
            return 1
    else:
        if args.input_path is None:
            parser.error("--in is required in CLI mode.")
        if not frequencies:
            parser.error("Provide at least one --ft target frequency in CLI mode.")

    if args.preview_seconds is not None:
        if launch_gui:
            LOG.warning(
                "--preview is ignored in interactive mode; use the GUI preview button instead."
            )
        else:
            total = len(frequencies)

            for index, freq in enumerate(frequencies, start=1):
                freq_output = annotate_path(args.output_path, freq, total)
                freq_dump = annotate_path(args.dump_iq, freq, total)
                freq_plot = annotate_path(args.plot_stages, freq, total)
                config = ProcessingConfig(
                    in_path=args.input_path,
                    target_freq=freq,
                    output_path=freq_output,
                    dump_iq_path=freq_dump,
                    plot_stages_path=freq_plot,
                    **shared_kwargs,
                )
                LOG.info(
                    "=== Previewing target %.0f Hz (%d/%d) ===",
                    freq,
                    index,
                    total,
                )
                try:
                    preview_sink = TqdmProgressSink()
                except RuntimeError as exc:
                    LOG.warning("Progress reporting disabled: %s", exc)
                    preview_sink = None
                try:
                    _, preview_path = run_preview(
                        config,
                        args.preview_seconds,
                        progress_sink=preview_sink,
                    )
                except ProcessingCancelled:
                    LOG.info("Preview cancelled by user.")
                    return 0
                except Exception as exc:
                    LOG.error("Preview failed for %.0f Hz: %s", freq, exc)
                    if args.verbose:
                        LOG.exception("Preview error details")
                    return 1
                LOG.info("Preview written to %s", preview_path)
            return 0

    if not launch_gui:
        total = len(frequencies)
        configured_configs = []
        for freq in frequencies:
            freq_output = annotate_path(args.output_path, freq, total)
            freq_dump = annotate_path(args.dump_iq, freq, total)
            freq_plot = annotate_path(args.plot_stages, freq, total)
            config = ProcessingConfig(
                in_path=args.input_path,
                target_freq=freq,
                output_path=freq_output,
                dump_iq_path=freq_dump,
                plot_stages_path=freq_plot,
                **shared_kwargs,
            )
            configured_configs.append(config)
    total = len(configured_configs)
    if total == 0:
        LOG.info("No target frequencies to process.")
        return 0

    results: list[tuple[ProcessingConfig, ProcessingResult]] = []
    for index, config in enumerate(configured_configs, start=1):
        LOG.info(
            "=== Processing target %.0f Hz (%d/%d) ===",
            config.target_freq,
            index,
            total,
        )
        pipeline = ProcessingPipeline(config)
        if launch_gui and index == 1 and progress_sink is not None:
            sink = progress_sink
            progress_sink = None
        else:
            try:
                sink = TqdmProgressSink()
            except RuntimeError as exc:
                LOG.warning("Progress reporting disabled: %s", exc)
                sink = None
        try:
            result = pipeline.run(progress_sink=sink)
        except ProcessingCancelled:
            LOG.info("Processing cancelled by user.")
            return 0
        except Exception as exc:  # pragma: no cover - ensure user friendly exit
            LOG.error("Processing failed for %.0f Hz: %s", config.target_freq, exc)
            if args.verbose:
                LOG.exception("Debug traceback")
            return 1
        results.append((config, result))

    if args.probe_only:
        for config, result in results:
            info = result.sample_rate_probe
            print(
                f"[{int(round(config.target_freq))}] Sample rate: {info.value:.2f} Hz "
                f"(ffprobe={info.ffprobe}, header={info.header}, wave={info.wave})"
            )
            print(
                f"[{int(round(config.target_freq))}] Center frequency: {result.center_freq:.0f} Hz, "
                f"target: {result.target_freq:.0f} Hz, offset: {result.freq_offset:.0f} Hz"
            )
            print(
                f"[{int(round(config.target_freq))}] Channel decimation: {result.decimation} "
                f"-> {result.fs_channel:.2f} Hz, mixer sign {result.mix_sign}"
            )
    else:
        for config, result in results:
            if result.audio_peak > 0:
                peak_db = 20.0 * math.log10(result.audio_peak)
                mode = (config.demod_mode or "").lower()
                if mode == "none":
                    print(
                        f"[{int(round(config.target_freq))}] IQ slice peak magnitude: {peak_db:.2f} dBFS"
                    )
                else:
                    print(
                        f"[{int(round(config.target_freq))}] Audio peak level: {peak_db:.2f} dBFS"
                    )

    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
