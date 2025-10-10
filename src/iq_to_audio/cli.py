from __future__ import annotations

import argparse
import math
import logging
import sys
from pathlib import Path
from typing import Optional

from .processing import ProcessingConfig, ProcessingPipeline

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
        required=True,
        type=Path,
        help="Input SDR++ baseband WAV file.",
    )
    parser.add_argument(
        "--ft",
        dest="target_freq",
        type=positive_float,
        help="Target RF frequency in Hz.",
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
        "--deemph",
        dest="deemph_us",
        type=positive_float,
        default=300.0,
        help="FM de-emphasis time constant in microseconds (default: 300).",
    )
    parser.add_argument(
        "--squelch",
        dest="squelch_dbfs",
        type=float,
        help="Squelch threshold in dBFS. If omitted, threshold adapts to noise.",
    )
    parser.add_argument(
        "--silence-trim",
        dest="silence_trim",
        action="store_true",
        help="Drop chunks that fall below squelch threshold.",
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
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.interactive and args.target_freq is None:
        parser.error("--ft is required unless --interactive is enabled.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = ProcessingConfig(
        in_path=args.input_path,
        target_freq=args.target_freq or 0.0,
        bandwidth=args.bandwidth,
        center_freq=args.center_freq,
        fs_ch_target=args.fs_ch,
        deemph_us=args.deemph_us,
        squelch_dbfs=args.squelch_dbfs,
        silence_trim=args.silence_trim,
        output_path=args.output_path,
        dump_iq_path=args.dump_iq,
        chunk_size=args.chunk_size,
        filter_block=args.filter_block,
        iq_order=args.iq_order,
        probe_only=args.probe_only,
        mix_sign_override=args.mix_sign,
        plot_stages_path=args.plot_stages,
    )

    if args.interactive:
        try:
            from .interactive import interactive_select
        except ImportError as exc:  # pragma: no cover - user feedback only
            LOG.error("Interactive mode unavailable: %s", exc)
            return 1
        outcome = interactive_select(config, seconds=args.interactive_seconds)
        config.center_freq = outcome.center_freq
        config.target_freq = outcome.target_freq
        config.bandwidth = outcome.bandwidth
        LOG.info(
            "Interactive selection: center %.0f Hz, target %.0f Hz, bandwidth %.0f Hz",
            outcome.center_freq,
            outcome.target_freq,
            outcome.bandwidth,
        )

    pipeline = ProcessingPipeline(config)
    try:
        result = pipeline.run()
    except Exception as exc:  # pragma: no cover - ensure user friendly exit
        LOG.error("Processing failed: %s", exc)
        if args.verbose:
            LOG.exception("Debug traceback")
        return 1

    if args.probe_only:
        info = result.sample_rate_probe
        print(
            f"Sample rate: {info.value:.2f} Hz (ffprobe={info.ffprobe}, header={info.header}, wave={info.wave})"
        )
        print(
            f"Center frequency: {result.center_freq:.0f} Hz, target: {result.target_freq:.0f} Hz, "
            f"offset: {result.freq_offset:.0f} Hz"
        )
        print(
            f"Channel decimation: {result.decimation} -> {result.fs_channel:.2f} Hz, mixer sign {result.mix_sign}"
        )
    else:
        if result.audio_peak > 0:
            peak_db = 20.0 * math.log10(result.audio_peak)
            print(f"Audio peak level: {peak_db:.2f} dBFS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
