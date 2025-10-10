# IQ to Audio

Python CLI for extracting a narrowband FM (NFM) channel from large SDR++ baseband WAV recordings. The tool streams I/Q samples from disk, isolates the requested RF slice, demodulates to audio, applies optional squelch or silence trimming, and writes a mono 48 kHz WAV.

## Features

- Parses SDR++ filenames to recover center frequency automatically (override with `--fc`).
- Probes true sample rate via `ffprobe -ignore_length 1`, falling back to libsndfile headers.
- Streams multi‑gigabyte recordings without loading them into memory.
- Channelizes with an FFT-based FIR low-pass/decimator and quadrature FM demodulator.
- Applies configurable de-emphasis and adaptive squelch (or user threshold).
- Outputs resampled, limiter-protected mono audio via FFmpeg; optional channelized IQ dump for debugging.

## Installation (with uv)

This project is ready to run with [uv](https://github.com/astral-sh/uv), which resolves dependencies directly from `pyproject.toml`.

```bash
uv run iq-to-audio --help
```

`uv run` will build a temporary environment, install the package (and its dependencies) on demand, and execute the console entry-point. If you prefer a persistent environment, you can create one and sync dependencies:

```bash
uv venv
source .venv/bin/activate  # or uv run --project .
uv pip install -e .
```

FFmpeg must be on your `PATH` for ingestion and WAV output.

## Usage

```bash
uv run iq-to-audio \
  --in baseband_456834049Hz_14-17-29_09-10-2025.wav \
  --ft 453112500 \
  --bw 7000 \
  --squelch -45
```

Key options:

- `--fc` override center frequency if filename parsing fails.
- `--fs-ch` choose the complex channel rate before demod (default 96 kHz).
- `--deemph` set the de-emphasis time constant in microseconds.
- `--silence-trim` drop chunks below the squelch threshold instead of inserting quiet audio.
- `--dump-iq` write intermediate channelized IQ as `cf32` for debugging.
- `--probe-only` inspect derived parameters without demodulating.
- `--verbose` enable detailed logging for DSP stages.
- `--interactive` open a matplotlib spectrum viewer to choose the RF channel visually (drag to set span, double-click or press Enter to confirm). Adjust snapshot duration with `--interactive-seconds`.
- `--plot-stages` save FFT snapshots (PNG) for the first chunk at key DSP stages (input, mixed, filtered, decimated, demod, deemphasis, squelch) for debugging and documentation.

## Testing

```bash
uv run --with dev pytest
```

## Notes

- FFmpeg is used both for resilient WAV ingestion (`-ignore_length 1`) and final WAV encoding/resampling to 48 kHz.
- Processing defaults target land-mobile style NFM voice (7–12.5 kHz bandwidth) but can be tuned for other narrowband signals.
- The adaptive squelch estimates noise power while closed and opens ~6 dB above the noise floor; provide `--squelch` to set a fixed threshold.
- Sample rate probing uses ffprobe when available, falling back to libsndfile metadata and finally the standard library `wave` reader.
- Matplotlib powers optional visualization; install a backend (e.g. Tk, Qt) when using the interactive picker on headless systems.
