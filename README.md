# IQ to Audio

Python CLI for extracting a narrowband FM (NFM) channel from large SDR++ baseband WAV recordings. The tool streams I/Q samples from disk, isolates the requested RF slice, demodulates to audio, and writes a mono 48 kHz WAV.

## Features

- Streams multi‑gigabyte SDR++ recordings without loading them into memory.
- Auto-recovers center frequency from filenames (override with `--fc`) and probes true sample rate via `ffprobe`.
- Modular decoding pipeline with pluggable stages supporting NFM, AM, USB, and LSB demodulation.
- Interactive Tk/Matplotlib workspace that covers file discovery, spectrum preview (with adjustable FFT size, theme, smoothing, dynamic range) and a companion waterfall view for time-varying activity, plus selection and run-time monitoring.
- Live progress bars for each DSP stage plus overall completion in both CLI and interactive flows.
- Optional PSD snapshots, channelized IQ dumps, and probe-only mode for diagnostics.
- Built-in synthetic benchmarking harness (`--benchmark`) for repeatable throughput measurements.

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

FFmpeg must be on your `PATH` for ingestion and WAV output. Install the optional extras when needed:

- `uv run --extra interactive iq-to-audio --interactive`: pulls in Matplotlib for the GUI/plotting workflow.
- `uv run --group dev pytest`: installs development tooling (pytest) for local testing.
- System Tk packages (`python3-tk`, `python-tk@3.14`, etc.) are required for the interactive GUI; see your OS package manager.

## Usage

```bash
uv run iq-to-audio \
  --in baseband_456834049Hz_14-17-29_09-10-2025.wav \
  --ft 453112500 \
  --bw 7000 \
  --preview 5
```

Running `uv run iq-to-audio` with no extra flags launches the interactive GUI; add `--cli` to stay entirely in terminal mode.

The CLI prints per-stage progress bars (ingest, channelize, demodulate, encode) and an overall percentage. The interactive GUI shows matching progress in a dedicated window once processing begins.

Interactive mode now exposes demod selection, AGC control (for SSB modes), spectrum tools (FFT size, smoothing, dynamic range, color themes, pan/zoom toolbar), and an auto-launched waterfall so you can dial in weak signals just like an SDR waterfall. Squelch/silence trim toggles remain in the UI as placeholders for future enhancements, and target frequency boxes let you queue up to five channels—leave extras blank to process only the active selection. Drag to highlight a channel, use the mouse wheel or double-click to zoom, then click “Preview DSP” to demodulate the current preview window or “Confirm & Run” to process the full capture. Adjust options as needed, then hit “Refresh preview” to regenerate the plots.
Toggle “Analyze entire recording” in the GUI to average the full capture into the preview spectrum when you need maximum frequency resolution.

## CLI Arguments

- `--in PATH` (required unless `--interactive`): SDR++ baseband WAV input.
- `--ft HZ`: target RF frequency (required unless using interactive selection); repeat up to five times to batch multiple channels.
- `--bw HZ`: channel bandwidth (default 12 500).
- `--fc HZ`: override center frequency if filename parsing fails.
- `--fs-ch HZ`: complex channel sample rate before demod (default 96 000).
- `--demod MODE`: choose demodulator (`nfm`, `am`, `usb`, `lsb`, `ssb` alias for `usb`).
- `--deemph µs`: FM deemphasis time constant (default 300).
- `--no-agc`: disable automatic gain control in supported demodulators.
- `--preview SECONDS`: process only the first `SECONDS` of the recording and exit (preview mode).
- `--cli`: force CLI mode (default with no flag launches the interactive GUI).
- `--out PATH`: override output WAV (default `audio_<FT>_48k.wav` beside input).
- `--dump-iq PATH`: write channelized complex float32 IQ stream for debugging.
- `--plot-stages PATH`: save PSD snapshots for key DSP stages to a PNG.
- `--chunk N`: complex samples per processing block (default 1 048 576).
- `--fft-workers N`: override automatic FFT worker selection for filter/demod stages.
- `--filter-block N`: FFT overlap-save block size for the channel filter (default 65 536).
- `--iq-order {iq,qi,iq_inv,qi_inv}`: interpret stereo order and polarity.
- `--mix-sign {-1,1}`: manually override automatic mixer sign selection.
- `--probe-only`: inspect derived parameters and exit without demodulating.
- `--interactive`: launch the Tk/Matplotlib UI (no other args required).
- `--interactive-seconds SEC`: snapshot duration for the interactive spectrum (default 2.0).
- `--benchmark`: generate a synthetic capture, run the full pipeline, and report throughput; exits afterward.
- `--benchmark-seconds SEC`: duration of the synthetic capture used for benchmarking (default 5).
- `--benchmark-sample-rate HZ`: sample rate for the synthetic benchmark capture (default 2.5 MHz).
- `--benchmark-offset HZ`: frequency offset between synthetic center and target (default 25 000 Hz).
- `--verbose`: enable verbose logging (stack traces on failure when combined with `--verbose`).

## Examples

```bash
# Standard CLI demodulation
uv run iq-to-audio --in capture.wav --fc 456834049 --ft 453112500 --bw 9000

# Probe-only metadata inspection
uv run iq-to-audio --in capture.wav --fc 456834049 --ft 453112500 --probe-only

# Dump channelized IQ and generate stage plots for debugging
uv run iq-to-audio --in capture.wav --fc 456834049 --ft 453112500 \
  --dump-iq debug.cf32 --plot-stages plots/stages.png

# Medium-wave AM broadcast
uv run iq-to-audio --in mw_capture.wav --fc 1015000 --ft 1010000 --demod am --bw 10000

# 5-second CLI preview to validate settings
uv run iq-to-audio --in capture.wav --fc 456834049 --ft 453112500 --preview 5

# Batch demodulate three targets in one pass
uv run iq-to-audio --in capture.wav --fc 456834049 --ft 453112500 --ft 453137500 --ft 453200000 --out multi.wav

# Synthetic benchmark at 2 MS/s with AM demod
uv run iq-to-audio --benchmark --demod am --benchmark-sample-rate 2000000 --benchmark-seconds 3

# LSB voice monitoring with AGC disabled
uv run iq-to-audio --in hf_voice.wav --fc 7105000 --ft 7090000 --demod lsb --no-agc

# Interactive GUI workflow (requires matplotlib extra + Tk)
uv run --extra interactive iq-to-audio --interactive
```

## Testing

```bash
uv run --group dev pytest
```

## Notes

- FFmpeg handles resilient WAV ingestion (`-ignore_length 1`) and resampling/encoding to 48 kHz output.
- Missing `ffmpeg`/`ffprobe` executables now yield actionable installation hints instead of generic failures.
- The waterfall view mirrors SDR++: adjust colormap, slice density, or dynamic range, and click directly on a signal trace to retune the main selector.
- Preview runs (`--preview` or the GUI "Preview DSP" button) emit an audio file with a `_preview` suffix so you can confirm settings before processing the full capture.
- Progress bars never exceed 100 %: totals are estimated from file size, decimation, and audio duration, then clamped.
- The decoder stack is modular—new demodulators can be added under `iq_to_audio/decoders` with minimal pipeline changes.
- Interactive spectrum controls (FFT size, smoothing, theme, dynamic range) mirror SDR-style UX; use the toolbar to pan/zoom and highlight weak signals. Remember to hit “Refresh preview” after changing spectrum/waterfall/decoder options so plots stay in sync.
- Ensure system Tk libraries are installed before launching `--interactive`; missing dependencies trigger a helpful installation hint.
