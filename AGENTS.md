# AGENTS.md

AI agent guidelines for the iq-to-audio SDR demodulation toolkit.

## Do
- Use `uv` exclusively for environment/dependency management and running commands
- Follow PEP 8: 4-space indents, snake_case functions/variables, dataclasses for configs
- Use `Path` objects for filesystem paths, never strings
- Name loggers `LOG = logging.getLogger(__name__)` at module level
- Vectorize with NumPy/SciPy—avoid per-sample loops unless profiled
- Keep DSP/utilities shared between CLI and interactive modes (no duplication)
- Mirror test layout: `tests/test_<module>.py` for every `src/iq_to_audio/<module>.py`
- Use type hints on public APIs and dataclass fields

## Don't
- Don't use pip/pipx/conda—only `uv`
- Don't mutate audio in-place—return new arrays from DSP stages
- Don't add dependencies without justification
- Don't duplicate processing logic between `cli.py` and the `interactive/` package
- Don't hard-code sample rates or frequencies - pull from config instead

## Commands

**File-scoped (preferred for speed):**
```bash
# Lint and format single file
uv run ruff check --fix path/to/file.py
uv run ruff format path/to/file.py

# Type check single file
uv run python -m mypy path/to/file.py

# Run specific test
uv run --with dev pytest tests/test_module.py::test_function
```

**Project-wide (use sparingly):**
```bash
# Full test suite
uv run --with dev pytest

# Build distributions
uv build

# CLI help
uv run iq-to-audio --help
```

**Always run after edits:** Type check and test modified files before commit.

## Project Structure

```
src/iq_to_audio/
  cli.py              # Argument parsing, entry point
  processing.py       # Core pipeline: IQReader, OverlapSaveFIR, Decimator
  interactive.py      # PySide6 GUI (Qt6 + Matplotlib)
  preview.py          # Short preview runs before full processing
  probe.py            # Auto-detect sample rate via ffprobe
  utils.py            # Filename parsing (center freq extraction)
  spectrum.py         # FFT/PSD computation, waterfall
  progress.py         # Tqdm/Qt progress sinks
  benchmark.py        # Synthetic signal throughput testing
  visualize.py        # Matplotlib helpers (deprecated in favor of interactive.py)
  decoders/
    base.py           # Decoder ABC
    nfm.py            # Narrowband FM
    am.py             # Amplitude modulation
    ssb.py            # USB/LSB single-sideband
    common.py         # Shared DSP (DC blocker, AGC)
tests/
  test_*.py           # Pytest test modules
```

**Key files:**
- `processing.py`: Start here for pipeline changes
- `interactive.py`: Qt6 UI logic (recently migrated from Tkinter)
- `decoders/`: Add new demodulators here via `Decoder` ABC
- `cli.py`: CLI argument definitions
- `README.md`: User-facing docs
- `pyproject.toml`: Dependencies and metadata

## Good Examples

**Vectorized DSP (prefer):**
```python
# decoders/nfm.py lines 50-54
instantaneous_phase = np.angle(samples)
phase_diff = np.diff(instantaneous_phase)
phase_diff = np.unwrap(phase_diff)
audio = phase_diff * (sample_rate / (2.0 * np.pi))
```

**Streaming pipeline pattern:**
```python
# processing.py lines 297-326
with IQReader(config.in_path, chunk_size, config.iq_order) as reader:
    for iq_block in reader:
        mixed = osc.mix(iq_block, mix_sign)
        filtered = fir.process(mixed)
        decimated = decimator.process(filtered)
        audio, stats = decoder.process(decimated)
        writer.write_audio(audio)
```

**Qt signal/slot threading:**
```python
# interactive.py lines 2431-2435
class _InteractiveApp(QMainWindow):
    status_signal = Signal(str, bool)  # message, is_error
    
    def __init__(self):
        self.status_signal.connect(self._set_status_ui)
```

## Bad Examples (avoid)

**Per-sample loops:**
```python
# Slow—use np.angle() instead
for i in range(len(samples)):
    phase[i] = math.atan2(samples[i].imag, samples[i].real)
```

**String paths:**
```python
# Bad: path = "/tmp/output.wav"
path = Path("/tmp/output.wav")  # Good
```

**Hardcoded parameters:**
```python
# Bad: sample_rate = 96000
sample_rate = config.fs_ch_target  # Good
```

## Testing

Run focused tests:
```bash
uv run --with dev pytest -k test_nfm_decoder
uv run --with dev pytest tests/test_processing.py -v
```

**Test fixtures (synthesize, don't load):**
```python
# tests/test_processing.py lines 10-15
@pytest.fixture
def synthetic_iq():
    t = np.arange(1000) / 1e6
    signal = np.exp(2j * np.pi * 10e3 * t)
    return signal.astype(np.complex64)
```

**Verify outputs match expectations:**
```python
def test_decimator():
    decimator = Decimator(factor=10)
    out = decimator.process(np.ones(100, dtype=np.complex64))
    assert out.shape == (10,)
```

## Safety & Permissions

**Allowed without prompt:**
- Read files, list directories, grep search
- Run single-file type checks, ruff checks, or tests
- Format/lint individual files with ruff
- View logs, inspect variables

**Ask before:**
- Installing packages (modify `pyproject.toml`)
- Deleting files or running `git push`
- Project-wide builds (`uv build`)
- Full test suite on large changesets
- Modifying FFmpeg pipelines or external tool invocations

## PR Checklist

Before opening a PR:
1. **Ruff clean:** `uv run ruff check` and `uv run ruff format --check`
2. **Tests pass:** `uv run --with dev pytest`
3. **Type check clean:** Verify modified files with mypy
4. **Small diff:** Focus changes on one feature/fix
5. **Commit message:** Imperative, ≤50 chars (e.g., "Add LSB decoder AGC control")
6. **Description:** Explain motivation, list CLI flag changes, link related issues
7. **Screenshots:** Include before/after spectrograms for DSP changes

## When Stuck

- Ask clarifying questions rather than guessing implementation details
- Propose a short plan for complex changes
- Check existing decoder implementations (`decoders/*.py`) as templates
- Review `processing.py` pipeline structure before major refactors
- Open draft PR with questions if unsure about direction

## External Dependencies

**Required on PATH:**
- `ffmpeg` / `ffprobe`: WAV ingestion and resampling
  - Verify: `ffmpeg -version`
  - Install: macOS `brew install ffmpeg`, Linux `apt install ffmpeg`

**Python packages (managed via pyproject.toml):**
- NumPy ≥1.24, SciPy ≥1.10 (DSP core)
- PySide6 ≥6.6.0 (Qt6 GUI)
- Matplotlib ≥3.10.7 (Spectrum/waterfall plots)
- soundfile, soxr, tqdm (audio I/O, resampling, progress)

**Qt platform plugins (Linux):**
- Install `qtwayland` or `qtbase-x11` for `--interactive` mode
- Verify: `uv run python -c "from PySide6 import QtWidgets; print('OK')"`

## Architecture Notes

**Streaming DSP:** All processing uses chunked iteration—never load full recordings into memory.

**Modularity:** Decoders implement `Decoder` ABC (`setup`, `process`, `finalize`, `intermediates`). Add new modes under `decoders/` without touching `processing.py` pipeline.

**No code duplication:** DSP logic lives in `processing.py` or `decoders/*`, consumed by both `cli.py` and `interactive.py`. UI-specific code (Qt widgets, Matplotlib canvas) stays isolated in `interactive.py`.

**Squelch/trimming:** Custom NumPy implementation in `squelch.py`—no external tools. Adaptive squelch tracks noise floor with exponential smoothing; silence trimming operates on edges only.

**Progress tracking:** `ProgressSink` ABC allows CLI (tqdm) and GUI (Qt signals) to share progress updates without coupling.
