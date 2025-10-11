# Repository Guidelines

## Project Structure & Module Organization
Core source lives in `src/iq_to_audio`; `cli.py` wires argument parsing to the DSP pipeline, while `processing.py` houses channelization and demodulation, and `visualize.py` plus `interactive.py` handle an optional interactive mode. Utilities such as filename parsing reside in `utils.py`, and sample-rate probing sits in `probe.py`. Tests mirror the package layout under `tests/`—keep new modules paired with `tests/test_<module>.py` to preserve coverage symmetry.

## Build, Test, and Development Commands
Use `uv run iq-to-audio --help` to confirm the CLI installs and prints usage without creating a persistent environment. For iterative work, bootstrap a venv (`uv venv && source .venv/bin/activate`) then install in editable mode with `uv pip install -e .`. Execute end-to-end demodulation locally via `uv run iq-to-audio --in <wav> --ft <Hz>`, and run the full test suite with `uv run --with dev pytest`. When packaging or validating metadata, `uv build` produces source and wheel distributions.

Interactive UX relies on PySide6. Helpful commands during development:
- `uv run iq-to-audio --interactive`: launches the Qt-based GUI.
- `uv pip install -e ".[dev]"`: install the package for hacking (also pulls in PySide6).
- `uv run python -c "from PySide6 import QtWidgets; print(QtWidgets.__version__)"`: verify the Qt bindings are importable.

Functionality between interactive and CLI modes should be comaprable with the exception of visualisation/interactive functionality. All DSP and processing and utils must live in a shared file consumed by interactive.py or cli.py respectively to avoid code duplication.

Always use 'uv' for python environment management, dependency management and running commands/scripts

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents and snake_case for modules, functions, and locals (`mix_sign_override`, `parse_center_frequency`). Prefer dataclasses for configurations (see `ProcessingConfig`) and annotate public APIs with typing hints. Use `Path` objects for filesystem paths, and keep module-level loggers named `LOG = logging.getLogger(__name__)`. Maintain NumPy/SciPy vectorized operations and avoid per-sample loops unless profiled.

## Testing 
Pytest drives validation; place new tests under `tests/` with files beginning `test_` and functions named `test_<behavior>()`. Include fixtures that synthesize deterministic IQ arrays where possible, avoiding large binary assets. Run `uv run --with dev pytest -k <keyword>` for focused debugging, and ensure new DSP paths exercise both demodulated audio and auxiliary outputs (e.g., `--dump-iq`).

## Commit & Pull Request Guidelines
Commit messages should be imperative and concise (current history: “initial add”), summarizing the behavior change in ≤50 characters and elaborating in the body when necessary. Each PR should describe the motivation, highlight user-facing CLI flags or defaults that changed, list test commands executed, and link related issues or recordings. Include before/after spectrogram screenshots when altering signal processing stages to document the impact.

## Environment & External Tools
FFmpeg must be available on `PATH`; confirm with `ffmpeg -version` before running the pipeline. The interactive spectrum viewer depends on PySide6/Qt plus Matplotlib—install suitable Qt platform plugins when working on headless servers (e.g., `qtwayland` or `qtbase-x11`). Large SDR recordings can exceed RAM, so keep local scratch space on fast storage and prefer streaming workflows rather than copying files into the repo.
