# PyInstaller Packaging for iq-to-audio

This directory contains PyInstaller configuration and support files for building standalone executables of iq-to-audio.

## Entry Point Architecture

### `run_cli.py` - Unified Entry Point

The PyInstaller bundle uses a smart entry point that automatically detects how the application is launched:

**Behavior:**
- **No arguments (double-click)** → Launches GUI with default settings
- **With arguments (terminal)** → Runs CLI with full argument parsing

**Why this design?**
- Most users will double-click the app → instant GUI
- Power users can still use terminal for batch processing
- No need for separate GUI/CLI executables
- Qt libraries only load when needed (saves memory in CLI mode)

### Example Usage

```bash
# Double-click the app or run without args → GUI mode
./iq-to-audio

# Run with arguments → CLI mode
./iq-to-audio input.wav --target-freq 100000000 --output out.wav
./iq-to-audio --help
./iq-to-audio input.wav --interactive  # Still works!
```

## Import Resolution

The entry point ensures all imports work correctly by:
1. Adding the source directory to `sys.path`
2. Importing `iq_to_audio` as a proper package (preserves relative imports)
3. Detecting PyInstaller's temp directory (`sys._MEIPASS`) when frozen

This fixes the `ImportError: attempted relative import with no known parent package` issue.

## Console Behavior

**Setting:** `console=False` in spec file

### Windows
- ✅ Double-click: Clean GUI launch (no console window)
- ✅ Terminal: Output appears in the existing cmd/PowerShell window
- ℹ️  Does not create a NEW console window (this is desired)

### macOS
- ✅ Double-click `.app`: Launches GUI via Finder
- ✅ Terminal: `./iq-to-audio.app/Contents/MacOS/iq-to-audio` works normally

### Linux
- ✅ Desktop: Launches GUI when double-clicked
- ✅ Terminal: Normal CLI behavior

## Build Process

The spec file (`../../iq-to-audio.spec`) handles:
- Platform-specific icon generation
- FFmpeg binary bundling
- Qt/PySide6 data collection
- Code signing and notarization (macOS)
- Platform-specific architecture targeting

### Building

```bash
# Standard build (auto-detects architecture)
uv run pyinstaller iq-to-audio.spec

# macOS Intel (x86_64)
IQ_TO_AUDIO_TARGET_ARCH=x86_64 uv run pyinstaller iq-to-audio.spec

# macOS Apple Silicon (arm64)
IQ_TO_AUDIO_TARGET_ARCH=arm64 uv run pyinstaller iq-to-audio.spec
```

## Runtime Hooks

### `runtime_environment.py`
Sets up the runtime environment for the PyInstaller bundle:
- Configures FFmpeg paths
- Sets Qt platform plugins
- Handles platform-specific initialization

## Hidden Imports

The spec file automatically collects:
- All `iq_to_audio` submodules
- PySide6 components
- Matplotlib backends
- NumPy, SciPy, soundfile, soxr

This ensures all dependencies are bundled correctly.

## Troubleshooting

### Import Errors
**Issue:** `ImportError: attempted relative import with no known parent package`

**Fix:** Ensure `run_cli.py` is the entry point (already configured in spec file)

### Missing Qt Plugins
**Issue:** `Could not load the Qt platform plugin`

**Fix:** PySide6 data files are collected automatically. Check that `collect_data_files("PySide6")` is in the spec.

### FFmpeg Not Found
**Issue:** `ffmpeg: command not found` or similar

**Fix:**
1. Ensure FFmpeg is installed during build: `uv sync --dev`
2. Verify `_ffmpeg_binaries()` finds ffmpeg/ffprobe
3. Check environment variables: `IQ_TO_AUDIO_BUILD_FFMPEG`, `IQ_TO_AUDIO_BUILD_FFPROBE`

### GUI Doesn't Launch on Double-Click
**Issue:** Nothing happens when double-clicking

**Fix:**
1. Check that `console=False` in spec file
2. Ensure `launch_interactive_session` is imported correctly in `run_cli.py`
3. Check platform-specific logs:
   - Windows: Event Viewer → Application logs
   - macOS: Console.app → search for "iq-to-audio"
   - Linux: Run from terminal to see errors

### CLI Arguments Not Working
**Issue:** Arguments are ignored or cause errors

**Fix:**
1. Ensure you're passing arguments: `./iq-to-audio --help`
2. Check that `len(sys.argv) > 1` condition works
3. Verify `from iq_to_audio.cli import main` succeeds

## Platform-Specific Notes

### macOS
- **App Bundle:** Creates `iq-to-audio.app` (can be dragged to Applications)
- **CLI Access:** `./iq-to-audio.app/Contents/MacOS/iq-to-audio --help`
- **Code Signing:** Set `IQ_TO_AUDIO_CODESIGN_IDENTITY` environment variable
- **Notarization:** Requires Apple Developer account and app-specific password

### Windows
- **Executable:** Creates `iq-to-audio.exe` in `dist/iq-to-audio/`
- **DLL Dependencies:** All bundled automatically
- **Antivirus:** May require code signing to avoid false positives

### Linux
- **Binary:** Creates `iq-to-audio` in `dist/iq-to-audio/`
- **Dependencies:** Qt platform plugins included
- **Desktop Integration:** Can create `.desktop` file for menu integration

## Files in this Directory

- `run_cli.py` - Main entry point with GUI/CLI detection
- `runtime_environment.py` - Runtime setup and configuration
- `hooks/` - Custom PyInstaller hooks (if any)
- `README.md` - This file

## See Also

- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [PySide6 + PyInstaller Guide](https://doc.qt.io/qtforpython-6/deployment/deployment-pyinstaller.html)
- Main project README: `../../README.md`