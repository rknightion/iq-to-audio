#!/usr/bin/env python3
"""
PyInstaller entry point for iq-to-audio.

This wrapper ensures modules are imported as part of the iq_to_audio package,
preserving relative imports and package structure.

Behavior:
- No arguments: Launch GUI (double-click behavior)
- With arguments: Run CLI (terminal usage)
"""

import sys
from pathlib import Path

# Ensure the src directory is on the path so iq_to_audio can be imported as a package
if getattr(sys, 'frozen', False):
    # Running in PyInstaller bundle
    application_path = Path(sys._MEIPASS)
else:
    # Running in development
    application_path = Path(__file__).parent.parent.parent / 'src'

sys.path.insert(0, str(application_path))

if __name__ == '__main__':
    # Check if any arguments were provided (beyond the executable name)
    if len(sys.argv) > 1:
        # Arguments provided -> CLI mode
        from iq_to_audio.cli import main
        sys.exit(main())
    else:
        # No arguments -> GUI mode (double-click behavior)
        # Import here to avoid loading Qt unnecessarily in CLI mode
        from iq_to_audio.interactive import launch_interactive_session

        # Launch GUI with no input file and default settings
        session = launch_interactive_session(
            input_path=None,
            base_kwargs={
                'demod_mode': 'nfm',
                'agc_enabled': True,
                'bandwidth': 12_500.0,
                'deemph_us': 50.0,
                'audio_sr': 48_000,
                'squelch_db': None,
                'trim_silence_ms': None,
            },
            snapshot_seconds=2.0,
        )

        # If user configured outputs in GUI, they were already processed
        # Exit successfully
        sys.exit(0)
