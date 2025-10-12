# Application icon assets

Icons are derived from the repository root `logo.png`. Run `tools/generate_app_icons.py`
to create the platform-specific outputs before packaging:

```bash
uv run python tools/generate_app_icons.py
```

The script will:

- build `iq_to_audio.ico` for Windows using `ffmpeg`
- build `iq_to_audio.icns` for macOS using `sips` + `iconutil` (macOS hosts only)
- leave intermediate `iq_to_audio.iconset/` artefacts that can be inspected or
  deleted later

Linux builds use the original PNG via the PyInstaller spec.
