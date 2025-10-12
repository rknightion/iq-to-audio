# Packaging with PyInstaller

This guide documents the new PyInstaller-based distribution workflow for
`iq-to-audio`. It covers local builds, CI automation, and the scaffolding that
was added for code signing and FFmpeg bundling.

## Overview

- PyInstaller runs out of the existing `uv` environment (`uv run pyinstaller`).
- The spec file lives at `iq-to-audio.spec` and targets an onedir layout so the
  Qt/FFmpeg payload remains inspectable.
- FFmpeg and FFprobe are bundled from the builder's PATH (or the
  `IQ_TO_AUDIO_BUILD_FFMPEG`/`IQ_TO_AUDIO_BUILD_FFPROBE` overrides) and copied
  into `dist/ffmpeg/`. The app resolves them at runtime via
  `utils.resolve_ffmpeg_executable()` / `resolve_ffprobe_executable()`.
- macOS builds request `universal2` binaries per the PyInstaller feature notes so
  the resulting `.app` ships both arm64 and x86_64 slices.
- The repository exposes a thin wrapper around PyInstaller (installed as the
  `pyinstaller` console script) that strips `--target-arch` when a spec file is
  provided. This matches PyInstaller's guidance for universal builds: the spec
  file sets `target_arch="universal2"`, so the CLI flag would otherwise cause an
  error. No workflow changes are required.
- `packaging/pyinstaller/runtime_environment.py` adds minor runtime polish:
  prepends `dist/ffmpeg` to `PATH`, enables layer-backed Qt rendering on macOS,
  and ensures Qt plugins resolve from the bundle.

## Icon assets

Our source artwork remains `logo.png` in the repository root. Generate platform
variants with the helper script:

```bash
uv run python tools/generate_app_icons.py
```

- Always run this on macOS before packaging to refresh `iq_to_audio.icns`.
- Windows/Linux builds reuse the generated `iq_to_audio.ico` and the original PNG
  (Linux does not require a dedicated icon format).

The script is idempotent and will skip steps if tooling is missing (e.g. `iconutil`
when not on macOS).

## Local builds

Typical local packaging loop:

```bash
uv run python tools/generate_app_icons.py           # ensures icons exist
uv run pyinstaller iq-to-audio.spec                 # produces dist/iq-to-audio*
```

Environment variables you can use to customise the build:

- `IQ_TO_AUDIO_BUILD_FFMPEG` / `IQ_TO_AUDIO_BUILD_FFPROBE` – absolute paths to
  the binaries you want bundled (useful when PATH exposes wrappers).
- `IQ_TO_AUDIO_CODESIGN_IDENTITY` – codesign identity passed directly to
  PyInstaller when building on macOS.
- `IQ_TO_AUDIO_CODESIGN_ENTITLEMENTS` – optional override pointing at a custom
  entitlements plist; defaults to `packaging/pyinstaller/macos-entitlements.plist`.

On macOS the spec requests `--target-arch universal2`. Ensure your interpreter is
also universal (the python.org builds shipped on GitHub Actions already are).
If you need to validate the result, `lipo -info dist/iq-to-audio.app/Contents/MacOS/iq-to-audio`
should report both `arm64` and `x86_64`.

## Continuous integration

`.github/workflows/build-release.yml` automates the packaging story:

1. Runs on tag pushes (`v*`) or manual dispatch.
2. Installs Python/uv, provisions FFmpeg via `FedericoCarboni/setup-ffmpeg`, and
   syncs the project with `uv sync --dev`.
3. Calls `tools/generate_app_icons.py` so the required icons exist on each
   runner.
4. Invokes PyInstaller. The macOS job passes `--target-arch universal2` so the
   uploaded `.app` is multi-arch. Windows/Linux run the default build.
5. Archives the results (`.tar.gz` for Linux, `.zip` for macOS/Windows) and
   uploads them as workflow artifacts.

Because GitHub cannot cross-compile, each platform builds natively. The macOS job
runs on `macos-14` (arm64) which satisfies the universal build requirements from
<https://pyinstaller.org/en/stable/feature-notes.html#macos-multi-arch-support>.

## Code signing scaffolding

We do not sign by default, but the plumbing is ready:

- macOS: set the `MACOS_CODESIGN_IDENTITY` repository secret (e.g.
  `Developer ID Application: ...`). The workflow exports it to
  `IQ_TO_AUDIO_CODESIGN_IDENTITY` for PyInstaller. Update
  `packaging/pyinstaller/macos-entitlements.plist` as you finalise sandbox
  requirements. When you're ready to notarise, follow the recipe in
  <https://gist.github.com/txoof/0636835d3cc65245c6288b2374799c43>.
- Windows: plan to integrate `signtool` once a certificate is available. The
  spec intentionally bundles FFmpeg inside `dist/ffmpeg` so you can sign the
  entire directory tree with a post-build step.
- Linux: consider `osslsigncode` or detached signatures if downstream
  distributions require them.

The documentation above should help you wire the remaining secrets without
modifying the build logic.
