# iq-to-audio.spec
# -*- mode: python ; coding: utf-8 -*-
"""
Production PyInstaller build recipe for the iq-to-audio GUI.

Key decisions documented inline so the packaging flow is easy to audit:

* We ship an onedir build so FFmpeg binaries and Qt plugins remain inspectable.
* FFmpeg and FFprobe are mandatory; we lift them from PATH (or explicit env
  overrides) and drop them under dist/ffmpeg for deterministic runtime lookup.
* PySide6 data files are collected explicitly to ensure the embedded Qt stack
  is complete on all three desktop platforms.
* macOS builds target universal2 so a single .app works on both Intel and Apple
  Silicon. GitHub Actions assembles the universal binary via the documented
  two-step process (per PyInstaller feature notes).
* Codesigning is wired but optional: set IQ_TO_AUDIO_CODESIGN_IDENTITY and/or
  IQ_TO_AUDIO_CODESIGN_ENTITLEMENTS before invoking PyInstaller to enable it.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

APP_NAME = "iq-to-audio"
BUNDLE_IDENTIFIER = "com.rknightion.iq-to-audio"

_SPEC_PATH = Path(globals().get("__file__", Path.cwd() / "iq-to-audio.spec"))
if _SPEC_PATH.is_file():
    PROJECT_ROOT = _SPEC_PATH.resolve().parent
else:
    PROJECT_ROOT = Path.cwd()

SRC_ROOT = PROJECT_ROOT / "src"
PACKAGING_ROOT = PROJECT_ROOT / "packaging"
PYINSTALLER_ROOT = PACKAGING_ROOT / "pyinstaller"
ICON_ROOT = PACKAGING_ROOT / "icons"

with (PROJECT_ROOT / "pyproject.toml").open("rb") as fp:
    PROJECT_VERSION = tomllib.load(fp)["project"]["version"]


def _runtime_hook_paths() -> list[str]:
    hook = PYINSTALLER_ROOT / "runtime_environment.py"
    return [str(hook)] if hook.exists() else []


def _optional_icon(filename: str) -> str | None:
    path = ICON_ROOT / filename
    if path.exists():
        return str(path)
    print(f"[pyinstaller] Icon asset missing: {path} – packaging will continue without it.")
    return None


def _require_host_tool(name: str, env_var: str) -> Path:
    """Resolve a build-time tool from an override or PATH."""
    override = os.environ.get(env_var)
    if override:
        resolved = Path(override).expanduser().resolve()
        if resolved.exists():
            return resolved
        raise SystemExit(
            f"{env_var} was provided ({override}) but the path does not exist."
        )
    discovered = shutil.which(name)
    if not discovered:
        raise SystemExit(
            f"Missing required tool '{name}'. Ensure it is on PATH or set {env_var}."
        )
    return Path(discovered).resolve()


def _ffmpeg_binaries() -> list[tuple[str, str]]:
    """Bundle FFmpeg and FFprobe beside the executable for runtime lookup."""
    staging_dir = Path("ffmpeg")
    binaries: list[tuple[str, str]] = []
    tools = {
        "ffmpeg": "IQ_TO_AUDIO_BUILD_FFMPEG",
        "ffprobe": "IQ_TO_AUDIO_BUILD_FFPROBE",
    }
    for tool, env_var in tools.items():
        source = _require_host_tool(tool, env_var)
        target = staging_dir / source.name
        binaries.append((str(source), str(target)))
    return binaries


SYSTEM = platform.system()
TARGET_ARCH = os.environ.get("IQ_TO_AUDIO_TARGET_ARCH") if SYSTEM == "Darwin" else None

WINDOWS_ICON = _optional_icon("iq_to_audio.ico")
MAC_ICON = _optional_icon("iq_to_audio.icns")
LINUX_ICON = str(PROJECT_ROOT / "logo.png") if (PROJECT_ROOT / "logo.png").exists() else None

MAC_ENTITLEMENTS_ENV = os.environ.get("IQ_TO_AUDIO_CODESIGN_ENTITLEMENTS")
MAC_ENTITLEMENTS = (
    str(Path(MAC_ENTITLEMENTS_ENV).expanduser().resolve())
    if MAC_ENTITLEMENTS_ENV
    else str(PYINSTALLER_ROOT / "macos-entitlements.plist")
    if (PYINSTALLER_ROOT / "macos-entitlements.plist").exists()
    else None
)

PYTHON_SOURCES = [str(PYINSTALLER_ROOT / "run_cli.py")]

if SYSTEM == "Darwin":
    # PySide6 frameworks already arrive via the official hook; re-collecting them
    # as data files trashes the macOS symlink layout inside *.framework bundles and
    # triggers duplicate symlink errors during COLLECT. Skip framework payloads and
    # let PyInstaller handle them via the binary path instead.
    datas = collect_data_files(
        "PySide6",
        excludes=[
            "Qt/lib/*.framework",
            "Qt/lib/*.framework/**",
        ],
    )
else:
    datas = collect_data_files("PySide6")
hiddenimports = sorted(
    set(
        collect_submodules("PySide6")
        + collect_submodules("matplotlib")
        + collect_submodules("iq_to_audio")
        + [
            "numpy",
            "scipy",
            "soundfile",
            "soxr",
        ]
    )
)

a = Analysis(
    PYTHON_SOURCES,
    pathex=[str(SRC_ROOT)],
    binaries=_ffmpeg_binaries(),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PYINSTALLER_ROOT / "hooks")],
    hooksconfig={},
    runtime_hooks=_runtime_hook_paths(),
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=TARGET_ARCH,
    codesign_identity=os.environ.get("IQ_TO_AUDIO_CODESIGN_IDENTITY"),
    entitlements_file=MAC_ENTITLEMENTS,
    icon=WINDOWS_ICON if SYSTEM == "Windows" else LINUX_ICON,
    argv_emulation=(SYSTEM == "Darwin"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

if SYSTEM == "Darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=MAC_ICON,
        bundle_identifier=BUNDLE_IDENTIFIER,
        version=PROJECT_VERSION,
        info_plist={
            "CFBundleDisplayName": "IQ to Audio",
            "CFBundleName": "IQ to Audio",
            "CFBundleShortVersionString": PROJECT_VERSION,
            "CFBundleVersion": PROJECT_VERSION,
            "CFBundlePackageType": "APPL",
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
