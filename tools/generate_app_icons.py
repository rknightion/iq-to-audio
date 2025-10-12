#!/usr/bin/env python3
"""Utility script to derive platform-specific icons from the shared logo."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

ICON_ROOT = Path("packaging/icons")
LOGO = Path("logo.png")


def _ensure_logo_exists() -> None:
    if not LOGO.exists():
        raise FileNotFoundError(
            "logo.png is required to generate icons. Confirm the file exists in the repository root."
        )


def _generate_windows_icon() -> None:
    output = ICON_ROOT / "iq_to_audio.ico"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[icons] Skipping Windows icon; ffmpeg not available on PATH.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(LOGO),
        "-vf",
        "scale=256:256",
        str(output),
    ]
    print(f"[icons] Writing {output}")
    subprocess.run(cmd, check=True)


def _generate_macos_icon() -> None:
    if platform.system() != "Darwin":
        print("[icons] Skipping macOS .icns generation (non-macOS host).")
        return

    iconutil = shutil.which("iconutil")
    sips = shutil.which("sips")
    if not iconutil or not sips:
        print("[icons] Skipping macOS icon; iconutil and sips are required.")
        return

    iconset_dir = ICON_ROOT / "iq_to_audio.iconset"
    iconset_dir.mkdir(parents=True, exist_ok=True)
    for child in iconset_dir.iterdir():
        child.unlink()

    # PyInstaller requires the canonical macOS iconset names. Each pair is 1x/2x.
    entries = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    for size, filename in entries:
        target = iconset_dir / filename
        subprocess.run(
            [sips, "-z", str(size), str(size), str(LOGO), "--out", str(target)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    icns_path = ICON_ROOT / "iq_to_audio.icns"
    print(f"[icons] Writing {icns_path}")
    subprocess.run([iconutil, "-c", "icns", str(iconset_dir), "-o", str(icns_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate platform icons from logo.png")
    parser.add_argument(
        "--skip-macos",
        action="store_true",
        help="Do not generate the macOS .icns asset.",
    )
    parser.add_argument(
        "--skip-windows",
        action="store_true",
        help="Do not generate the Windows .ico asset.",
    )
    args = parser.parse_args()

    _ensure_logo_exists()
    if not args.skip_windows:
        _generate_windows_icon()
    if not args.skip_macos:
        _generate_macos_icon()


if __name__ == "__main__":
    main()
