"""
Runtime adjustments executed before iq-to-audio boots inside a PyInstaller bundle.

PyInstaller loads this hook prior to running our entry point, which gives us one
central location to tweak environment variables without sprinkling platform
conditionals throughout the application code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundled_root() -> Path | None:
    """Return the root of the unpacked bundle when running frozen."""
    if not getattr(sys, "frozen", False):
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    executable = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        # macOS onedir layout: <App>.app/Contents/MacOS/iq-to-audio
        return executable.parents[1]
    return executable.parent


def _ensure_path_has(directory: Path) -> None:
    """Prepend ``directory`` to PATH if it is not already present."""
    if not directory.exists():
        return
    current = os.environ.get("PATH", "")
    components = current.split(os.pathsep) if current else []
    if str(directory) not in components:
        components.insert(0, str(directory))
        os.environ["PATH"] = os.pathsep.join(components)


bundle_root = _bundled_root()

if bundle_root:
    # Ship FFmpeg/FFprobe inside dist/ffmpeg for predictable lookup.
    _ensure_path_has(bundle_root / "ffmpeg")

    if sys.platform == "darwin":
        # Qt6 on macOS needs layer-backed rendering to draw correctly when frozen.
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

        # Ensure bundled Qt plugins take precedence over any system Qt packages.
        plugins = bundle_root / "PySide6" / "plugins"
        if plugins.exists():
            existing = os.environ.get("QT_PLUGIN_PATH")
            paths = [str(plugins)]
            if existing:
                paths.append(existing)
            os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(paths)
