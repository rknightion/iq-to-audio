"""
Project-level sitecustomize to smooth out PyInstaller CLI quirks.

GitHub Actions invokes ``pyinstaller iq-to-audio.spec --target-arch universal2``.
PyInstaller rejects ``--target-arch`` when a spec file is supplied, so we strip
that option proactively before PyInstaller's argument parser runs.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _sanitize_pyinstaller_args() -> None:
    argv = sys.argv
    if not argv:
        return

    program = Path(argv[0]).name.lower()
    if program not in {"pyinstaller", "pyinstaller.exe"}:
        return

    args = argv[1:]
    has_spec = any(arg.endswith(".spec") for arg in args if not arg.startswith("-"))
    if not has_spec:
        return

    sanitized: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        option = arg.lower()
        if option in {"--target-arch", "--target-architecture"}:
            skip_next = True
            continue
        if option.startswith("--target-arch=") or option.startswith("--target-architecture="):
            continue
        sanitized.append(arg)

    argv[:] = [argv[0], *sanitized]


_sanitize_pyinstaller_args()
