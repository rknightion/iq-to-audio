"""Wrapper around PyInstaller to harmonise CLI use with spec files.

PyInstaller's documentation recommends setting ``target_arch`` inside the
spec file for macOS universal builds. When a spec file is provided, the
``--target-arch`` CLI option is rejected. Our GitHub Actions workflow still
passes the flag, so we intercept it here, strip the unsupported options, and
then delegate to PyInstaller's real entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.__main__ import _console_script_run


_TARGET_ARCH_FLAGS = {"--target-arch", "--target-architecture"}


def _sanitise_argv(argv: list[str]) -> list[str]:
    if len(argv) <= 1:
        return argv

    program = Path(argv[0]).name.lower()
    if program not in {"pyinstaller", "pyinstaller.exe"}:
        return argv

    args = argv[1:]
    has_spec = any(arg.endswith(".spec") for arg in args if not arg.startswith("-"))
    if not has_spec:
        return argv

    sanitised: list[str] = [argv[0]]
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        lowered = arg.lower()
        if lowered in _TARGET_ARCH_FLAGS:
            skip_next = True
            continue
        if any(lowered.startswith(flag + "=") for flag in _TARGET_ARCH_FLAGS):
            continue
        sanitised.append(arg)
    return sanitised


def main() -> None:
    sys.argv = _sanitise_argv(sys.argv)
    sys.exit(_console_script_run())
