from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_FC_PATTERN = re.compile(r"_(\d+)Hz_")


def parse_center_frequency(path: Path) -> Optional[float]:
    """Extract center frequency in Hz from an SDR++ filename.

    SDR++ baseband filenames typically look like:
        baseband_456834049Hz_14-17-29_09-10-2025.wav
    This helper returns 456834049.0 for that example.
    """
    match = _FC_PATTERN.search(path.name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
