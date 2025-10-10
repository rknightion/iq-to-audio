from __future__ import annotations

from typing import Optional

from .am import AMDecoder
from .base import Decoder
from .nfm import NarrowbandFMDecoder
from .ssb import SSBDecoder


def create_decoder(
    mode: str,
    *,
    deemph_us: float,
    squelch_dbfs: Optional[float],
    silence_trim: bool,
    squelch_enabled: bool,
    agc_enabled: bool,
) -> Decoder:
    mode = mode.lower()
    if mode in {"nfm", "fm"}:
        return NarrowbandFMDecoder(
            deemph_us=deemph_us,
            squelch_dbfs=squelch_dbfs,
            silence_trim=silence_trim,
            squelch_enabled=squelch_enabled,
        )
    if mode == "am":
        return AMDecoder(
            squelch_dbfs=squelch_dbfs,
            silence_trim=silence_trim,
            squelch_enabled=squelch_enabled,
        )
    if mode in {"usb", "ssb"}:
        return SSBDecoder(
            sideband="usb",
            squelch_dbfs=squelch_dbfs,
            silence_trim=silence_trim,
            squelch_enabled=squelch_enabled,
            agc_enabled=agc_enabled,
        )
    if mode == "lsb":
        return SSBDecoder(
            sideband="lsb",
            squelch_dbfs=squelch_dbfs,
            silence_trim=silence_trim,
            squelch_enabled=squelch_enabled,
            agc_enabled=agc_enabled,
        )
    raise ValueError(f"Unsupported demod mode '{mode}'.")


__all__ = [
    "Decoder",
    "create_decoder",
    "NarrowbandFMDecoder",
    "AMDecoder",
    "SSBDecoder",
]
