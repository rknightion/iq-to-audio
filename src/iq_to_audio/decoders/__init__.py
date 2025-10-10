from __future__ import annotations

from .am import AMDecoder
from .base import Decoder
from .nfm import NarrowbandFMDecoder
from .ssb import SSBDecoder


def create_decoder(
    mode: str,
    *,
    deemph_us: float,
    agc_enabled: bool,
) -> Decoder:
    mode = mode.lower()
    if mode in {"nfm", "fm"}:
        return NarrowbandFMDecoder(deemph_us=deemph_us)
    if mode == "am":
        return AMDecoder()
    if mode in {"usb", "ssb"}:
        return SSBDecoder(sideband="usb", agc_enabled=agc_enabled)
    if mode == "lsb":
        return SSBDecoder(sideband="lsb", agc_enabled=agc_enabled)
    raise ValueError(f"Unsupported demod mode '{mode}'.")


__all__ = [
    "Decoder",
    "create_decoder",
    "NarrowbandFMDecoder",
    "AMDecoder",
    "SSBDecoder",
]
