from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .docker_backend import default_decoder_command


@dataclass(slots=True, frozen=True)
class DigitalDecoder:
    """Metadata for one decoder routing target."""

    key: str
    label: str
    description: str
    default_command: tuple[str, ...]


def _build_decoders() -> tuple[DigitalDecoder, ...]:
    return (
        DigitalDecoder(
            key="dsd-fme",
            label="DSD-FME — Digital voice decoding",
            description="Configure piping voice channels to the DSD-FME toolkit.",
            default_command=default_decoder_command(("dsd-fme", "--help")),
        ),
        DigitalDecoder(
            key="multimon-ng",
            label="Multimon-NG — Packet data decoding",
            description="Batch process paging/data bursts via multimon-ng.",
            default_command=default_decoder_command(("multimon-ng", "--help")),
        ),
        DigitalDecoder(
            key="ft8",
            label="FT8/FT4 — Weak signal workflows",
            description="Stage audio bursts for FT8 or FT4 decoders.",
            default_command=default_decoder_command(("ft8", "--help")),
        ),
    )


DIGITAL_DECODERS: tuple[DigitalDecoder, ...] = _build_decoders()
DIGITAL_DECODER_MAP: dict[str, DigitalDecoder] = {
    decoder.key: decoder for decoder in DIGITAL_DECODERS
}
DEFAULT_DECODER_KEY = DIGITAL_DECODERS[0].key if DIGITAL_DECODERS else "dsd-fme"


def iter_decoders() -> Iterable[DigitalDecoder]:
    return DIGITAL_DECODERS


def get_decoder(key: str) -> DigitalDecoder:
    try:
        return DIGITAL_DECODER_MAP[key]
    except KeyError as exc:
        raise KeyError(f"Unknown decoder key: {key}") from exc
