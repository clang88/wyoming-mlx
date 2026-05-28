from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class STTBackend(Protocol):
    """Speech-to-text backend.

    Implementations must be safe to call concurrently from multiple asyncio
    tasks. A real implementation may serialise GPU work internally.
    """

    async def transcribe(self, audio: bytes, sample_rate: int) -> str:
        ...


@runtime_checkable
class TTSBackend(Protocol):
    """Text-to-speech backend.

    `synthesize` returns an async iterator of raw PCM frames (mono, int16,
    little-endian) at the backend's native sample rate. The sample rate is
    exposed via the `sample_rate` attribute so callers (Wyoming, HTTP) can
    advertise or wrap it appropriately.
    """

    voices: list[str]
    sample_rate: int

    def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]:
        ...
