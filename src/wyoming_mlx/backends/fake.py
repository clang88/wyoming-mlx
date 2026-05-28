from __future__ import annotations

from collections.abc import AsyncIterator


class FakeSTTBackend:
    """In-memory STT backend for unit tests.

    Returns a canned transcript and records every call it receives.
    """

    def __init__(self, transcript: str = "") -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, int]] = []

    async def transcribe(self, audio: bytes, sample_rate: int) -> str:
        self.calls.append((audio, sample_rate))
        return self.transcript


class FakeTTSBackend:
    """In-memory TTS backend for unit tests.

    Yields a fixed list of chunks for every synthesize call.
    """

    def __init__(
        self,
        chunks: list[bytes],
        voices: list[str] | None = None,
        sample_rate: int = 24000,
    ) -> None:
        self._chunks = list(chunks)
        self.voices = voices or ["default"]
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]:
        self.calls.append((text, voice))
        for chunk in self._chunks:
            yield chunk
