from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass
class STTUpdate:
    """Incremental transcription update.

    `text` is newly confirmed text (never retracted by later updates).
    `final` is the full transcript, set only on the last update of a session.
    """

    text: str = ""
    final: str | None = None


@runtime_checkable
class STTSession(Protocol):
    """One utterance's streaming transcription session."""

    async def feed(self, audio: bytes, sample_rate: int) -> None: ...
    async def finish(self) -> None: ...
    async def close(self) -> None: ...
    def updates(self) -> AsyncIterator[STTUpdate]: ...


@runtime_checkable
class STTBackend(Protocol):
    """Speech-to-text backend.

    Implementations must support multiple concurrent sessions; a real
    implementation may serialise GPU work internally.
    """

    def start_session(
        self,
        *,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
    ) -> STTSession: ...


async def collect_transcript(
    backend: STTBackend,
    audio: bytes,
    sample_rate: int,
    *,
    language: str | None = None,
    prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    """Run one complete utterance through a session and return the final text.

    `language`/`prompt`/`temperature` are per-request overrides (e.g. from the
    OpenAI-compatible HTTP API); backends that can't honour one silently
    ignore it and fall back to their configured default.
    """
    session = backend.start_session(language=language, prompt=prompt, temperature=temperature)
    try:
        await session.feed(audio, sample_rate)
        await session.finish()
        final = ""
        async for update in session.updates():
            if update.final is not None:
                final = update.final
        return final
    finally:
        try:
            await session.close()
        except Exception:
            log.warning("session.close() failed; suppressing to preserve primary error")


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

    def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]: ...
