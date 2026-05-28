from __future__ import annotations

import logging
from typing import Protocol

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.tts import Synthesize

from wyoming_mlx.backends.base import TTSBackend

log = logging.getLogger(__name__)

# Kokoro emits mono 16-bit PCM.
_AUDIO_WIDTH = 2
_AUDIO_CHANNELS = 1


class _Writer(Protocol):
    async def write_event(self, event: Event) -> None: ...


class TtsEventHandler:
    """Wyoming event handler for one TTS client connection."""

    def __init__(
        self,
        backend: TTSBackend,
        writer: _Writer,
        default_voice: str,
    ) -> None:
        self._backend = backend
        self._writer = writer
        self._default_voice = default_voice

    async def handle_event(self, event: Event) -> bool:
        if not Synthesize.is_type(event.type):
            return True

        synth = Synthesize.from_event(event)
        text = synth.text or ""
        requested = getattr(synth.voice, "name", None) if synth.voice else None
        voice = requested or self._default_voice

        rate = self._backend.sample_rate
        await self._writer.write_event(
            AudioStart(rate=rate, width=_AUDIO_WIDTH, channels=_AUDIO_CHANNELS).event()
        )
        async for chunk in self._backend.synthesize(text, voice):
            await self._writer.write_event(
                AudioChunk(
                    rate=rate,
                    width=_AUDIO_WIDTH,
                    channels=_AUDIO_CHANNELS,
                    audio=chunk,
                ).event()
            )
        await self._writer.write_event(AudioStop().event())
        return True
