from __future__ import annotations

import asyncio
import logging

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

from wyoming_mlx.backends.base import TTSBackend

log = logging.getLogger(__name__)

# Kokoro emits mono 16-bit PCM.
_AUDIO_WIDTH = 2
_AUDIO_CHANNELS = 1


class TtsEventHandler(AsyncEventHandler):
    """Wyoming event handler for one TTS client connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        backend: TTSBackend,
        default_voice: str,
        info: Info,
    ) -> None:
        super().__init__(reader=reader, writer=writer)
        self._backend = backend
        self._default_voice = default_voice
        self._info = info

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if not Synthesize.is_type(event.type):
            return True

        synth = Synthesize.from_event(event)
        text = synth.text or ""
        requested = getattr(synth.voice, "name", None) if synth.voice else None
        voice = requested or self._default_voice

        rate = self._backend.sample_rate
        await self.write_event(
            AudioStart(rate=rate, width=_AUDIO_WIDTH, channels=_AUDIO_CHANNELS).event()
        )
        async for chunk in self._backend.synthesize(text, voice):
            await self.write_event(
                AudioChunk(
                    rate=rate,
                    width=_AUDIO_WIDTH,
                    channels=_AUDIO_CHANNELS,
                    audio=chunk,
                ).event()
            )
        await self.write_event(AudioStop().event())
        return True
