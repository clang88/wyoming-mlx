from __future__ import annotations

import asyncio
import logging

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from wyoming_mlx.backends.base import STTBackend

log = logging.getLogger(__name__)


class SttEventHandler(AsyncEventHandler):
    """Wyoming event handler for one STT client connection.

    Buffers audio between AudioStart and AudioStop, then transcribes.
    Stateless across utterances apart from `_buffer` / `_sample_rate`.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        backend: STTBackend,
        info: Info,
    ) -> None:
        super().__init__(reader=reader, writer=writer)
        self._backend = backend
        self._info = info
        self._buffer = bytearray()
        self._sample_rate: int | None = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._buffer.clear()
            self._sample_rate = start.rate
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._buffer.extend(chunk.audio)
            if self._sample_rate is None:
                self._sample_rate = chunk.rate
            return True

        if AudioStop.is_type(event.type):
            if self._sample_rate is None:
                log.warning("audio-stop received without audio-start; ignoring")
                return True
            try:
                text = await self._backend.transcribe(bytes(self._buffer), self._sample_rate)
            except Exception:
                log.exception("transcription failed")
                self._buffer.clear()
                self._sample_rate = None
                return True
            await self.write_event(Transcript(text=text).event())
            self._buffer.clear()
            self._sample_rate = None
            return True

        return True
