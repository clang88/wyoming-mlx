from __future__ import annotations

import logging
from typing import Protocol

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

from wyoming_mlx.backends.base import STTBackend

log = logging.getLogger(__name__)


class _Writer(Protocol):
    async def write_event(self, event: Event) -> None: ...


class SttEventHandler:
    """Wyoming event handler for one STT client connection.

    Buffers audio between AudioStart and AudioStop, then transcribes.
    Stateless across utterances apart from `_buffer` / `_sample_rate`.
    """

    def __init__(self, backend: STTBackend, writer: _Writer) -> None:
        self._backend = backend
        self._writer = writer
        self._buffer = bytearray()
        self._sample_rate: int | None = None

    async def handle_event(self, event: Event) -> bool:
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
            text = await self._backend.transcribe(bytes(self._buffer), self._sample_rate)
            await self._writer.write_event(Transcript(text=text).event())
            self._buffer.clear()
            self._sample_rate = None
            return True

        return True
