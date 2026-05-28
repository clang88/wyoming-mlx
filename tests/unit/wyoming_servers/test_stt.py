import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Info

from wyoming_mlx.backends.fake import FakeSTTBackend
from wyoming_mlx.wyoming_servers.stt import SttEventHandler


def _pcm_chunk(n_samples: int = 1600) -> bytes:
    """A silent 16 kHz mono 16-bit PCM block."""
    return b"\x00\x00" * n_samples


class _CaptureWriter:
    """Captures events written by the handler via write_event override."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def capture(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_stt_handler_emits_transcript_after_audio_stop():
    backend = FakeSTTBackend(transcript="the quick brown fox")
    capture = _CaptureWriter()
    handler = SttEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=AsyncMock(),
        backend=backend,
        info=Info(),
    )
    # Override write_event to capture events
    handler.write_event = AsyncMock(side_effect=lambda e: capture.capture(e))

    assert await handler.handle_event(
        AudioStart(rate=16000, width=2, channels=1).event()
    )
    assert await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=_pcm_chunk()).event()
    )
    assert await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=_pcm_chunk()).event()
    )
    assert await handler.handle_event(AudioStop().event())

    transcripts = [Transcript.from_event(e) for e in capture.events if Transcript.is_type(e.type)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "the quick brown fox"


@pytest.mark.asyncio
async def test_stt_handler_passes_concatenated_audio_to_backend():
    backend = FakeSTTBackend(transcript="x")
    handler = SttEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=AsyncMock(),
        backend=backend,
        info=Info(),
    )

    await handler.handle_event(
        AudioStart(rate=16000, width=2, channels=1).event()
    )
    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x01\x02").event()
    )
    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x03\x04").event()
    )
    await handler.handle_event(AudioStop().event())

    assert len(backend.calls) == 1
    audio, rate = backend.calls[0]
    assert audio == b"\x01\x02\x03\x04"
    assert rate == 16000


@pytest.mark.asyncio
async def test_stt_handler_responds_to_describe():
    from wyoming.info import Describe

    backend = FakeSTTBackend(transcript="hello")
    capture = _CaptureWriter()
    handler = SttEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=AsyncMock(),
        backend=backend,
        info=Info(),
    )
    handler.write_event = AsyncMock(side_effect=lambda e: capture.capture(e))

    assert await handler.handle_event(Describe().event())

    info_events = [e for e in capture.events if Info.is_type(e.type)]
    assert len(info_events) == 1
    result_info = Info.from_event(info_events[0])
    assert isinstance(result_info, Info)
