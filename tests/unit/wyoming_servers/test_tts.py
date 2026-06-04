import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Info
from wyoming.tts import Synthesize

from wyoming_mlx.backends.fake import FakeTTSBackend
from wyoming_mlx.wyoming_servers.tts import TtsEventHandler


class _CaptureWriter:
    """Captures events written by the handler via write_event override."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def capture(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_tts_handler_emits_audio_envelope():
    backend = FakeTTSBackend(chunks=[b"\x01\x02", b"\x03\x04"], voices=["v1"], sample_rate=24000)
    capture = _CaptureWriter()
    handler = TtsEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=MagicMock(spec=asyncio.StreamWriter),
        backend=backend,
        default_voice="v1",
        info=Info(),
    )
    handler.write_event = AsyncMock(side_effect=lambda e: capture.capture(e))

    await handler.handle_event(Synthesize(text="hello", voice=None).event())

    assert capture.events, "no events emitted"
    assert AudioStart.is_type(capture.events[0].type)
    assert AudioStop.is_type(capture.events[-1].type)
    chunk_events = [AudioChunk.from_event(e) for e in capture.events if AudioChunk.is_type(e.type)]
    assert b"".join(c.audio for c in chunk_events) == b"\x01\x02\x03\x04"


@pytest.mark.asyncio
async def test_tts_handler_uses_default_voice_when_none_supplied():
    backend = FakeTTSBackend(chunks=[b"\x00"], voices=["alice", "bob"], sample_rate=24000)
    handler = TtsEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=MagicMock(spec=asyncio.StreamWriter),
        backend=backend,
        default_voice="alice",
        info=Info(),
    )
    await handler.handle_event(Synthesize(text="hi", voice=None).event())
    assert backend.calls == [("hi", "alice")]


@pytest.mark.asyncio
async def test_tts_handler_responds_to_describe():
    from wyoming.info import Describe

    backend = FakeTTSBackend(chunks=[b"\x00"], voices=["v1"], sample_rate=24000)
    capture = _CaptureWriter()
    handler = TtsEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=MagicMock(spec=asyncio.StreamWriter),
        backend=backend,
        default_voice="v1",
        info=Info(),
    )
    handler.write_event = AsyncMock(side_effect=lambda e: capture.capture(e))

    assert await handler.handle_event(Describe().event())

    info_events = [e for e in capture.events if Info.is_type(e.type)]
    assert len(info_events) == 1
    result_info = Info.from_event(info_events[0])
    assert isinstance(result_info, Info)


@pytest.mark.asyncio
async def test_tts_handler_uses_requested_voice():
    from wyoming.tts import SynthesizeVoice

    backend = FakeTTSBackend(chunks=[b"\x00"], voices=["alice", "bob"], sample_rate=24000)
    handler = TtsEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=MagicMock(spec=asyncio.StreamWriter),
        backend=backend,
        default_voice="alice",
        info=Info(),
    )
    await handler.handle_event(Synthesize(text="hello", voice=SynthesizeVoice(name="bob")).event())
    assert backend.calls == [("hello", "bob")]


@pytest.mark.asyncio
async def test_tts_handler_ignores_empty_text():
    backend = FakeTTSBackend(chunks=[b"\x00"], voices=["v1"], sample_rate=24000)
    capture = _CaptureWriter()
    handler = TtsEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=MagicMock(spec=asyncio.StreamWriter),
        backend=backend,
        default_voice="v1",
        info=Info(),
    )
    handler.write_event = AsyncMock(side_effect=lambda e: capture.capture(e))

    result = await handler.handle_event(Synthesize(text="", voice=None).event())

    assert result is True
    assert capture.events == []
    assert backend.calls == []
