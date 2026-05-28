import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.tts import Synthesize

from wyoming_mlx.backends.fake import FakeTTSBackend
from wyoming_mlx.wyoming_servers.tts import TtsEventHandler


class _Capture:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def write_event(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_tts_handler_emits_audio_envelope():
    backend = FakeTTSBackend(chunks=[b"\x01\x02", b"\x03\x04"], voices=["v1"], sample_rate=24000)
    writer = _Capture()
    handler = TtsEventHandler(backend=backend, writer=writer, default_voice="v1")

    await handler.handle_event(Synthesize(text="hello", voice=None).event())

    assert writer.events, "no events emitted"
    assert AudioStart.is_type(writer.events[0].type)
    assert AudioStop.is_type(writer.events[-1].type)
    chunk_events = [AudioChunk.from_event(e) for e in writer.events if AudioChunk.is_type(e.type)]
    assert b"".join(c.audio for c in chunk_events) == b"\x01\x02\x03\x04"


@pytest.mark.asyncio
async def test_tts_handler_uses_default_voice_when_none_supplied():
    backend = FakeTTSBackend(chunks=[b"\x00"], voices=["alice", "bob"], sample_rate=24000)
    handler = TtsEventHandler(backend=backend, writer=_Capture(), default_voice="alice")
    await handler.handle_event(Synthesize(text="hi", voice=None).event())
    assert backend.calls == [("hi", "alice")]
