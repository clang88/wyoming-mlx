import pytest
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

from wyoming_mlx.backends.fake import FakeSTTBackend
from wyoming_mlx.wyoming_servers.stt import SttEventHandler


class _Capture:
    """Stand-in for the Wyoming writer; captures emitted events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def write_event(self, event: Event) -> None:
        self.events.append(event)


def _pcm_chunk(n_samples: int = 1600) -> bytes:
    """A silent 16 kHz mono 16-bit PCM block."""
    return b"\x00\x00" * n_samples


@pytest.mark.asyncio
async def test_stt_handler_emits_transcript_after_audio_stop():
    backend = FakeSTTBackend(transcript="the quick brown fox")
    writer = _Capture()
    handler = SttEventHandler(backend=backend, writer=writer)

    assert await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=_pcm_chunk()).event())
    assert await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=_pcm_chunk()).event())
    assert await handler.handle_event(AudioStop().event())

    transcripts = [Transcript.from_event(e) for e in writer.events if Transcript.is_type(e.type)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "the quick brown fox"


@pytest.mark.asyncio
async def test_stt_handler_passes_concatenated_audio_to_backend():
    backend = FakeSTTBackend(transcript="x")
    handler = SttEventHandler(backend=backend, writer=_Capture())
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x01\x02").event())
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x03\x04").event())
    await handler.handle_event(AudioStop().event())

    assert len(backend.calls) == 1
    audio, rate = backend.calls[0]
    assert audio == b"\x01\x02\x03\x04"
    assert rate == 16000
