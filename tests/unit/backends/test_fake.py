import pytest

from wyoming_mlx.backends.base import STTBackend, TTSBackend
from wyoming_mlx.backends.fake import FakeSTTBackend, FakeTTSBackend


def test_fakes_satisfy_protocols():
    assert isinstance(FakeSTTBackend(), STTBackend)
    assert isinstance(FakeTTSBackend(chunks=[b""]), TTSBackend)


@pytest.mark.asyncio
async def test_fake_stt_returns_canned_transcript():
    backend = FakeSTTBackend(transcript="hello world")
    result = await backend.transcribe(b"\x00" * 1600, sample_rate=16000)
    assert result == "hello world"


@pytest.mark.asyncio
async def test_fake_stt_records_calls():
    backend = FakeSTTBackend(transcript="x")
    await backend.transcribe(b"abc", sample_rate=16000)
    await backend.transcribe(b"def", sample_rate=22050)
    assert backend.calls == [(b"abc", 16000), (b"def", 22050)]


@pytest.mark.asyncio
async def test_fake_tts_yields_canned_chunks():
    backend = FakeTTSBackend(chunks=[b"AAAA", b"BBBB"], voices=["v1", "v2"])
    out = [chunk async for chunk in backend.synthesize("hi", voice="v1")]
    assert out == [b"AAAA", b"BBBB"]


@pytest.mark.asyncio
async def test_fake_tts_voices_attribute():
    backend = FakeTTSBackend(chunks=[b""], voices=["alice", "bob"])
    assert backend.voices == ["alice", "bob"]
