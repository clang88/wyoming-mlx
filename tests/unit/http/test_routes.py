import io
import wave

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from wyoming_mlx.backends.fake import FakeSTTBackend, FakeTTSBackend
from wyoming_mlx.config import ModelsConfig
from wyoming_mlx.http.app import create_app


def _wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def app():
    stt = FakeSTTBackend(transcript="hi there")
    tts = FakeTTSBackend(
        chunks=[b"\x01\x02", b"\x03\x04"],
        voices=["alice", "bob"],
        sample_rate=24000,
    )
    return create_app(
        stt=stt,
        tts=tts,
        api_keys={"sekret"},
        models=ModelsConfig(whisper="mlx-community/distil-whisper-large-v3"),
    )


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_models_route_is_public(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert any(m["id"] == "alice" for m in body["data"])
    assert any(
        m["id"] == "mlx-community/distil-whisper-large-v3" for m in body["data"]
    )


@pytest.mark.asyncio
async def test_transcriptions_requires_auth(client):
    r = await client.post(
        "/v1/audio/transcriptions",
        files={
            "file": (
                "a.wav",
                _wav_bytes(b"\x00" * 1600),
                "audio/wav",
            )
        },
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_transcriptions_with_auth(client):
    r = await client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sekret"},
        files={
            "file": (
                "a.wav",
                _wav_bytes(b"\x00" * 1600),
                "audio/wav",
            )
        },
    )
    assert r.status_code == 200
    assert r.json() == {"text": "hi there"}


@pytest.mark.asyncio
async def test_speech_requires_auth(client):
    r = await client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "alice"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_speech_returns_wav(client):
    r = await client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer sekret"},
        json={"input": "hello", "voice": "alice"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"
    assert b"\x01\x02\x03\x04" in r.content


@pytest.mark.asyncio
async def test_speech_rejects_unsupported_format(client):
    r = await client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer sekret"},
        json={"input": "hello", "voice": "alice", "response_format": "mp3"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_transcriptions_rejects_large_file(client):
    large_wav = _wav_bytes(b"\x00" * 200_000_000)
    r = await client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sekret"},
        files={"file": ("big.wav", large_wav, "audio/wav")},
    )
    assert r.status_code == 413
    assert "exceeds" in r.json()["detail"]


@pytest.mark.asyncio
async def test_transcriptions_rejects_unknown_magic():
    from wyoming_mlx.http.routes import _decode_audio_to_pcm16_mono

    with pytest.raises(HTTPException) as exc_info:
        _decode_audio_to_pcm16_mono(b"This is not audio")
    assert exc_info.value.status_code == 400
    assert "unsupported audio format" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_speech_rejects_unknown_voice(client):
    r = await client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer sekret"},
        json={"input": "hello", "voice": "nonexistent_voice"},
    )
    assert r.status_code == 400
    assert "unknown voice" in r.json()["detail"]
