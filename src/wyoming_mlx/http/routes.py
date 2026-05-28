from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterator

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from wyoming_mlx.backends.base import STTBackend, TTSBackend
from wyoming_mlx.config import ModelsConfig


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str
    response_format: str = "wav"


def _require_api_key(api_keys: set[str]):
    def dep(request: Request) -> None:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token"
            )
        token = auth[7:].strip()
        if token not in api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key"
            )

    return dep


def _decode_audio_to_pcm16_mono(raw: bytes) -> tuple[bytes, int]:
    """Decode an uploaded audio file to int16 mono PCM and return (pcm, sample_rate)."""
    data, rate = sf.read(io.BytesIO(raw), dtype="int16", always_2d=True)
    mono = data.mean(axis=1).astype(np.int16) if data.shape[1] > 1 else data[:, 0]
    return mono.tobytes(), int(rate)


def _wav_header(sample_rate: int) -> bytes:
    """A WAV header for an unknown-length stream (data size = 0xFFFFFFFF)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"")
    return buf.getvalue()


def build_router(
    *,
    stt: STTBackend,
    tts: TTSBackend,
    api_keys: set[str],
    models: ModelsConfig,
) -> APIRouter:
    router = APIRouter()
    auth = Depends(_require_api_key(api_keys))

    @router.get("/v1/models")
    async def list_models() -> dict:
        data = [
            {"id": v, "object": "model", "owned_by": "wyoming-mlx"}
            for v in tts.voices
        ]
        data.append(
            {"id": models.whisper, "object": "model", "owned_by": "wyoming-mlx"}
        )
        return {"object": "list", "data": data}

    @router.post("/v1/audio/transcriptions", dependencies=[auth])
    async def transcribe(
        file: UploadFile = File(...),  # noqa: B008
        model: str | None = Form(None),
    ) -> dict:
        raw = await file.read()
        pcm, rate = _decode_audio_to_pcm16_mono(raw)
        text = await stt.transcribe(pcm, rate)
        return {"text": text}

    @router.post("/v1/audio/speech", dependencies=[auth])
    async def speech(req: SpeechRequest):
        if req.response_format != "wav":
            raise HTTPException(
                status_code=400,
                detail=f"unsupported response_format: {req.response_format}",
            )
        if req.voice not in tts.voices:
            raise HTTPException(
                status_code=400, detail=f"unknown voice: {req.voice}"
            )

        async def stream() -> AsyncIterator[bytes]:
            yield _wav_header(tts.sample_rate)
            async for chunk in tts.synthesize(req.input, req.voice):
                yield chunk

        return StreamingResponse(stream(), media_type="audio/wav")

    return router
