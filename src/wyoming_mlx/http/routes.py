from __future__ import annotations

import hmac
import io
import struct
from collections.abc import AsyncIterator

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from wyoming_mlx.backends.base import STTBackend, TTSBackend, collect_transcript
from wyoming_mlx.config import ModelsConfig


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str
    response_format: str = "wav"


_ALLOWED_AUDIO_FORMATS = {"WAV", "FLAC", "OGG", "MP3"}


def _require_api_key(api_keys: set[str]):
    def dep(request: Request) -> None:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token"
            )
        token = auth[7:].strip()
        if not api_keys or not any(hmac.compare_digest(token, key) for key in api_keys):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")

    return dep


def _sniff_audio_format(raw: bytes) -> str | None:
    """Best-effort audio container/format detection from magic bytes."""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "WAV"
    if raw[:4] == b"fLaC":
        return "FLAC"
    if raw[:4] == b"OggS":
        return "OGG"
    if raw[:3] == b"ID3" or (raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0):
        # ID3v2 tag (ffmpeg/OpenWebUI's default mp3 muxer prepends one) or a
        # raw MPEG frame sync word (any MPEG version/layer, not just v1 layer 3).
        return "MP3"
    return None


def _decode_audio_to_pcm16_mono(raw: bytes) -> tuple[bytes, int]:
    """Decode an uploaded audio file to int16 mono PCM and return (pcm, sample_rate)."""
    if len(raw) < 12:
        raise HTTPException(status_code=400, detail="audio file too small")

    detected = _sniff_audio_format(raw)
    if detected is None:
        raise HTTPException(
            status_code=400,
            detail="unsupported audio format; only WAV, FLAC, OGG, MP3 accepted",
        )

    if detected not in _ALLOWED_AUDIO_FORMATS:
        raise HTTPException(status_code=400, detail=f"audio format {detected} not allowed")

    data, rate = sf.read(io.BytesIO(raw), dtype="int16", always_2d=True)
    mono = data.mean(axis=1).astype(np.int16) if data.shape[1] > 1 else data[:, 0]
    return mono.tobytes(), int(rate)


def _wav_header(sample_rate: int) -> bytes:
    """Streaming WAV header for mono int16 PCM.

    Declares RIFF and data sizes as 0xFFFFFFFF — libsndfile and most other
    readers treat this as "unknown length" and use the file/stream's actual
    byte length instead. Using ``wave.open`` here is wrong: it writes a
    header with data size = 0 and the trailing PCM is ignored.
    """
    nchannels = 1
    sampwidth = 2
    byte_rate = sample_rate * nchannels * sampwidth
    block_align = nchannels * sampwidth
    bits_per_sample = sampwidth * 8
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack(
            "<HHIIHH",
            1,
            nchannels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )


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
        data = [{"id": v, "object": "model", "owned_by": "wyoming-mlx"} for v in tts.voices]
        data.append({"id": models.whisper, "object": "model", "owned_by": "wyoming-mlx"})
        return {"object": "list", "data": data}

    @router.get("/v1/audio/voices")
    async def list_voices() -> dict:
        # Open WebUI (and other OpenAI-compatible clients) call this to populate
        # the TTS voice picker; without it they fall back to hardcoded OpenAI
        # voice names (alloy, echo, ...) that don't exist in `tts.voices`.
        return {"voices": [{"id": v, "name": v} for v in tts.voices]}

    @router.get("/v1/audio/models")
    async def list_audio_models() -> dict:
        return {"models": [{"id": models.kokoro}]}

    @router.post("/v1/audio/transcriptions", dependencies=[auth])
    async def transcribe(
        file: UploadFile = File(...),  # noqa: B008
        model: str | None = Form(None),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(None),
    ):
        if response_format not in ("json", "text"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unsupported response_format: {response_format!r}; "
                    "only 'json' and 'text' are supported"
                ),
            )
        if file.size is not None and file.size > 100_000_000:
            raise HTTPException(status_code=413, detail="audio file exceeds 100MB limit")
        raw = await file.read()
        if file.size is None and len(raw) > 100_000_000:
            raise HTTPException(status_code=413, detail="audio file exceeds 100MB limit")
        pcm, rate = _decode_audio_to_pcm16_mono(raw)
        try:
            text = await collect_transcript(
                stt, pcm, rate, language=language, prompt=prompt, temperature=temperature
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="transcription failed") from exc
        if response_format == "text":
            return PlainTextResponse(text)
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
                status_code=400,
                detail=f"unknown voice: {req.voice!r}, expected one of {tts.voices}",
            )

        async def stream() -> AsyncIterator[bytes]:
            yield _wav_header(tts.sample_rate)
            async for chunk in tts.synthesize(req.input, req.voice):
                yield chunk

        return StreamingResponse(stream(), media_type="audio/wav")

    return router
