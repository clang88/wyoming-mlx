"""MLX Whisper STT backend."""

from __future__ import annotations

import asyncio
import logging

import mlx_whisper
import numpy as np

log = logging.getLogger(__name__)


class MLXWhisperBackend:
    """STT backend powered by MLX Whisper.

    Thread-safe via ``asyncio.Lock`` — multiple concurrent transcribe calls
    are serialised on the GPU.
    """

    def __init__(self, model_id: str = "mlx-community/distil-whisper-large-v3") -> None:
        self._model_id = model_id
        self._lock = asyncio.Lock()

    async def transcribe(self, audio: bytes, sample_rate: int) -> str:
        # PCM int16 → float32 in [-1, 1]
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        async with self._lock:
            result: dict = mlx_whisper.transcribe(  # type: ignore[assignment]
                pcm,
                path_or_hf_repo=self._model_id,
                fp16=True,
            )
        text = str(result.get("text", "") or "")
        return text.strip()
