"""MLX Whisper STT backend."""

from __future__ import annotations

import mlx_whisper
import numpy as np


class MLXWhisperBackend:
    """STT backend powered by MLX Whisper."""

    def __init__(self, model_id: str = "mlx-community/distil-whisper-large-v3") -> None:
        self._model_id = model_id

    async def transcribe(self, audio: bytes, sample_rate: int) -> str:
        # PCM int16 -> float32 in [-1, 1]
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            result: dict = mlx_whisper.transcribe(  # type: ignore[assignment]
                pcm,
                model=self._model_id,
                device="metal",
                fp16=True,
            )
        except Exception:
            log = __import__("logging").getLogger(__name__)
            log.exception("MLX Whisper transcription failed")
            return ""
        text = str(result.get("text", "") or "")
        return text.strip()
