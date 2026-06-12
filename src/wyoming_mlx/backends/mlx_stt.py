"""MLX Whisper STT backend."""

from __future__ import annotations

import asyncio
import logging
from math import gcd

import numpy as np

try:
    import mlx_whisper as _mlx_whisper  # pyright: ignore[reportMissingImports]

    _MLX_AVAILABLE = True
except ImportError:
    _mlx_whisper = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def _resample_to_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample float32 mono audio to Whisper's required 16 kHz.

    Whisper produces empty/garbled transcripts when fed audio at the wrong
    rate, since its mel spectrogram is computed assuming 16 kHz input.
    """
    if sample_rate == WHISPER_SAMPLE_RATE:
        return pcm
    g = gcd(sample_rate, WHISPER_SAMPLE_RATE)
    up = WHISPER_SAMPLE_RATE // g
    down = sample_rate // g
    return resample_poly(pcm, up, down).astype(np.float32)


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
        pcm = _resample_to_16k(pcm, sample_rate)
        assert _MLX_AVAILABLE, "mlx_whisper required"
        assert _mlx_whisper is not None
        async with self._lock:
            result: dict = _mlx_whisper.transcribe(  # type: ignore[assignment]
                pcm,
                path_or_hf_repo=self._model_id,
                fp16=True,
            )
        text = str(result.get("text", "") or "")
        return text.strip()
