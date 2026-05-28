"""Kokoro TTS backend using PyTorch MPS on Apple Silicon."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator

import numpy as np

_kokoro_ok = False
try:
    from kokoro import KPipeline as _KPipeline  # type: ignore[import-not-found, unused-ignore]

    _kokoro_ok = True
except ImportError:
    _KPipeline = None  # type: ignore[assignment,misc]


class KokoroBackend:
    """TTS backend powered by Kokoro-82M via PyTorch MPS."""

    def __init__(
        self,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
    ) -> None:
        if not _kokoro_ok:
            raise ImportError("kokoro is required for KokoroBackend")
        self._voice = voice
        self._model_id = model_id
        self.sample_rate = 24000
        self.voices: list[str] = [
            "af_heart",
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michelle",
            "am_oxford",
            "am_puck",
            "am_santa",
        ]
        self._pipeline: object = None
        self._lock = asyncio.Lock()

    def _ensure_pipeline(self) -> object:
        if self._pipeline is None:
            assert _KPipeline is not None, "kokoro is not installed"
            self._pipeline = _KPipeline(lang_code="a", repo_id=self._model_id, device="mps")
        return self._pipeline

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        pipeline = self._ensure_pipeline()
        generator = pipeline(text, voice=voice or self._voice)  # pyright: ignore
        buf = io.BytesIO()
        async with self._lock:
            for result in generator:
                audio = result.audio  # torch.Tensor on CPU
                assert audio is not None
                arr = audio.detach().cpu().numpy().astype(np.float32)
                # int16 PCM
                pcm = (arr * 32767).astype(np.int16).tobytes()
                buf.write(pcm)
        # Yield in chunks
        data = buf.getvalue()
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
