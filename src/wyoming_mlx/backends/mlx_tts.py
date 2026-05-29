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
            # American English
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck", "am_santa",
            # British English
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            # Japanese
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
            "jm_kumo",
            # Mandarin Chinese
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
            # Spanish
            "ef_dora", "em_alex", "em_santa",
            # French
            "ff_siwis",
            # Hindi
            "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
            # Italian
            "if_sara", "im_nicola",
            # Brazilian Portuguese
            "pf_dora", "pm_alex", "pm_santa",
        ]
        self._pipeline: object = None
        self._lock = asyncio.Lock()

    def _ensure_pipeline(self) -> object:
        if self._pipeline is None:
            assert _KPipeline is not None, "kokoro is not installed"
            self._pipeline = _KPipeline(lang_code="a", repo_id=self._model_id, device="mps")
        return self._pipeline

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        v = voice or self._voice
        if v not in self.voices:
            raise ValueError(f"unknown voice: {v!r}, expected one of {self.voices}")
        pipeline = self._ensure_pipeline()
        generator = pipeline(text, voice=v)  # pyright: ignore
        buf = io.BytesIO()
        async with self._lock:
            # Serialize all MPS work: the generator yields torch.Tensor results
            # from the Kokoro pipeline, which must stay on CPU for detach/cpu().
            # The lock is released before the yield loop below, so the consumer
            # reads chunks without holding the lock.
            for result in generator:
                audio = result.audio
                assert audio is not None
                if audio.numel() == 0:
                    continue
                arr = audio.detach().cpu().numpy().astype(np.float32)
                # int16 PCM
                pcm = (arr * 32767).astype(np.int16).tobytes()
                buf.write(pcm)
        # Yield in chunks
        data = buf.getvalue()
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
