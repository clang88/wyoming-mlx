"""Non-streaming (batch) Whisper STT backend using mlx-whisper directly.

Buffers the whole utterance in memory and transcribes it in a single pass on
finish(), instead of using an incremental confirmation policy. This avoids
the word-reordering and trailing-silence-hallucination artifacts that a
streaming policy's forced final flush can introduce (see backends/wlk_stt.py
for the streaming alternative). mlx_whisper.transcribe() also applies
whole-clip hallucination suppression (no_speech_threshold, logprob_threshold,
compression_ratio_threshold) that a partial-chunk decode cannot.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

import numpy as np

from wyoming_mlx.backends.audio import WHISPER_SAMPLE_RATE, resample_pcm16
from wyoming_mlx.backends.base import STTUpdate

try:
    import mlx_whisper as _mlx_whisper  # type: ignore[import-not-found]

    _MLX_WHISPER_AVAILABLE = True
except ImportError:
    _mlx_whisper = None  # type: ignore[assignment]
    _MLX_WHISPER_AVAILABLE = False

log = logging.getLogger(__name__)


class _BatchSession:
    """One utterance: buffers PCM in memory, transcribes once on finish()."""

    def __init__(self, model: str, language: str | None, lock: asyncio.Lock) -> None:
        self._model = model
        self._language = language
        self._lock = lock
        self._chunks: list[bytes] = []
        self._final = ""

    async def feed(self, audio: bytes, sample_rate: int) -> None:
        self._chunks.append(resample_pcm16(audio, sample_rate))

    async def finish(self) -> None:
        raw = b"".join(self._chunks)
        if not raw:
            self._final = ""
            return
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        assert _mlx_whisper is not None
        # mlx-whisper's model cache is process-global and not concurrency-safe;
        # serialize decodes and run the blocking call off the event loop.
        async with self._lock:
            result = await asyncio.to_thread(
                _mlx_whisper.transcribe,
                pcm,
                path_or_hf_repo=self._model,
                language=self._language,
            )
        self._final = (result.get("text") or "").strip()

    async def close(self) -> None:
        self._chunks = []

    async def updates(self) -> AsyncGenerator[STTUpdate, None]:
        # No incremental partials: the whole utterance is decoded in one pass.
        yield STTUpdate(final=self._final)


class MlxWhisperBackend:
    """Batch STT backend: one mlx-whisper decode per utterance, no streaming.

    Simpler and more reliable than a streaming confirmation policy at the
    cost of live partial transcripts: no incremental "confirmed vs buffer"
    split, and no separate flush decode over trailing near-silent audio.
    """

    def __init__(
        self,
        model: str = "mlx-community/whisper-large-v3-turbo",
        language: str | None = None,
    ) -> None:
        if not _MLX_WHISPER_AVAILABLE or _mlx_whisper is None:
            raise ImportError("mlx-whisper is required but not installed")
        self._model = model
        self._language = language
        self._lock = asyncio.Lock()
        log.info("Loading mlx-whisper model %s (language=%s) …", model, language or "auto")
        # Warm up now so a bad model id / download failure surfaces at startup,
        # not on the first user request. Also populates mlx-whisper's model cache.
        _mlx_whisper.transcribe(
            np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=model,
            language=language,
        )
        log.info("[STT] mlx-whisper model ready")

    def start_session(self) -> _BatchSession:
        return _BatchSession(self._model, self._language, self._lock)
