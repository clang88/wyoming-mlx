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

    def __init__(
        self,
        model: str,
        language: str | None,
        lock: asyncio.Lock,
        *,
        prompt: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._model = model
        self._language = language
        self._lock = lock
        self._prompt = prompt
        self._temperature = temperature
        self._chunks: list[bytes] = []
        self._final = ""
        # updates() is consumed by a pump task started immediately on AudioStart,
        # before finish() has run; gate the yield so it doesn't race ahead.
        self._finished = asyncio.Event()

    async def feed(self, audio: bytes, sample_rate: int) -> None:
        self._chunks.append(resample_pcm16(audio, sample_rate))

    async def finish(self) -> None:
        raw = b"".join(self._chunks)
        if not raw:
            self._final = ""
            self._finished.set()
            return
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        assert _mlx_whisper is not None
        kwargs: dict = {"path_or_hf_repo": self._model, "language": self._language}
        if self._prompt is not None:
            kwargs["initial_prompt"] = self._prompt
        if self._temperature is not None:
            # Only override mlx-whisper's own temperature *fallback ladder* when the
            # caller explicitly asked for one; a bare 0.0 default here would disable
            # its automatic retry-on-failure behaviour for every request.
            kwargs["temperature"] = self._temperature
        try:
            # mlx-whisper's model cache is process-global and not concurrency-safe;
            # serialize decodes and run the blocking call off the event loop.
            async with self._lock:
                result = await asyncio.to_thread(_mlx_whisper.transcribe, pcm, **kwargs)
            self._final = (result.get("text") or "").strip()
        finally:
            self._finished.set()

    async def close(self) -> None:
        self._chunks = []
        self._finished.set()

    async def updates(self) -> AsyncGenerator[STTUpdate, None]:
        # No incremental partials: wait for finish() to populate the final text
        # before yielding it (this generator is consumed concurrently with
        # feed()/finish(), starting right after the session is created).
        await self._finished.wait()
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

    def start_session(
        self,
        *,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
    ) -> _BatchSession:
        return _BatchSession(
            self._model,
            language if language is not None else self._language,
            self._lock,
            prompt=prompt,
            temperature=temperature,
        )
