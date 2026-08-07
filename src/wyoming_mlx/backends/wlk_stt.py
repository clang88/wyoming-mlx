"""WhisperLiveKit streaming STT backend (AlignAtt / SimulStreaming on MLX)."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

from wyoming_mlx.backends.audio import WHISPER_SAMPLE_RATE, resample_pcm16, resample_to_16k
from wyoming_mlx.backends.base import STTUpdate

try:
    from whisperlivekit import (  # pyright: ignore[reportMissingImports]
        AudioProcessor as _AudioProcessor,
    )
    from whisperlivekit import (  # pyright: ignore[reportMissingImports]
        TranscriptionEngine as _TranscriptionEngine,
    )

    _WLK_AVAILABLE = True
except ImportError:
    _AudioProcessor = None  # type: ignore[assignment]
    _TranscriptionEngine = None  # type: ignore[assignment]
    _WLK_AVAILABLE = False

log = logging.getLogger(__name__)

# Re-exported for backward compatibility; canonical definitions live in audio.py.
_resample_to_16k = resample_to_16k


def _joined_text(lines: list[Any]) -> str:
    """Join WLK confirmed segments into a single transcript string."""
    return " ".join(seg.text.strip() for seg in lines if seg.text and seg.text.strip())


def _delta(emitted: str, confirmed: str) -> str:
    """Newly confirmed suffix, or "" if `confirmed` is not an extension of `emitted`.

    WLK confirmed lines only grow, so the prefix property normally holds; if a
    text revision ever breaks it we emit nothing and let the final transcript
    (which is authoritative) carry the correction.
    """
    if confirmed.startswith(emitted):
        return confirmed[len(emitted) :]
    return ""


# Common short phrases Whisper hallucinates over near-silent trailing audio
# (e.g. mic tail captured after the user stopped speaking). Checked against
# the last sentence of the transcript only, so real speech is never touched.
_HALLUCINATION_FILLERS: dict[str, set[str]] = {
    "en": {
        "okay", "ok", "thank you", "thanks", "thank you for watching",
        "bye", "bye bye", "goodbye", "see you", "see you next time",
    },
    "de": {"okay", "ok", "danke", "tschüss", "bis bald", "vielen dank", "tschau"},
    "es": {"vale", "gracias", "adiós", "de nada", "ok", "okay"},
    "it": {"ok", "okay", "grazie", "arrivederci", "ciao"},
    "ja": {"はい", "ありがとう", "ありがとうございました"},
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _strip_trailing_hallucinations(text: str, language: str | None = None) -> str:
    """Drop trailing filler sentences ("Okay.", "Thank you.", ...) that Whisper
    hallucinates over near-silent tail audio, most often during the final flush.

    Only removes whole trailing sentences that exactly match a known filler
    phrase (case-insensitively); real content is never altered.
    """
    fillers = set(_HALLUCINATION_FILLERS["en"])
    if language:
        fillers |= _HALLUCINATION_FILLERS.get(language.split("-")[0].lower(), set())
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    while len(parts) > 1:
        candidate = parts[-1].strip().rstrip(".!?。！？").strip().lower()
        if candidate in fillers:
            parts.pop()
        else:
            break
    return " ".join(parts).strip()


class _WLKSession:
    """One utterance: wraps a WhisperLiveKit AudioProcessor."""

    def __init__(
        self,
        engine: Any,
        language: str | None = None,
        filter_hallucinations: bool = True,
    ) -> None:
        assert _AudioProcessor is not None
        self._processor = _AudioProcessor(transcription_engine=engine)
        self._results: AsyncIterator[Any] | None = None
        self._emitted = ""
        self._language = language
        self._filter_hallucinations = filter_hallucinations

    async def _ensure_started(self) -> AsyncIterator[Any]:
        if self._results is None:
            self._results = cast(AsyncIterator[Any], await self._processor.create_tasks())
        return self._results

    async def feed(self, audio: bytes, sample_rate: int) -> None:
        await self._ensure_started()
        await self._processor.process_audio(resample_pcm16(audio, sample_rate))

    async def finish(self) -> None:
        await self._ensure_started()
        # Empty message triggers WLK's flush + stop sequence; the results
        # generator terminates once all processing tasks finish.
        await self._processor.process_audio(None)

    async def close(self) -> None:
        await self._processor.cleanup()

    async def updates(self) -> AsyncGenerator[STTUpdate, None]:
        results = await self._ensure_started()
        # Track the longest confirmed text seen — WLK's final flush frame can
        # return empty front.lines, which must not overwrite a good value.
        best_confirmed = ""
        async for front in results:
            if front.status == "error":
                raise RuntimeError(f"WhisperLiveKit error: {front.error}")
            candidate = _joined_text(front.lines)
            if len(candidate) > len(best_confirmed):
                best_confirmed = candidate
            new_text = _delta(self._emitted, best_confirmed)
            if new_text:
                self._emitted = best_confirmed
                yield STTUpdate(text=new_text)
        # Use only confirmed lines as the final transcript. buffer_transcription
        # after flush is unreliable — Whisper hallucinates silence tokens
        # ("Okay.", "Thank you.", etc.) that were never actually spoken.
        final = best_confirmed
        if self._filter_hallucinations:
            final = _strip_trailing_hallucinations(final, self._language)
        yield STTUpdate(final=final)


class WhisperLiveKitBackend:
    """STT backend powered by WhisperLiveKit (SimulStreaming policy, MLX).

    The TranscriptionEngine is a process-wide singleton holding the model;
    it is created eagerly so startup fails fast if the model can't load.
    Each session gets its own AudioProcessor and may run concurrently.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        language: str | None = None,
        filter_hallucinations: bool = True,
    ) -> None:
        if "/" in model:
            raise ValueError(
                f"model must be a WhisperLiveKit size name (e.g. 'large-v3-turbo'), "
                f"not a Hugging Face repo id: {model!r}"
            )
        if not _WLK_AVAILABLE or _TranscriptionEngine is None:
            raise ImportError("whisperlivekit is required but not installed")
        log.info("Loading WhisperLiveKit engine (model=%s, language=%s) …", model, language or "auto")
        engine_kwargs: dict[str, Any] = {
            "model_size": model,
            "backend": "mlx-whisper",
            "backend_policy": "simulstreaming",
            "pcm_input": True,
        }
        if language:
            engine_kwargs["language"] = language
        self._engine = _TranscriptionEngine(**engine_kwargs)
        log.info("[STT] WhisperLiveKit engine ready")
        self._language = language
        self._filter_hallucinations = filter_hallucinations

    def start_session(self) -> _WLKSession:
        return _WLKSession(
            self._engine, language=self._language, filter_hallucinations=self._filter_hallucinations
        )
