"""Unit tests for the batch mlx-whisper STT backend (no real model required)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from wyoming_mlx.backends import mlx_stt
from wyoming_mlx.backends.mlx_stt import MlxWhisperBackend, _BatchSession


class _FakeMlxWhisper:
    """Records transcribe() calls and returns a canned result."""

    def __init__(self, text: str = "hello world") -> None:
        self.calls: list[dict] = []
        self.text = text

    def transcribe(self, audio, *, path_or_hf_repo, language=None):
        self.calls.append(
            {"audio": audio, "path_or_hf_repo": path_or_hf_repo, "language": language}
        )
        return {"text": self.text}


def test_backend_raises_when_mlx_whisper_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mlx_stt, "_MLX_WHISPER_AVAILABLE", False)
    with pytest.raises(ImportError, match="mlx-whisper"):
        MlxWhisperBackend()


def test_backend_warms_up_model_at_init(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMlxWhisper()
    monkeypatch.setattr(mlx_stt, "_MLX_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(mlx_stt, "_mlx_whisper", fake)

    MlxWhisperBackend(model="mlx-community/whisper-large-v3-turbo", language="en")

    assert len(fake.calls) == 1
    assert fake.calls[0]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert fake.calls[0]["language"] == "en"


async def test_session_transcribes_fed_audio_once_on_finish(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMlxWhisper(text="turn on the kitchen light")
    monkeypatch.setattr(mlx_stt, "_mlx_whisper", fake)

    session = _BatchSession(model="some/model", language="en", lock=asyncio.Lock())
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    await session.feed(silence, 16000)
    await session.feed(silence, 16000)
    await session.finish()

    updates = [u async for u in session.updates()]

    assert len(updates) == 1
    assert updates[0].final == "turn on the kitchen light"
    assert len(fake.calls) == 1  # single decode for the whole utterance


async def test_session_finish_with_no_audio_yields_empty_final(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMlxWhisper()
    monkeypatch.setattr(mlx_stt, "_mlx_whisper", fake)

    session = _BatchSession(model="some/model", language=None, lock=asyncio.Lock())
    await session.finish()

    updates = [u async for u in session.updates()]

    assert updates[0].final == ""
    assert len(fake.calls) == 0


async def test_updates_waits_for_finish_when_consumed_concurrently(monkeypatch: pytest.MonkeyPatch):
    """Regression test: the Wyoming handler starts consuming updates() via a
    background task immediately on session creation, before finish() runs.
    updates() must block until finish() completes, not yield a stale ""."""
    fake = _FakeMlxWhisper(text="turn off the lights")
    monkeypatch.setattr(mlx_stt, "_mlx_whisper", fake)

    session = _BatchSession(model="some/model", language="en", lock=asyncio.Lock())

    async def pump() -> str:
        final = ""
        async for update in session.updates():
            if update.final is not None:
                final = update.final
        return final

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(0)  # let the pump task start consuming updates() first

    await session.feed(np.zeros(1600, dtype=np.int16).tobytes(), 16000)
    await session.finish()

    assert await pump_task == "turn off the lights"
