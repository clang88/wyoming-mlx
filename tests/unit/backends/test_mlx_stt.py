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

    def transcribe(
        self,
        audio: object,
        *,
        path_or_hf_repo: str,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        self.calls.append(
            {
                "audio": audio,
                "path_or_hf_repo": path_or_hf_repo,
                "language": language,
                "initial_prompt": initial_prompt,
                "temperature": temperature,
            }
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


async def test_session_forwards_prompt_and_temperature_when_set(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMlxWhisper()
    monkeypatch.setattr(mlx_stt, "_mlx_whisper", fake)

    session = _BatchSession(
        model="some/model", language="en", lock=asyncio.Lock(), prompt="hint", temperature=0.4
    )
    await session.feed(np.zeros(1600, dtype=np.int16).tobytes(), 16000)
    await session.finish()

    assert fake.calls[0]["initial_prompt"] == "hint"
    assert fake.calls[0]["temperature"] == 0.4


async def test_session_omits_prompt_and_temperature_when_unset(monkeypatch: pytest.MonkeyPatch):
    """Only pass temperature/initial_prompt through when explicitly requested,
    so mlx-whisper's own retry-on-failure temperature ladder stays active by
    default."""
    fake = _FakeMlxWhisper()
    monkeypatch.setattr(mlx_stt, "_mlx_whisper", fake)

    session = _BatchSession(model="some/model", language="en", lock=asyncio.Lock())
    await session.feed(np.zeros(1600, dtype=np.int16).tobytes(), 16000)
    await session.finish()

    assert fake.calls[0]["initial_prompt"] is None
    assert fake.calls[0]["temperature"] is None


def test_backend_start_session_overrides_default_language(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMlxWhisper()
    monkeypatch.setattr(mlx_stt, "_MLX_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(mlx_stt, "_mlx_whisper", fake)

    backend = MlxWhisperBackend(model="some/model", language="en")

    session = backend.start_session(language="de")
    assert session._language == "de"

    default_session = backend.start_session()
    assert default_session._language == "en"
