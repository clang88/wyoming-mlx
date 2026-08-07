"""Unit tests for WhisperLiveKit backend helpers (no whisperlivekit install required)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from wyoming_mlx.backends.wlk_stt import (
    WHISPER_SAMPLE_RATE,
    WhisperLiveKitBackend,
    _delta,
    _joined_text,
    _resample_to_16k,
    _strip_trailing_hallucinations,
    _WLKSession,
    resample_pcm16,
)


def test_backend_rejects_hf_repo_id():
    """A Hugging Face repo id is rejected up front, before any model load."""
    with pytest.raises(ValueError, match="repo id"):
        WhisperLiveKitBackend(model="mlx-community/distil-whisper-large-v3")


def test_resample_passthrough_at_16k():
    pcm = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    out = _resample_to_16k(pcm, 16000)
    assert out is pcm or np.array_equal(out, pcm)
    assert out.dtype == np.float32


def test_resample_24k_to_16k_changes_length_correctly():
    pcm = np.zeros(24000, dtype=np.float32)
    out = _resample_to_16k(pcm, 24000)
    assert out.dtype == np.float32
    assert out.shape == (WHISPER_SAMPLE_RATE,)


def test_resample_preserves_sinusoid_frequency():
    sr_in = 24000
    duration = 0.5
    freq = 440.0
    t = np.arange(int(sr_in * duration), dtype=np.float32) / sr_in
    pcm = np.sin(2 * np.pi * freq * t).astype(np.float32)
    out = _resample_to_16k(pcm, sr_in)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), d=1 / 16000)[int(np.argmax(spectrum))]
    assert abs(peak_hz - freq) < 5


def test_resample_pcm16_passthrough_at_16k_is_lossless():
    audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16).tobytes()
    assert resample_pcm16(audio, 16000) == audio


def test_resample_pcm16_halves_48k_input():
    audio = np.zeros(4800, dtype=np.int16).tobytes()
    out = resample_pcm16(audio, 48000)
    assert len(out) == 1600 * 2  # 1600 samples of int16


def test_joined_text_skips_empty_segments():
    lines = [
        SimpleNamespace(text="hello"),
        SimpleNamespace(text=None),
        SimpleNamespace(text="  "),
        SimpleNamespace(text="world"),
    ]
    assert _joined_text(lines) == "hello world"


def test_delta_returns_new_suffix():
    assert _delta("hello ", "hello world") == "world"


def test_delta_empty_when_not_a_prefix():
    assert _delta("hello there", "hello world") == ""


def test_delta_from_empty_emits_everything():
    assert _delta("", "hello") == "hello"


def _session_with_results(
    fronts: list[SimpleNamespace],
    language: str | None = None,
    filter_hallucinations: bool = True,
) -> _WLKSession:
    """Build a _WLKSession without an AudioProcessor, injecting fabricated results."""
    session = object.__new__(_WLKSession)
    session._emitted = ""
    session._language = language
    session._filter_hallucinations = filter_hallucinations

    async def results():
        for front in fronts:
            yield front

    session._results = results()
    return session


def _front(texts: list[str | None], buffer: str = "", status: str = "active_transcription"):
    return SimpleNamespace(
        status=status,
        error="",
        lines=[SimpleNamespace(text=t) for t in texts],
        buffer_transcription=buffer,
    )


async def test_updates_emits_deltas_then_final_ignoring_buffer_tail():
    """buffer_transcription is dropped from the final — it's speculative and
    unreliable (Whisper hallucinates silence tokens there after flush)."""
    session = _session_with_results(
        [
            _front(["hello"]),
            _front(["hello", "world"], buffer="tail "),
        ]
    )

    updates = [u async for u in session.updates()]

    assert [u.text for u in updates[:-1]] == ["hello", " world"]
    assert updates[-1].final == "hello world"


async def test_updates_strips_trailing_hallucinated_filler():
    """A confirmed 'Okay.' tacked on after real speech is dropped from the final."""
    session = _session_with_results(
        [_front(["Let's see if this fixes it.", "Okay."])],
    )

    updates = [u async for u in session.updates()]

    assert updates[-1].final == "Let's see if this fixes it."


async def test_updates_keeps_trailing_filler_when_filter_disabled():
    """filter_hallucinations=False leaves the confirmed transcript untouched."""
    session = _session_with_results(
        [_front(["Let's see if this fixes it.", "Okay."])],
        filter_hallucinations=False,
    )

    updates = [u async for u in session.updates()]

    assert updates[-1].final == "Let's see if this fixes it. Okay."


def test_strip_trailing_hallucinations_drops_known_filler():
    text = "This is the real sentence. Okay."
    assert _strip_trailing_hallucinations(text) == "This is the real sentence."


def test_strip_trailing_hallucinations_drops_repeated_filler():
    text = "This is the real sentence. Okay. Okay."
    assert _strip_trailing_hallucinations(text) == "This is the real sentence."


def test_strip_trailing_hallucinations_uses_language_specific_fillers():
    text = "Das ist der echte Satz. Danke."
    assert _strip_trailing_hallucinations(text, language="de") == "Das ist der echte Satz."


def test_strip_trailing_hallucinations_leaves_real_content_untouched():
    text = "Turn on the kitchen light."
    assert _strip_trailing_hallucinations(text) == text


def test_strip_trailing_hallucinations_leaves_sole_filler_utterance():
    # If the entire utterance is just the filler word, it's presumably real speech.
    assert _strip_trailing_hallucinations("Okay.") == "Okay."


async def test_updates_yields_only_final_when_nothing_confirmed():
    session = _session_with_results([_front([], status="no_audio_detected")])

    updates = [u async for u in session.updates()]

    assert len(updates) == 1
    assert updates[0].final == ""


async def test_updates_raises_on_error_status():
    session = _session_with_results(
        [SimpleNamespace(status="error", error="boom", lines=[], buffer_transcription="")]
    )

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in session.updates():
            pass


def test_backend_start_session_language_override_only_affects_filtering(
    monkeypatch: pytest.MonkeyPatch,
):
    """The recognition engine's language is fixed at construction; a per-request
    override is only honoured for trailing-hallucination filtering."""
    import wyoming_mlx.backends.wlk_stt as wlk_stt_module

    created: list[dict[str, Any]] = []

    class _FakeAudioProcessor:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(wlk_stt_module, "_AudioProcessor", _FakeAudioProcessor)

    backend = object.__new__(WhisperLiveKitBackend)
    backend._engine = cast(Any, object())
    backend._language = "en"
    backend._filter_hallucinations = True

    session = backend.start_session(language="de")

    assert session._language == "de"
    assert created[-1]["transcription_engine"] is backend._engine

    default_session = backend.start_session()
    assert default_session._language == "en"
