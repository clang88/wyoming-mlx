"""Unit tests for MLXWhisperBackend helpers (no real model required)."""

from __future__ import annotations

import numpy as np

from wyoming_mlx.backends.mlx_stt import WHISPER_SAMPLE_RATE, _resample_to_16k


def test_resample_passthrough_at_16k():
    pcm = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    out = _resample_to_16k(pcm, 16000)
    assert out is pcm or np.array_equal(out, pcm)
    assert out.dtype == np.float32


def test_resample_24k_to_16k_changes_length_correctly():
    # 1 second of 24 kHz audio should become 16000 samples at 16 kHz
    pcm = np.zeros(24000, dtype=np.float32)
    out = _resample_to_16k(pcm, 24000)
    assert out.dtype == np.float32
    assert out.shape == (WHISPER_SAMPLE_RATE,)


def test_resample_preserves_sinusoid_frequency():
    # A 440Hz tone at 24kHz should still be ~440Hz after resampling to 16kHz
    sr_in = 24000
    duration = 0.5
    freq = 440.0
    t = np.arange(int(sr_in * duration), dtype=np.float32) / sr_in
    pcm = np.sin(2 * np.pi * freq * t).astype(np.float32)

    out = _resample_to_16k(pcm, sr_in)
    assert out.shape == (int(WHISPER_SAMPLE_RATE * duration),)

    # FFT peak should land on 440Hz (within a few Hz)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1 / WHISPER_SAMPLE_RATE)
    peak_freq = freqs[int(np.argmax(spectrum))]
    assert abs(peak_freq - freq) < 5.0, f"expected ~440Hz, got {peak_freq:.1f}Hz"
