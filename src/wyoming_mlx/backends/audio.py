"""Shared audio resampling helpers used by STT backends."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

WHISPER_SAMPLE_RATE = 16000


def resample_to_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
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


def resample_pcm16(audio: bytes, sample_rate: int) -> bytes:
    """Resample int16 mono PCM bytes to 16 kHz int16 mono PCM bytes."""
    if sample_rate == WHISPER_SAMPLE_RATE:
        return audio
    pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    pcm16k = resample_to_16k(pcm, sample_rate)
    return (np.clip(pcm16k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
