import asyncio

import pytest

from wyoming_mlx.backends.fake import FakeSTTBackend, FakeTTSBackend
from wyoming_mlx.config import Config


async def test_run_servers_starts_and_stops():
    """Smoke-test: run_servers comes up on free ports and shuts down on cancel."""
    cfg = Config()
    cfg.wyoming.stt_port = 0  # OS picks free port
    cfg.wyoming.tts_port = 0
    cfg.http.port = 0

    stt = FakeSTTBackend(transcript="x")
    tts = FakeTTSBackend(chunks=[b"\x00"], voices=["af_heart"], sample_rate=24000)

    from wyoming_mlx.__main__ import run_servers

    task = asyncio.create_task(run_servers(cfg=cfg, stt=stt, tts=tts, api_keys=set()))
    # Give the servers a moment to come up.
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
