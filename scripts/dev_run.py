"""Dev runner with fake backends.

Usage:
    uv run python scripts/dev_run.py
"""

import asyncio

from wyoming_mlx.__main__ import run_servers
from wyoming_mlx.backends.fake import FakeSTTBackend, FakeTTSBackend
from wyoming_mlx.config import Config


async def main():
    cfg = Config()
    cfg.http.port = 18400  # local dev ports
    cfg.wyoming.stt_port = 18300
    cfg.wyoming.tts_port = 18200
    await run_servers(
        cfg=cfg,
        stt=FakeSTTBackend(transcript="from the fake"),
        tts=FakeTTSBackend(chunks=[b"\x00\x01" * 2400]),
        api_keys={"dev"},
    )


if __name__ == "__main__":
    asyncio.run(main())
