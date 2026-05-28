"""Post-deploy smoke checks.  Hits all three servers and reports OK/FAIL.

Usage:
    uv run python scripts/smoke.py \\
        --host <host> \\
        --api-key "$(cat ~/.config/wyoming-mlx/dev-apikey)"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize


async def check_stt(host: str, port: int) -> None:
    pcm = b"\x00\x00" * 16000  # 1 s of silence at 16 kHz
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        await client.write_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event()
        )
        await client.write_event(AudioStop().event())
        event = await client.read_event()
        print(f"[STT] got: {event.type if event else 'no response'}")


async def check_tts(host: str, port: int) -> None:
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Synthesize(text="hello", voice=None).event())
        n_chunks = 0
        while True:
            event = await client.read_event()
            if event is None or event.type == "audio-stop":
                break
            if event.type == "audio-chunk":
                n_chunks += 1
        print(f"[TTS] got {n_chunks} chunks")


def check_http(host: str, port: int, api_key: str) -> None:
    url = f"http://{host}:{port}/v1/models"
    with urllib.request.urlopen(url) as resp:
        body = json.loads(resp.read())
    print(f"[HTTP] /v1/models returned {len(body.get('data', []))} entries")

    req = urllib.request.Request(
        f"http://{host}:{port}/v1/audio/speech",
        data=json.dumps({"input": "smoke", "voice": "af_heart"}).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        n = len(resp.read())
    print(f"[HTTP] /v1/audio/speech returned {n} bytes")


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--stt-port", type=int, default=10300)
    p.add_argument("--tts-port", type=int, default=10200)
    p.add_argument("--http-port", type=int, default=10400)
    p.add_argument("--api-key", required=True)
    args = p.parse_args()

    try:
        await check_stt(args.host, args.stt_port)
        await check_tts(args.host, args.tts_port)
        check_http(args.host, args.http_port, args.api_key)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
