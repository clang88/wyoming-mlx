"""Entry point: start Wyoming STT, Wyoming TTS, and HTTP servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import uvicorn
from wyoming.event import Event
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Info,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncServer, async_read_event

from wyoming_mlx.auth import load_api_keys
from wyoming_mlx.backends.base import STTBackend, TTSBackend
from wyoming_mlx.config import Config, load_config
from wyoming_mlx.http.app import create_app
from wyoming_mlx.wyoming_servers.stt import SttEventHandler
from wyoming_mlx.wyoming_servers.tts import TtsEventHandler

log = logging.getLogger(__name__)


class _AsyncWriter:
    """Adapts an asyncio.StreamWriter to the _Writer protocol used by handler classes."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def write_event(self, event: Event) -> None:
        data = event.to_dict()
        json_line = json.dumps({"type": event.type, **data}, ensure_ascii=False)
        self._writer.writelines((json_line.encode(), b"\n"))
        if event.payload:
            self._writer.write(event.payload)
        await self._writer.drain()


def _build_stt_info(model_id: str) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="wyoming-mlx",
                attribution=Attribution(name="wyoming-mlx", url=""),
                installed=True,
                description="MLX whisper STT",
                version="0.1.0",
                models=[
                    AsrModel(
                        name=model_id,
                        attribution=Attribution(name="OpenAI/MLX", url=""),
                        installed=True,
                        description=model_id,
                        version="0.1.0",
                        languages=["en"],
                    )
                ],
            )
        ],
    )


def _build_tts_info(voices: list[str]) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="wyoming-mlx",
                attribution=Attribution(name="wyoming-mlx", url=""),
                installed=True,
                description="MLX Kokoro TTS",
                version="0.1.0",
                voices=[
                    TtsVoice(
                        name=v,
                        attribution=Attribution(name="Kokoro", url=""),
                        installed=True,
                        description=v,
                        version="0.1.0",
                        languages=["en"],
                    )
                    for v in voices
                ],
            )
        ],
    )


def _stt_handler_factory(
    backend: STTBackend,
    info: Info,
) -> type:
    """Return a handler class for the Wyoming STT server."""

    class _H:
        def __init__(self, reader, writer):
            self._inner = SttEventHandler(backend=backend, writer=_AsyncWriter(writer))
            self._reader = reader

        async def run(self):
            while True:
                event = await async_read_event(self._reader)
                if event is None:
                    return
                if not await self._inner.handle_event(event):
                    return

    return _H


def _tts_handler_factory(
    backend: TTSBackend,
    default_voice: str,
    info: Info,
) -> type:
    """Return a handler class for the Wyoming TTS server."""

    class _H:
        def __init__(self, reader, writer):
            self._inner = TtsEventHandler(
                backend=backend,
                writer=_AsyncWriter(writer),
                default_voice=default_voice,
            )
            self._reader = reader

        async def run(self):
            while True:
                event = await async_read_event(self._reader)
                if event is None:
                    return
                if not await self._inner.handle_event(event):
                    return

    return _H


async def run_servers(
    *,
    cfg: Config,
    stt: STTBackend,
    tts: TTSBackend,
    api_keys: set[str],
) -> None:
    """Start Wyoming STT, Wyoming TTS, and HTTP servers and run until cancelled."""

    stt_info = _build_stt_info(cfg.models.whisper)
    tts_info = _build_tts_info(tts.voices)

    stt_server = AsyncServer.from_uri(
        f"tcp://{cfg.wyoming.stt_host}:{cfg.wyoming.stt_port}"
    )
    tts_server = AsyncServer.from_uri(
        f"tcp://{cfg.wyoming.tts_host}:{cfg.wyoming.tts_port}"
    )

    app = create_app(
        stt=stt,
        tts=tts,
        api_keys=api_keys,
        whisper_model_id=cfg.models.whisper,
    )
    http_config = uvicorn.Config(
        app,
        host=cfg.http.host,
        port=cfg.http.port,
        log_level=cfg.logging.level.lower(),
    )
    http_server = uvicorn.Server(http_config)

    log.info("Starting wyoming-mlx (STT=%s:%s TTS=%s:%s HTTP=%s:%s)",
             cfg.wyoming.stt_host, cfg.wyoming.stt_port,
             cfg.wyoming.tts_host, cfg.wyoming.tts_port,
             cfg.http.host, cfg.http.port)

    stt_server_task = asyncio.create_task(
        stt_server.run(
            _stt_handler_factory(backend=stt, info=stt_info),
        ),
        name="stt-server",
    )
    tts_server_task = asyncio.create_task(
        tts_server.run(
            _tts_handler_factory(
                backend=tts,
                default_voice=cfg.models.kokoro_default_voice,
                info=tts_info,
            ),
        ),
        name="tts-server",
    )
    http_server_task = asyncio.create_task(
        http_server.serve(),
        name="http-server",
    )

    await asyncio.gather(stt_server_task, tts_server_task, http_server_task)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wyoming-mlx: MLX STT/TTS server")
    parser.add_argument("--config", type=Path, default=None, help="Path to TOML config file")
    parser.add_argument("--stt-host", default=None)
    parser.add_argument("--stt-port", type=int, default=None)
    parser.add_argument("--tts-host", default=None)
    parser.add_argument("--tts-port", type=int, default=None)
    parser.add_argument("--http-host", default=None)
    parser.add_argument("--http-port", type=int, default=None)
    parser.add_argument("--http-api-keys-file", default=None)
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--kokoro-model", default=None)
    parser.add_argument("--kokoro-default-voice", default=None)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """Override config values with CLI arguments (non-None)."""
    if args.stt_host is not None:
        cfg.wyoming.stt_host = args.stt_host
    if args.stt_port is not None:
        cfg.wyoming.stt_port = args.stt_port
    if args.tts_host is not None:
        cfg.wyoming.tts_host = args.tts_host
    if args.tts_port is not None:
        cfg.wyoming.tts_port = args.tts_port
    if args.http_host is not None:
        cfg.http.host = args.http_host
    if args.http_port is not None:
        cfg.http.port = args.http_port
    if args.http_api_keys_file is not None:
        cfg.http.api_keys_file = args.http_api_keys_file
    if args.whisper_model is not None:
        cfg.models.whisper = args.whisper_model
    if args.kokoro_model is not None:
        cfg.models.kokoro = args.kokoro_model
    if args.kokoro_default_voice is not None:
        cfg.models.kokoro_default_voice = args.kokoro_default_voice
    if args.log_level is not None:
        cfg.logging.level = args.log_level


def main(argv: list[str] | None = None) -> None:
    """Entry point for the wyoming-mlx console script."""
    args = _parse_args(argv)
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    api_keys = load_api_keys(cfg.http.api_keys_file)

    # Lazy import real backends only at runtime
    try:
        from wyoming_mlx.backends.mlx_stt import MLXWhisperBackend
        from wyoming_mlx.backends.mlx_tts import KokoroBackend
    except ImportError:
        log.error(
            "MLX backends not available (mlx, mlx-whisper, kokoro packages missing). "
            "Install the MLX dependencies to enable real models."
        )
        sys.exit(1)

    stt_backend = MLXWhisperBackend(model_id=cfg.models.whisper)
    tts_backend = KokoroBackend(model_id=cfg.models.kokoro, voice=cfg.models.kokoro_default_voice)

    asyncio.run(run_servers(cfg=cfg, stt=stt_backend, tts=tts_backend, api_keys=api_keys))


if __name__ == "__main__":
    main()
