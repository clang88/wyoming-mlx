from __future__ import annotations

from fastapi import FastAPI

from wyoming_mlx.backends.base import STTBackend, TTSBackend
from wyoming_mlx.config import ModelsConfig
from wyoming_mlx.http.routes import build_router


def create_app(
    stt: STTBackend,
    tts: TTSBackend,
    api_keys: set[str],
    models: ModelsConfig,
) -> FastAPI:
    app = FastAPI(title="wyoming-mlx", version="0.1.0")
    app.include_router(
        build_router(
            stt=stt,
            tts=tts,
            api_keys=api_keys,
            models=models,
        )
    )
    return app
