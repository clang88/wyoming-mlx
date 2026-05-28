# wyoming-mlx

Apple-Silicon-native TTS (Kokoro) and STT (distil-whisper) for Home Assistant
and OpenAI-compatible clients.

## Quick start (dev)

```bash
mise install
uv sync
uv run pytest
```

## Run locally (fake backends)

```bash
uv run python _dev_run.py
```

## Run locally (real MLX backends)

```bash
uv run wyoming-mlx
```

By default it loads:
- distil-whisper-large-v3 (MLX) on Wyoming port 10300 / HTTP `/v1/audio/transcriptions`
- Kokoro-82M (MLX) on Wyoming port 10200 / HTTP `/v1/audio/speech`
- HTTP on port 10400 with API-key auth

Models download on first use to the Hugging Face cache.

## HTTP API

### List models

```bash
curl http://localhost:10400/v1/models
```

### Transcribe an audio file

```bash
curl http://localhost:10400/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@some.wav
```

### Synthesize speech

```bash
curl http://localhost:10400/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello there.","voice":"af_heart"}' \
  --output /tmp/out.wav
```

## Home Assistant integration

Settings → Integrations → Wyoming Protocol → Add:

- STT: `<host>:10300`
- TTS: `<host>:10200`

No keys, no TLS (HA convention, trusted LAN).

## Configuration

Pass `--config /path/to/config.toml` or set env vars with the
`WYOMING_MLX_` prefix and `__` for nesting (e.g.
`WYOMING_MLX_HTTP__PORT=10401`). See `src/wyoming_mlx/config.py` for the
full schema.

## Deployment to the the host

See `deploy/README.md` (created in Task 11) for the LaunchDaemon plist and
install steps.

## Smoke test

```bash
uv run python scripts/smoke.py --host <host> --api-key "$KEY"
```
