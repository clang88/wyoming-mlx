# wyoming-mlx

Apple-Silicon-native TTS (Kokoro) and STT (distil-whisper) for Home Assistant
and OpenAI-compatible clients.

Speech-to-text runs on [MLX](https://github.com/ml-explore/mlx) via
`mlx-whisper`; text-to-speech runs Kokoro on Metal via PyTorch. The real
backends therefore require an Apple Silicon Mac. Everything else (config,
HTTP API, Wyoming protocol handling, fake backends, tests) is portable, and
CI runs on Linux against the fake backends.

## Quick start (dev)

```bash
mise install
uv sync
uv run pytest
```

## Run locally (fake backends)

```bash
uv run python scripts/dev_run.py
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

### API keys

HTTP endpoints require a bearer token. Keys are read at startup from
`~/.config/wyoming-mlx/apikeys` (override with `--http-api-keys-file`),
one key per line, `#` comments allowed. The file should be mode `0600`.
If the file is missing or empty, all HTTP requests are rejected with 401.

```bash
mkdir -p ~/.config/wyoming-mlx
(umask 077; openssl rand -hex 32 > ~/.config/wyoming-mlx/apikeys)
```

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

## License

[Apache-2.0](LICENSE)
