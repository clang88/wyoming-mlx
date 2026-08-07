"""wyoming-mlx: Apple-Silicon-native TTS/STT for Home Assistant and OpenAI-compatible clients."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wyoming-mlx")
except PackageNotFoundError:
    # Running from a source checkout without an installed distribution (e.g. some editable-install
    # edge cases); keep this in sync with the [project].version fallback in pyproject.toml.
    __version__ = "1.0.0"
