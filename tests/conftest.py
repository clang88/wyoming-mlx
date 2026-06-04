"""Test configuration for wyoming-mlx.

Integration tests (marked with -m "integration") are skipped by default.
Run them with: uv run pytest --integration
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require real MLX models",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip("Run with --integration flag to execute integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
