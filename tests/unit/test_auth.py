from pathlib import Path

import pytest

from wyoming_mlx.auth import load_api_keys


def test_returns_empty_set_when_file_missing(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    with caplog.at_level("WARNING"):
        keys = load_api_keys(tmp_path / "nope")
    assert keys == set()
    assert any("does not exist" in rec.message for rec in caplog.records)


def test_parses_multiple_keys(tmp_path: Path):
    f = tmp_path / "keys"
    f.write_text("key1\nkey2\n  key3  \n\n# a comment line, ignored\nkey4\n")
    f.chmod(0o600)
    assert load_api_keys(f) == {"key1", "key2", "key3", "key4"}


def test_warns_on_loose_permissions(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    f = tmp_path / "keys"
    f.write_text("k\n")
    f.chmod(0o644)
    with caplog.at_level("WARNING"):
        load_api_keys(f)
    assert any("mode" in rec.message.lower() for rec in caplog.records)


def test_expands_user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    f = tmp_path / "keys"
    f.write_text("k1\n")
    f.chmod(0o600)
    assert load_api_keys("~/keys") == {"k1"}


def test_empty_file_returns_empty_set(tmp_path: Path):
    f = tmp_path / "keys"
    f.write_text("")
    assert load_api_keys(f) == set()


def test_warns_when_file_has_no_keys(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    f = tmp_path / "keys"
    f.write_text("# comments only, no keys\n\n")
    f.chmod(0o600)
    with caplog.at_level("WARNING"):
        assert load_api_keys(f) == set()
    assert any("no keys" in rec.message.lower() for rec in caplog.records)
