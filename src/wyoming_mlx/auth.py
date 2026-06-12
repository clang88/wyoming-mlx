from __future__ import annotations

import logging
import stat
from pathlib import Path

log = logging.getLogger(__name__)


def load_api_keys(path: Path | str) -> set[str]:
    """Load API keys from a file (one per line; '#' comments and blank lines ignored).

    Returns an empty set if the file does not exist (a deployment may legitimately
    run with the HTTP API effectively closed).  Emits a WARNING if the file is
    readable by group/other.
    """
    p = Path(path).expanduser()
    if not p.exists():
        log.warning("API keys file %s does not exist; HTTP endpoints will reject all requests", p)
        return set()

    mode = p.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        log.warning(
            "API keys file %s has loose mode %o; should be 0600",
            p,
            stat.S_IMODE(mode),
        )

    keys: set[str] = set()
    for line in p.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keys.add(stripped)
    if not keys:
        log.warning("API keys file %s contains no keys; HTTP endpoints will reject all requests", p)
    return keys
