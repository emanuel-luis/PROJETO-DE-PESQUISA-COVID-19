"""Helpers shared by the source adapters."""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from covidxr.errors import SourceUnavailableError
from covidxr.io import sha256_file

log = logging.getLogger(__name__)

# Indirection so tests can turn the backoff into a no-op.
_sleep: Callable[[float], None] = time.sleep


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_files(root: Path) -> dict[str, str]:
    """relpath (POSIX) -> sha256 of every file under root, sorted by relpath."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def with_retries[T](
    fn: Callable[[], T], *, what: str, attempts: int = 4, base_delay_s: float = 2.0
) -> T:
    """Call fn; retry SourceUnavailableError with exponential backoff (2, 4, 8 s by default).

    Downloads hit transient network errors (TLS resets, timeouts); anything else is raised at once.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except SourceUnavailableError as e:
            if attempt == attempts:
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            log.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.0f s",
                what,
                attempt,
                attempts,
                e,
                delay,
            )
            _sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
