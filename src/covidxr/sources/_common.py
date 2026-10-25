"""Helpers shared by the source adapters."""

from datetime import UTC, datetime
from pathlib import Path

from covidxr.io import sha256_file


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_files(root: Path) -> dict[str, str]:
    """relpath (POSIX) -> sha256 of every file under root, sorted by relpath."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}
