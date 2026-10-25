"""Source protocol and its data types."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal, Protocol


@dataclass(frozen=True)
class RawImage:
    relpath: str  # POSIX, relative to the source raw dir
    label: Literal["covid", "normal"]
    patient_id: str | None


@dataclass(frozen=True)
class Manifest:
    source: str
    version: str
    url: str
    license: str
    citation: str
    fetched_at: str
    file_count: int
    archive_sha256: str | None
    files: dict[str, str]  # relpath -> sha256

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        """Parse a manifest; raises ValueError on malformed content."""
        try:
            return cls(**json.loads(text))
        except (TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"invalid manifest: {e}") from e


class Source(Protocol):
    name: ClassVar[str]

    def version(self) -> str: ...

    def fetch(self, tmp_dir: Path, cache_dir: Path) -> Manifest:
        """Network allowed; fills tmp_dir."""
        ...

    def list_images(self, raw_dir: Path) -> list[RawImage]:
        """Filesystem only; sorted by relpath."""
        ...
