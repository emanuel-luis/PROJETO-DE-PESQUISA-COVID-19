"""Kaggle access shared by the Kaggle adapters. Credentials are only checked for presence."""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

from covidxr.config import KaggleConfig
from covidxr.errors import SourceUnavailableError
from covidxr.sources._common import hash_files, now_iso
from covidxr.sources.base import Manifest, RawImage

Label = Literal["covid", "normal"]
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def kaggle_credentials_available() -> bool:
    """True when a Kaggle credential source exists. Never reads or logs a value."""
    env = os.environ
    if env.get("KAGGLE_API_TOKEN") or (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")):
        return True
    candidates = [Path.home() / ".kaggle" / "kaggle.json", Path.home() / ".kaggle" / "access_token"]
    if env.get("KAGGLE_CONFIG_DIR"):
        candidates.append(Path(env["KAGGLE_CONFIG_DIR"]) / "kaggle.json")
    return any(p.is_file() for p in candidates)


def require_credentials(source_name: str) -> None:
    if not kaggle_credentials_available():
        raise SourceUnavailableError(
            "set KAGGLE_API_TOKEN (or KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle/kaggle.json), "
            f"or remove {source_name} from ingest.sources in params.yaml"
        )


def download(handle: str, version: int, cache_dir: Path, *, source: str) -> Path:
    """Download a dataset version with kagglehub into cache_dir/<source>/v<version>."""
    try:
        import kagglehub  # type: ignore[import-untyped]

        out = cache_dir / source / f"v{version}"
        out.mkdir(parents=True, exist_ok=True)
        path = kagglehub.dataset_download(f"{handle}/versions/{version}", output_dir=str(out))
    except Exception as e:
        raise SourceUnavailableError(f"kaggle download of {handle} v{version} failed: {e}") from e
    return Path(path)


class KaggleSource(ABC):
    name: ClassVar[str]
    license: ClassVar[str]
    citation: ClassVar[str]

    def __init__(self, cfg: KaggleConfig) -> None:
        self.cfg = cfg

    def version(self) -> str:
        return f"{self.cfg.handle}/versions/{self.cfg.version}"

    @staticmethod
    @abstractmethod
    def classify(parts: tuple[str, ...]) -> Label | None:
        """Label for a file given its path parts relative to the dataset root, or None."""

    def _scan(self, root: Path) -> list[RawImage]:
        out: list[RawImage] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                rel = p.relative_to(root)
                label = self.classify(rel.parts)
                if label is not None:
                    out.append(RawImage(rel.as_posix(), label, None))
        return sorted(out, key=lambda i: i.relpath)

    def fetch(self, tmp_dir: Path, cache_dir: Path) -> Manifest:
        require_credentials(self.name)
        src = download(self.cfg.handle, self.cfg.version, cache_dir, source=self.name)
        for img in self._scan(src):
            dest = tmp_dir / img.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / img.relpath, dest)
        files = hash_files(tmp_dir)
        return Manifest(
            source=self.name,
            version=self.version(),
            url=f"https://www.kaggle.com/datasets/{self.cfg.handle}/versions/{self.cfg.version}",
            license=self.license,
            citation=self.citation,
            fetched_at=now_iso(),
            file_count=len(files),
            archive_sha256=None,
            files=files,
        )

    def list_images(self, raw_dir: Path) -> list[RawImage]:
        return self._scan(raw_dir)
