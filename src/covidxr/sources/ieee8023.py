"""ieee8023/covid-chestxray-dataset, fetched as a GitHub codeload tarball."""

import logging
import tarfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal, cast

import pandas as pd

from covidxr.config import Ieee8023Config
from covidxr.errors import SourceUnavailableError
from covidxr.io import atomic_path, sha256_file
from covidxr.sources._common import hash_files, now_iso, with_retries
from covidxr.sources.base import Manifest, RawImage

log = logging.getLogger(__name__)

_TIMEOUT_S = 60
_FINDINGS: dict[str, Literal["covid", "normal"]] = {
    "Pneumonia/Viral/COVID-19": "covid",
    "No Finding": "normal",
}


def _download(url: str, dest: Path) -> None:
    """Stream url into dest in 1 MiB chunks. Tests monkeypatch this function."""
    req = urllib.request.Request(url, headers={"User-Agent": "covidxr/0.1 (research pipeline)"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp, dest.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SourceUnavailableError(f"cannot download {url}: {e}") from e


def _extract(archive: Path, tmp_dir: Path) -> None:
    """Extract metadata.csv and images/** below the top-level directory, stripping it."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                parts = PurePosixPath(member.name).parts
                if member.name.startswith("/") or ".." in parts or len(parts) < 2:
                    continue
                rest = parts[1:]
                if rest != ("metadata.csv",) and rest[0] != "images":
                    continue
                if not (member.isfile() or member.isdir()):
                    continue
                member.name = "/".join(rest)
                tar.extract(member, tmp_dir, filter="data")
    except (tarfile.TarError, EOFError, OSError) as e:
        raise SourceUnavailableError(f"cannot extract {archive.name}: {e}") from e


class Ieee8023Source:
    name: ClassVar[str] = "ieee8023"

    def __init__(self, cfg: Ieee8023Config) -> None:
        self.cfg = cfg

    def version(self) -> str:
        return f"{self.cfg.repo}@{self.cfg.ref}"

    def _url(self) -> str:
        return f"https://codeload.github.com/{self.cfg.repo}/tar.gz/{self.cfg.ref}"

    def fetch(self, tmp_dir: Path, cache_dir: Path) -> Manifest:
        archive = cache_dir / self.name / f"{self.cfg.ref}.tar.gz"
        if archive.is_file():
            log.info("using cached archive %s", archive)
        else:

            def attempt() -> None:
                with atomic_path(archive) as part:
                    _download(self._url(), part)

            with_retries(attempt, what=f"download {self._url()}")
        _extract(archive, tmp_dir)
        files = hash_files(tmp_dir)
        return Manifest(
            source=self.name,
            version=self.version(),
            url=self._url(),
            license="Metadata CC BY-NC-SA 4.0; per-image licenses in metadata.csv column 'license'",
            citation="Cohen et al., COVID-19 Image Data Collection, arXiv:2006.11988 (2020)",
            fetched_at=now_iso(),
            file_count=len(files),
            archive_sha256=sha256_file(archive),
            files=files,
        )

    def list_images(self, raw_dir: Path) -> list[RawImage]:
        df = pd.read_csv(raw_dir / "metadata.csv", dtype=str).drop_duplicates()
        df = df[df["filename"].notna()]
        keep = (
            (df["folder"] == "images")
            & (df["modality"] == "X-ray")
            & df["view"].isin(self.cfg.views)
            & df["finding"].isin(list(_FINDINGS))
        )
        df = df[keep]
        out = [
            RawImage(
                f"images/{r.filename}",
                _FINDINGS[cast(str, r.finding)],
                None if pd.isna(r.patientid) else str(r.patientid),
            )
            for r in df.itertuples(index=False)
        ]
        return sorted(out, key=lambda i: i.relpath)
