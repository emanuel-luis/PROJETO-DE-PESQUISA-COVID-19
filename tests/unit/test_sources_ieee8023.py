import io
import tarfile
import urllib.error
from pathlib import Path

import pytest

from covidxr.config import Ieee8023Config
from covidxr.errors import SourceUnavailableError
from covidxr.io import sha256_file
from covidxr.sources import ieee8023
from covidxr.sources.ieee8023 import Ieee8023Source

REF = "abc123"
TOP = f"covid-chestxray-dataset-{REF}"
HEADER = "patientid,finding,view,modality,folder,filename,license"
ROWS = [
    "1,Pneumonia/Viral/COVID-19,PA,X-ray,images,a.png,CC BY",
    "2,No Finding,AP,X-ray,images,b.jpg,CC BY",
    "3,Pneumonia/Viral/COVID-19,L,X-ray,images,c.png,CC BY",  # view not wanted
    "4,Pneumonia/Bacterial,PA,X-ray,images,e.png,CC BY",  # finding not mapped
    "5,Pneumonia/Viral/COVID-19,PA,CT,images,f.png,CC BY",  # modality
    "6,Pneumonia/Viral/COVID-19,PA,X-ray,volumes,v.nii.gz,CC BY",  # folder
    "1,Pneumonia/Viral/COVID-19,PA,X-ray,images,a.png,CC BY",  # exact duplicate row
    "1,Pneumonia/Viral/COVID-19,AP,X-ray,images,d.jpeg,CC BY",
]


def _add(tar: tarfile.TarFile, name: str, data: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def tarball(tmp_path: Path) -> Path:
    path = tmp_path / "src.tar.gz"
    csv = ("\n".join([HEADER, *ROWS]) + "\n").encode()
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, f"{TOP}/metadata.csv", csv)
        for n in ("a.png", "b.jpg", "c.png", "d.jpeg"):
            _add(tar, f"{TOP}/images/{n}", n.encode())
        _add(tar, f"{TOP}/volumes/v.nii.gz")
        _add(tar, f"{TOP}/README.md")
        _add(tar, "../evil.txt")
        _add(tar, "/abs-evil.txt")
        _add(tar, f"{TOP}/images/../../evil2.txt")
    return path


def _source() -> Ieee8023Source:
    return Ieee8023Source(Ieee8023Config("ieee8023/covid-chestxray-dataset", REF, ("PA", "AP")))


def test_fetch_extracts_only_wanted_members(
    tarball: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(tarball.read_bytes())

    monkeypatch.setattr(ieee8023, "_download", fake_download)
    src = _source()
    out = tmp_path / "work" / "tmp"
    cache = tmp_path / "work" / "cache"
    m = src.fetch(out, cache)

    assert sorted(m.files) == [
        "images/a.png",
        "images/b.jpg",
        "images/c.png",
        "images/d.jpeg",
        "metadata.csv",
    ]
    assert m.file_count == 5
    assert calls == [f"https://codeload.github.com/ieee8023/covid-chestxray-dataset/tar.gz/{REF}"]
    assert m.version == f"ieee8023/covid-chestxray-dataset@{REF}"
    archive = cache / "ieee8023" / f"{REF}.tar.gz"
    assert m.archive_sha256 == sha256_file(archive)
    assert [p.name for p in archive.parent.iterdir()] == [archive.name]
    for bad in ("evil.txt", "abs-evil.txt", "evil2.txt"):
        assert not list(tmp_path.rglob(bad))
    assert not (out / "volumes").exists() and not (out / "README.md").exists()

    # second fetch reuses the cached archive
    src.fetch(tmp_path / "again", cache)
    assert len(calls) == 1


def test_list_images_filters_and_maps(tarball: Path, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    with tarfile.open(tarball) as tar:
        member = tar.getmember(f"{TOP}/metadata.csv")
        member.name = "metadata.csv"
        tar.extract(member, raw, filter="data")
    imgs = _source().list_images(raw)
    assert [(i.relpath, i.label, i.patient_id) for i in imgs] == [
        ("images/a.png", "covid", "1"),
        ("images/b.jpg", "normal", "2"),
        ("images/d.jpeg", "covid", "1"),
    ]
    only_pa = Ieee8023Source(Ieee8023Config("r/r", REF, ("PA",))).list_images(raw)
    assert [i.relpath for i in only_pa] == ["images/a.png"]


def test_network_error_becomes_source_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(ieee8023.urllib.request, "urlopen", boom)
    with pytest.raises(SourceUnavailableError, match="offline"):
        _source().fetch(tmp_path / "t", tmp_path / "c")
    assert not (tmp_path / "c" / "ieee8023" / f"{REF}.tar.gz").exists()
