from pathlib import Path

import pytest

from covidxr.config import KaggleConfig
from covidxr.errors import SourceUnavailableError
from covidxr.sources import kaggle
from covidxr.sources.kaggle import kaggle_credentials_available, require_credentials
from covidxr.sources.kaggle_covidqu import KaggleCovidquSource
from covidxr.sources.kaggle_radiography import KaggleRadiographySource

RADIO = "COVID-19_Radiography_Dataset"
LUNG = "Lung Segmentation Data/Lung Segmentation Data"
INFECTION = "Infection Segmentation Data/Infection Segmentation Data"


@pytest.fixture(autouse=True)
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _touch(root: Path, rels: list[str]) -> None:
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())


def test_no_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    assert not kaggle_credentials_available()
    with pytest.raises(SourceUnavailableError) as exc:
        require_credentials("kaggle_radiography")
    msg = str(exc.value)
    assert "KAGGLE_API_TOKEN" in msg and "remove kaggle_radiography from ingest.sources" in msg


def test_presence_detection(
    clean_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "dummy-user")
    assert not kaggle_credentials_available()  # the key is missing
    monkeypatch.setenv("KAGGLE_KEY", "dummy-key")
    assert kaggle_credentials_available()
    monkeypatch.delenv("KAGGLE_USERNAME")
    monkeypatch.delenv("KAGGLE_KEY")

    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-token")
    assert kaggle_credentials_available()
    monkeypatch.delenv("KAGGLE_API_TOKEN")

    for name in ("kaggle.json", "access_token"):
        f = clean_env / ".kaggle" / name
        _touch(clean_env, [f".kaggle/{name}"])
        assert kaggle_credentials_available()
        f.unlink()
    assert not kaggle_credentials_available()

    cfg_dir = tmp_path / "cfg"
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(cfg_dir))
    assert not kaggle_credentials_available()
    _touch(cfg_dir, ["kaggle.json"])
    assert kaggle_credentials_available()


def test_radiography_layout(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        [
            f"{RADIO}/COVID/images/COVID-1.png",
            f"{RADIO}/COVID/masks/COVID-1.png",
            f"{RADIO}/Normal/images/Normal-1.png",
            f"{RADIO}/Normal/masks/Normal-1.png",
            f"{RADIO}/Lung_Opacity/images/LO-1.png",
            f"{RADIO}/Viral Pneumonia/images/VP-1.png",
            f"{RADIO}/COVID.metadata.xlsx",
        ],
    )
    src = KaggleRadiographySource(KaggleConfig("x/y", 5))
    assert src.version() == "x/y/versions/5"
    assert [(i.relpath, i.label, i.patient_id) for i in src.list_images(tmp_path)] == [
        (f"{RADIO}/COVID/images/COVID-1.png", "covid", None),
        (f"{RADIO}/Normal/images/Normal-1.png", "normal", None),
    ]


def test_covidqu_layout(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        [
            f"Extra/{LUNG}/Train/COVID-19/images/c1.png",
            f"Extra/{LUNG}/Train/COVID-19/lung masks/c1.png",
            f"Extra/{LUNG}/Val/Normal/images/n1.png",
            f"Extra/{LUNG}/Val/Non-COVID/images/x1.png",
            f"Extra/{INFECTION}/Test/COVID-19/images/c2.png",
            f"Extra/{INFECTION}/Test/COVID-19/infection masks/c2.png",
            f"Extra/{INFECTION}/Test/Normal/lung masks/n2.png",
            "Other/Train/COVID-19/images/stray.png",
        ],
    )
    src = KaggleCovidquSource(KaggleConfig("a/b", 3))
    assert [(i.relpath, i.label) for i in src.list_images(tmp_path)] == [
        (f"Extra/{INFECTION}/Test/COVID-19/images/c2.png", "covid"),
        (f"Extra/{LUNG}/Train/COVID-19/images/c1.png", "covid"),
        (f"Extra/{LUNG}/Val/Normal/images/n1.png", "normal"),
    ]


def test_fetch_copies_only_class_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloaded = tmp_path / "dl"
    _touch(
        downloaded,
        [
            f"{RADIO}/COVID/images/a.png",
            f"{RADIO}/COVID/masks/a.png",
            f"{RADIO}/Lung_Opacity/images/b.png",
            f"{RADIO}/Normal/images/c.png",
        ],
    )
    seen: list[tuple[str, int, str]] = []

    def fake_download(handle: str, version: int, cache_dir: Path, *, source: str) -> Path:
        seen.append((handle, version, source))
        return downloaded

    monkeypatch.setattr(kaggle, "download", fake_download)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-token")
    out = tmp_path / "out"
    m = KaggleRadiographySource(KaggleConfig("x/y", 5)).fetch(out, tmp_path / "cache")
    assert seen == [("x/y", 5, "kaggle_radiography")]
    assert sorted(m.files) == [f"{RADIO}/COVID/images/a.png", f"{RADIO}/Normal/images/c.png"]
    assert m.file_count == 2 and m.archive_sha256 is None
    assert m.version == "x/y/versions/5" and m.license.startswith("Kaggle:")
    assert not (out / RADIO / "COVID" / "masks").exists()


def test_fetch_without_credentials_does_not_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object, **__: object) -> Path:
        raise AssertionError("download must not run")

    monkeypatch.setattr(kaggle, "download", fail)
    with pytest.raises(SourceUnavailableError):
        KaggleCovidquSource(KaggleConfig("a/b", 1)).fetch(tmp_path / "o", tmp_path / "c")


def test_download_wraps_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import kagglehub  # type: ignore[import-untyped]

    def boom(*_: object, **__: object) -> str:
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(kagglehub, "dataset_download", boom)
    with pytest.raises(SourceUnavailableError, match="401 unauthorized"):
        kaggle.download("a/b", 1, tmp_path, source="s")
