from pathlib import Path

import pandas as pd
import pytest

from covidxr.catalog import CatalogEntry, build_catalog
from covidxr.config import Config, SyntheticConfig
from covidxr.errors import ImageDecodeError
from covidxr.imaging import read_gray
from covidxr.io import sha256_file
from covidxr.sources.synthetic import SyntheticSource


def _source(cfg: Config) -> SyntheticSource:
    scfg = cfg.ingest.sources["synthetic"]
    assert isinstance(scfg, SyntheticConfig)
    return SyntheticSource(scfg, cfg.seed)


def test_fetch_is_byte_deterministic(ci_cfg: Config, tmp_path: Path) -> None:
    src = _source(ci_cfg)
    a, b = tmp_path / "a", tmp_path / "b"
    ma = src.fetch(a, tmp_path)
    mb = src.fetch(b, tmp_path)
    assert ma.files == mb.files
    for rel in ma.files:
        assert (a / rel).read_bytes() == (b / rel).read_bytes()
    other = SyntheticSource(src.cfg, ci_cfg.seed + 1)
    assert other.fetch(tmp_path / "c", tmp_path).files != ma.files
    assert other.version() != src.version()


def test_counts_labels_and_manifest(ci_cfg: Config, tmp_path: Path) -> None:
    src = _source(ci_cfg)
    m = src.fetch(tmp_path, tmp_path)
    assert m.source == "synthetic"
    assert m.version == src.version() and m.version.startswith("synthetic-")
    assert m.url == "synthetic://generated" and m.archive_sha256 is None
    assert m.file_count == len(m.files) == 127
    assert "labels.csv" in m.files
    for rel, digest in m.files.items():
        assert sha256_file(tmp_path / rel) == digest

    table = pd.read_csv(tmp_path / "labels.csv", dtype=str)
    assert list(table.columns) == ["relpath", "label", "patient_id"]
    assert table["relpath"].is_monotonic_increasing and len(table) == 126
    imgs = src.list_images(tmp_path)
    assert [i.relpath for i in imgs] == list(table["relpath"])
    assert all("\\" not in i.relpath for i in imgs)

    regular = table[~table["relpath"].str.contains("dup_|conflict_|corrupt_")]
    assert regular["label"].value_counts().to_dict() == {"covid": 60, "normal": 60}
    assert regular["patient_id"].nunique() == 40
    assert regular["patient_id"].str.match(r"^(covid|normal)-p\d{3}$").all()


def test_injections(ci_cfg: Config, tmp_path: Path) -> None:
    src = _source(ci_cfg)
    src.fetch(tmp_path, tmp_path)
    table = pd.read_csv(tmp_path / "labels.csv", dtype=str).set_index("relpath")
    regular = {
        sha256_file(tmp_path / r): r
        for r in table.index
        if not any(k in r for k in ("dup_", "conflict_", "corrupt_"))
    }
    dups = [r for r in table.index if "/dup_" in r]
    assert len(dups) == 3
    for rel in dups:
        original = regular[sha256_file(tmp_path / rel)]
        assert table.loc[rel, "label"] == table.loc[original, "label"]
        assert table.loc[rel, "patient_id"] == table.loc[original, "patient_id"]

    conflicts = [r for r in table.index if "/conflict_" in r]
    assert conflicts == ["images/normal/conflict_000.png"]
    original = regular[sha256_file(tmp_path / conflicts[0])]
    assert table.loc[original, "label"] == "covid"
    assert table.loc[conflicts[0], "label"] == "normal"

    corrupt = [r for r in table.index if "/corrupt_" in r]
    assert corrupt == ["images/covid/corrupt_000.png", "images/covid/corrupt_001.png"]
    for rel in corrupt:
        with pytest.raises(ImageDecodeError):
            read_gray(tmp_path / rel)


def test_images_are_non_square_within_bounds(ci_cfg: Config, tmp_path: Path) -> None:
    src = _source(ci_cfg)
    src.fetch(tmp_path, tmp_path)
    for img in src.list_images(tmp_path)[:20]:
        if "corrupt_" in img.relpath:
            continue
        h, w = read_gray(tmp_path / img.relpath).shape
        assert h != w and 48 <= min(h, w) and max(h, w) <= 96


def test_catalog_has_only_injected_duplicates(ci_cfg: Config, tmp_path: Path) -> None:
    src = _source(ci_cfg)
    raw = tmp_path / "raw" / "synthetic"
    src.fetch(raw, tmp_path)
    entries = [CatalogEntry("synthetic", raw / i.relpath, i) for i in src.list_images(raw)]
    catalog, rejects, stats = build_catalog(entries, ci_cfg.catalog, ["synthetic"], tmp_path)
    assert stats["n_dup_clusters"] == 0
    assert stats["max_group_size"] == 3
    assert stats["n_exact_duplicate_rows_collapsed"] == 3
    assert stats["n_rejects"] == {"label_conflict": 2, "undecodable": 2}
    assert len(catalog) == 119 and len(rejects) == 4
