import json
from pathlib import Path

import pytest

from covidxr.config import Config
from covidxr.errors import ConfigError
from covidxr.ingest import run_ingest
from covidxr.layout import Layout
from covidxr.sources.base import Manifest
from covidxr.sources.synthetic import SyntheticSource


def _no_fetch(*_: object, **__: object) -> Manifest:
    raise AssertionError("fetch must not run")


def test_ingest_writes_manifest_and_run(ci_cfg: Config, tmp_path: Path) -> None:
    m = run_ingest(ci_cfg, tmp_path, "synthetic")
    raw = Layout(tmp_path).raw_dir("synthetic")
    assert Manifest.from_json((raw / "_manifest.json").read_text(encoding="utf-8")) == m
    assert m.file_count == 127 and (raw / "labels.csv").is_file()
    run = json.loads((raw / "_run.json").read_text(encoding="utf-8"))
    assert run["stage"] == "ingest" and run["output_rows"] == 127
    assert run["source"] == "synthetic" and run["version"] == m.version
    assert run["skipped"] is False
    assert not list(raw.parent.glob(".tmp-*"))


def test_second_call_skips_and_force_refetches(
    ci_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = run_ingest(ci_cfg, tmp_path, "synthetic")
    run_json = Layout(tmp_path).raw_dir("synthetic") / "_run.json"
    before = run_json.read_bytes()

    with monkeypatch.context() as mp:
        mp.setattr(SyntheticSource, "fetch", _no_fetch)
        assert run_ingest(ci_cfg, tmp_path, "synthetic") == first
    assert run_json.read_bytes() == before

    forced = run_ingest(ci_cfg, tmp_path, "synthetic", force=True)
    assert forced.files == first.files
    assert run_json.read_bytes() != before


def test_skip_restores_missing_run_json(
    ci_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ingest(ci_cfg, tmp_path, "synthetic")
    run_json = Layout(tmp_path).raw_dir("synthetic") / "_run.json"
    run_json.unlink()
    monkeypatch.setattr(SyntheticSource, "fetch", _no_fetch)
    run_ingest(ci_cfg, tmp_path, "synthetic")
    assert json.loads(run_json.read_text(encoding="utf-8"))["skipped"] is True


def test_missing_file_triggers_refetch(ci_cfg: Config, tmp_path: Path) -> None:
    m = run_ingest(ci_cfg, tmp_path, "synthetic")
    raw = Layout(tmp_path).raw_dir("synthetic")
    (raw / "images" / "covid" / "0000.png").unlink()
    run_ingest(ci_cfg, tmp_path, "synthetic")
    assert (raw / "images" / "covid" / "0000.png").is_file()
    assert m.file_count == 127


def test_failed_fetch_keeps_destination(
    ci_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ingest(ci_cfg, tmp_path, "synthetic")
    raw = Layout(tmp_path).raw_dir("synthetic")
    snapshot = {
        p.relative_to(raw).as_posix(): p.read_bytes() for p in raw.rglob("*") if p.is_file()
    }

    def failing(self: SyntheticSource, tmp_dir: Path, cache_dir: Path) -> Manifest:
        (tmp_dir / "partial.txt").write_text("x")
        raise RuntimeError("boom")

    monkeypatch.setattr(SyntheticSource, "fetch", failing)
    with pytest.raises(RuntimeError, match="boom"):
        run_ingest(ci_cfg, tmp_path, "synthetic", force=True)
    assert not list(raw.parent.glob(".tmp-*"))
    now = {p.relative_to(raw).as_posix(): p.read_bytes() for p in raw.rglob("*") if p.is_file()}
    assert now == snapshot


def test_unknown_source(ci_cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not in ingest.sources"):
        run_ingest(ci_cfg, tmp_path, "ieee8023")
    assert not (tmp_path / "data").exists()
