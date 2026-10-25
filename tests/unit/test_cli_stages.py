from pathlib import Path

import pytest
import yaml

from covidxr import cli
from covidxr.layout import Layout


def _argv(params: Path, root: Path, *rest: str) -> list[str]:
    return ["--params", str(params), "--root", str(root), *rest]


def test_ingest_catalog_run_all(ci_params: Path, tmp_path: Path) -> None:
    layout = Layout(tmp_path)
    assert cli.main(_argv(ci_params, tmp_path, "ingest", "--source", "synthetic")) == 0
    assert (layout.raw_dir("synthetic") / "_manifest.json").is_file()
    assert cli.main(_argv(ci_params, tmp_path, "ingest", "--source", "synthetic", "--force")) == 0
    assert cli.main(_argv(ci_params, tmp_path, "catalog")) == 0
    assert layout.catalog_table.is_file()

    other = tmp_path / "other"
    assert cli.main(_argv(ci_params, other, "run-all")) == 0
    assert Layout(other).catalog_rejects.is_file()


def test_catalog_before_ingest_fails_cleanly(
    ci_params: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_argv(ci_params, tmp_path, "catalog")) == 1
    err = capsys.readouterr().err
    assert "covidxr ingest --source synthetic" in err and "Traceback" not in err


def test_kaggle_without_credentials(
    ci_params: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    params = tmp_path / "params.yaml"
    data = yaml.safe_load(ci_params.read_text(encoding="utf-8"))
    data["ingest"] = {"sources": {"kaggle_radiography": {"handle": "a/b", "version": 1}}}
    params.write_text(yaml.safe_dump(data), encoding="utf-8")
    argv = _argv(params, tmp_path / "root", "ingest", "--source", "kaggle_radiography")
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "KAGGLE_API_TOKEN" in err and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert not list((tmp_path / "root" / "data" / "raw").glob(".tmp-*"))
