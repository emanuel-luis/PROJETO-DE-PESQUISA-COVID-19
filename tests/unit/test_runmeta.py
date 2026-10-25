import json
import re
from pathlib import Path

import pytest

import covidxr
from covidxr.config import CatalogConfig, params_hash
from covidxr.runmeta import stage_run

ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_run_json_fields(tmp_path: Path) -> None:
    params = CatalogConfig(near_dup_max_hamming=4)
    with stage_run("catalog", tmp_path, params) as stats:
        stats.input_rows = 10
        stats.output_rows = 8
        stats.rejects = 2
        stats.extra["largest_group"] = 3
    raw = (tmp_path / "_run.json").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    meta = json.loads(raw)
    assert set(meta) == {
        "run_id",
        "stage",
        "started_at",
        "finished_at",
        "input_rows",
        "output_rows",
        "rejects",
        "params_hash",
        "package_version",
        "largest_group",
    }
    assert meta["stage"] == "catalog"
    assert (meta["input_rows"], meta["output_rows"], meta["rejects"]) == (10, 8, 2)
    assert re.fullmatch(r"[0-9a-f]{32}", meta["run_id"])
    assert ISO_Z.match(meta["started_at"]) and ISO_Z.match(meta["finished_at"])
    assert meta["started_at"] <= meta["finished_at"]
    assert meta["params_hash"] == params_hash(params)
    assert meta["package_version"] == covidxr.__version__


def test_nothing_written_on_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), stage_run("catalog", tmp_path, {}):
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []
