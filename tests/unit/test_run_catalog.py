import json
from pathlib import Path

import pytest

from covidxr.catalog import run_catalog
from covidxr.config import Config
from covidxr.contracts import CATALOG, REJECTS
from covidxr.errors import SourceUnavailableError
from covidxr.ingest import run_ingest
from covidxr.io import read_table
from covidxr.layout import Layout


def test_run_catalog_on_synthetic(ci_cfg: Config, tmp_path: Path) -> None:
    run_ingest(ci_cfg, tmp_path, "synthetic")
    run_catalog(ci_cfg, tmp_path)
    layout = Layout(tmp_path)

    catalog = read_table(layout.catalog_table, CATALOG)
    rejects = read_table(layout.catalog_rejects, REJECTS)
    assert len(catalog) == 119
    assert catalog["image_id"].is_monotonic_increasing
    assert catalog["path"].str.startswith("data/raw/synthetic/images/").all()
    assert rejects["reason"].value_counts().to_dict() == {"label_conflict": 2, "undecodable": 2}
    assert (rejects["stage"] == "catalog").all()

    run = json.loads((layout.catalog_dir / "_run.json").read_text(encoding="utf-8"))
    assert run["stage"] == "catalog"
    assert (run["input_rows"], run["output_rows"], run["rejects"]) == (126, 119, 4)
    assert run["n_exact_duplicate_rows_collapsed"] == 3 and run["max_group_size"] == 3
    assert [p.name for p in layout.catalog_dir.parent.iterdir() if p.name != "raw"] == ["catalog"]


def test_run_catalog_replaces_previous_output(ci_cfg: Config, tmp_path: Path) -> None:
    run_ingest(ci_cfg, tmp_path, "synthetic")
    run_catalog(ci_cfg, tmp_path)
    first = Layout(tmp_path).catalog_table.read_bytes()
    run_catalog(ci_cfg, tmp_path)
    assert Layout(tmp_path).catalog_table.read_bytes() == first


def test_run_catalog_requires_ingest(ci_cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(SourceUnavailableError, match="covidxr ingest --source synthetic"):
        run_catalog(ci_cfg, tmp_path)
    assert not Layout(tmp_path).catalog_dir.exists()
