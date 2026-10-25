from pathlib import Path

import pytest

from covidxr.config import Config
from covidxr.contracts import CATALOG, REJECTS, features_contract
from covidxr.features import feature_columns
from covidxr.io import read_table
from covidxr.layout import Layout
from covidxr.pipeline import run_pipeline


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_pipeline_is_reproducible(ci_cfg: Config, tmp_path: Path) -> None:
    tables = []
    contract = features_contract(feature_columns(ci_cfg.features))
    for name in ("a", "b"):
        root = tmp_path / name
        run_pipeline(ci_cfg, root)
        layout = Layout(root)
        for stage_dir in (
            layout.raw_dir("synthetic"),
            layout.catalog_dir,
            layout.processed_dir,
            layout.features_dir,
        ):
            assert (stage_dir / "_run.json").is_file()
        assert layout.quality_json.is_file()
        tables.append(
            (
                read_table(layout.catalog_table, CATALOG),
                read_table(layout.catalog_rejects, REJECTS),
                read_table(layout.features_table, contract),
                layout.quality_json.read_bytes(),
            )
        )
    assert tables[0][0].equals(tables[1][0])
    assert tables[0][1].equals(tables[1][1])
    assert tables[0][2].equals(tables[1][2])
    assert tables[0][3] == tables[1][3]
