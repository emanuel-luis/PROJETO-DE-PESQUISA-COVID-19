from pathlib import Path

import pytest

from covidxr.config import Config
from covidxr.contracts import CATALOG, REJECTS
from covidxr.io import read_table
from covidxr.layout import Layout
from covidxr.pipeline import run_pipeline


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_pipeline_is_reproducible(ci_cfg: Config, tmp_path: Path) -> None:
    tables = []
    for name in ("a", "b"):
        root = tmp_path / name
        run_pipeline(ci_cfg, root)
        layout = Layout(root)
        assert (layout.raw_dir("synthetic") / "_run.json").is_file()
        assert (layout.catalog_dir / "_run.json").is_file()
        tables.append(
            (
                read_table(layout.catalog_table, CATALOG),
                read_table(layout.catalog_rejects, REJECTS),
            )
        )
    assert tables[0][0].equals(tables[1][0])
    assert tables[0][1].equals(tables[1][1])
