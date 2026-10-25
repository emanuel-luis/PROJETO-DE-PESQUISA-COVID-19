import json
from pathlib import Path

import pytest

from covidxr.config import Config
from covidxr.contracts import (
    CATALOG,
    EVAL,
    PREDICTIONS,
    REJECTS,
    SPLITS,
    features_contract,
)
from covidxr.features import feature_columns
from covidxr.io import read_table
from covidxr.layout import Layout
from covidxr.pipeline import run_pipeline


@pytest.mark.slow
@pytest.mark.timeout(300)
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
            layout.splits_dir,
            layout.models_baseline,
            layout.eval_dir,
        ):
            assert (stage_dir / "_run.json").is_file()
        assert layout.quality_json.is_file()
        tables.append(
            (
                read_table(layout.catalog_table, CATALOG),
                read_table(layout.catalog_rejects, REJECTS),
                read_table(layout.features_table, contract),
                layout.quality_json.read_bytes(),
                read_table(layout.splits_table, SPLITS),
                read_table(layout.models_baseline / "predictions.parquet", PREDICTIONS),
                read_table(layout.eval_table, EVAL),
                layout.metrics_json.read_bytes(),
            )
        )
        test_f1 = json.loads(layout.metrics_json.read_text(encoding="utf-8"))["models"]["baseline"][
            "test"
        ]["f1"]
        print(f"synthetic baseline test F1: {test_f1}")
        assert test_f1 >= 0.8
    for first, second in zip(*tables, strict=True):
        if isinstance(first, bytes):
            assert first == second
        else:
            assert first.equals(second)
