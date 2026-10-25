"""train_cnn behaviour that needs no torch: skipping, the missing-extra error, the key."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest
from helpers import copy_dirs

from covidxr import cli
from covidxr.config import Config
from covidxr.contracts import PREDICTIONS
from covidxr.evaluate import run_evaluate
from covidxr.io import read_table
from covidxr.layout import Layout
from covidxr.split import run_split
from covidxr.training.baseline import run_train_baseline
from covidxr.training.cnn import checkpoint_key, run_train_cnn


def _cnn(cfg: Config, **changes: object) -> Config:
    return dataclasses.replace(
        cfg,
        train_cnn=dataclasses.replace(cfg.train_cnn, **changes),  # type: ignore[arg-type]
    )


def test_disabled_writes_skipped_outputs_and_evaluate_keeps_the_baseline(
    ci_cfg: Config, pipeline_root: Path, tmp_path: Path
) -> None:
    layout = copy_dirs(pipeline_root, tmp_path, "data/catalog", "data/features")
    run_split(ci_cfg, tmp_path)
    run_train_baseline(ci_cfg, tmp_path)
    run_train_cnn(_cnn(ci_cfg, enabled=False), tmp_path)  # needs neither torch nor image tables
    assert sorted(p.name for p in layout.models_cnn.iterdir()) == [
        "_run.json",
        "predictions.parquet",
    ]
    assert json.loads((layout.models_cnn / "_run.json").read_text(encoding="utf-8"))["skipped"]
    assert read_table(layout.models_cnn / "predictions.parquet", PREDICTIONS).empty
    run_evaluate(ci_cfg, tmp_path)
    metrics = json.loads(layout.metrics_json.read_text(encoding="utf-8"))
    assert list(metrics["models"]) == ["baseline"]


def test_missing_torch_is_a_one_line_error(
    ci_params: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` now raises ImportError
    assert cli.main(["--params", str(ci_params), "--root", str(tmp_path), "train-cnn"]) == 1
    err = capsys.readouterr().err
    assert "uv sync --extra cnn" in err and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert not (tmp_path / "models").exists()


def test_checkpoint_key_follows_params_and_inputs_but_not_num_workers(
    ci_cfg: Config, pipeline_root: Path
) -> None:
    layout = Layout(pipeline_root)
    key = checkpoint_key(ci_cfg, layout, "cpu")
    assert len(key) == 16
    assert checkpoint_key(_cnn(ci_cfg, num_workers=3), layout, "cpu") == key
    for changes in ({"lr": 0.1}, {"epochs": 5}, {"device": "cuda"}, {"arch": "efficientnet_b0"}):
        assert checkpoint_key(_cnn(ci_cfg, **changes), layout, "cpu") != key
    assert checkpoint_key(dataclasses.replace(ci_cfg, seed=1), layout, "cpu") != key
    assert checkpoint_key(ci_cfg, layout, "cuda") != key  # a CUDA checkpoint never resumes on CPU


def test_checkpoint_key_follows_the_processed_run(
    ci_cfg: Config, pipeline_root: Path, tmp_path: Path
) -> None:
    layout = copy_dirs(pipeline_root, tmp_path, "data/splits", "data/processed")
    key = checkpoint_key(ci_cfg, layout, "cpu")
    run = layout.processed_dir / "_run.json"
    run.write_text(run.read_text(encoding="utf-8") + " ", encoding="utf-8")  # same tables
    assert checkpoint_key(ci_cfg, layout, "cpu") != key
