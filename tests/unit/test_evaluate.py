import json
from pathlib import Path

import pytest
from helpers import copy_dirs, empty_cnn

from covidxr import cli
from covidxr.config import Config
from covidxr.contracts import EVAL, PREDICTIONS
from covidxr.errors import ContractError, SourceUnavailableError
from covidxr.evaluate import binary_metrics, run_evaluate
from covidxr.io import read_table, write_table
from covidxr.layout import Layout
from covidxr.split import run_split
from covidxr.training.baseline import run_train_baseline


def test_binary_metrics_on_a_known_confusion_matrix() -> None:
    y_true = [1] * 5 + [0] * 5
    y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.1]
    assert binary_metrics(y_true, y_score, 0.5) == {
        "n": 10,
        "accuracy": 0.7,
        "precision": 0.75,
        "recall": 0.6,
        "specificity": 0.8,
        "f1": 0.6667,
        "roc_auc": 0.76,
        "confusion": {"tn": 4, "fp": 1, "fn": 2, "tp": 3},
    }


def test_threshold_is_inclusive_and_single_class_has_no_auc() -> None:
    m = binary_metrics([1, 1], [0.5, 0.2], 0.5)
    assert m["roc_auc"] is None and m["recall"] == 0.5
    assert m["confusion"] == {"tn": 0, "fp": 0, "fn": 1, "tp": 1}
    assert m["specificity"] == 0.0  # no negatives: defined as 0, never a division error


def test_run_evaluate(ci_cfg: Config, pipeline_root: Path, tmp_path: Path) -> None:
    layout = copy_dirs(pipeline_root, tmp_path, "data/catalog", "data/features")
    run_split(ci_cfg, tmp_path)
    run_train_baseline(ci_cfg, tmp_path)
    empty_cnn(layout)  # a skipped CNN: evaluate keeps only the baseline
    run_evaluate(ci_cfg, tmp_path)
    metrics = json.loads(layout.metrics_json.read_text(encoding="utf-8"))
    assert metrics["threshold"] == 0.5 and list(metrics["models"]) == ["baseline"]
    base = metrics["models"]["baseline"]
    assert set(base) == {"selected", "val", "test", "test_by_source"}
    assert list(base["test_by_source"]) == ["synthetic"]
    assert base["test_by_source"]["synthetic"] == base["test"]
    ev = read_table(layout.eval_table, EVAL)
    assert set(ev["family"]) == {"baseline"} and ev["model"].eq(base["selected"]).all()
    assert (ev["y_pred"] == (ev["y_score"] >= 0.5)).all()
    assert (layout.eval_dir / "_run.json").is_file()
    first = layout.metrics_json.read_bytes()
    run_evaluate(ci_cfg, tmp_path)
    assert layout.metrics_json.read_bytes() == first


def test_missing_predictions(ci_cfg: Config, pipeline_root: Path, tmp_path: Path) -> None:
    copy_dirs(pipeline_root, tmp_path, "data/catalog", "data/features")
    with pytest.raises(SourceUnavailableError, match="covidxr split"):
        run_evaluate(ci_cfg, tmp_path)
    run_split(ci_cfg, tmp_path)
    with pytest.raises(SourceUnavailableError, match="train-baseline"):
        run_evaluate(ci_cfg, tmp_path)


def _baseline_root(cfg: Config, pipeline_root: Path, tmp_path: Path) -> Layout:
    layout = copy_dirs(pipeline_root, tmp_path, "data/catalog", "data/features")
    run_split(cfg, tmp_path)
    run_train_baseline(cfg, tmp_path)
    return layout


def test_missing_cnn_predictions(ci_cfg: Config, pipeline_root: Path, tmp_path: Path) -> None:
    _baseline_root(ci_cfg, pipeline_root, tmp_path)
    with pytest.raises(SourceUnavailableError, match="train-cnn"):
        run_evaluate(ci_cfg, tmp_path)


def test_every_table_empty(ci_cfg: Config, pipeline_root: Path, tmp_path: Path) -> None:
    layout = _baseline_root(ci_cfg, pipeline_root, tmp_path)
    empty_cnn(layout)
    table = layout.models_baseline / "predictions.parquet"
    write_table(PREDICTIONS.schema.empty_table().to_pandas(), table, PREDICTIONS)
    with pytest.raises(SourceUnavailableError, match="nothing to evaluate"):
        run_evaluate(ci_cfg, tmp_path)


def test_predictions_for_unknown_images(
    ci_cfg: Config,
    pipeline_root: Path,
    tmp_path: Path,
    repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _baseline_root(ci_cfg, pipeline_root, tmp_path)
    empty_cnn(layout)
    table = layout.models_baseline / "predictions.parquet"
    preds = read_table(table, PREDICTIONS)
    preds.loc[0, "image_id"] = "f" * 64
    write_table(preds, table, PREDICTIONS)
    with pytest.raises(ContractError, match="missing from data/splits"):
        run_evaluate(ci_cfg, tmp_path)
    assert (
        cli.main(
            ["--params", str(repo_root / "params.ci.yaml"), "--root", str(tmp_path), "evaluate"]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1
